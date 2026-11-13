"""Review selected first-pass passages using Minerva perplexity and Qwen.

Use --prepare-only to inspect passage construction offline and --resume to
continue a run from matching checkpoints. Selective review uses a fixed
perplexity threshold.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import git_info, project_path, summarize_sources
from src.evaluation import load_reference, reference_passages
from src.language_models import (load_model, perplexity_batches, release_cuda_memory,
                                 require_cuda, review_passage)
from src.llm_filter import LABELS, merge_reviews
from src.passages import build_passages, select_passages, smoke_sample


def load_first_pass(path: Path) -> tuple[dict, str]:
    """Validate document identities and candidate spans before any model loads."""
    content = path.read_bytes()
    result = json.loads(content.decode("utf-8-sig"))
    if not isinstance(result, dict) or not isinstance(result.get("documents"), list) or not result["documents"]:
        raise ValueError("Input must be a first-pass JSON containing documents.")
    run = result.get("run")
    if not isinstance(run, dict) or run.get("stage") != "first_pass":
        raise ValueError("Input must contain valid first-pass run metadata.")
    seen = set()
    for document in result["documents"]:
        if not isinstance(document, dict):
            raise ValueError("Each input document must be an object.")
        identifier, text = document.get("id"), document.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Document identifiers must be non-empty and unique.")
        seen.add(identifier)
        if not isinstance(text, str) or not isinstance(document.get("entities"), list):
            raise ValueError(f"Document {identifier} requires text and entities.")
        for entity in document["entities"]:
            if not isinstance(entity, dict):
                raise ValueError(f"Invalid candidate object in {identifier}.")
            start, end = entity.get("start"), entity.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid entity offsets in {identifier}.")
            if (text[start:end] != entity.get("text") or not isinstance(entity.get("label"), str)
                    or entity["label"] not in LABELS):
                raise ValueError(f"Invalid entity text or category in {identifier}.")
            if (not isinstance(entity.get("source"), str) or not isinstance(entity.get("review_reasons"), list)
                    or not all(isinstance(reason, str) for reason in entity["review_reasons"])
                    or not isinstance(entity.get("needs_review"), bool)):
                raise ValueError(f"Missing candidate provenance in {identifier}.")
    return result, hashlib.sha256(content).hexdigest()


def validate_config(config: dict) -> None:
    """Validate resource limits and the fixed model settings used by this runner."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object.")
    for key in ("input_path", "output_path"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing configuration field: {key}")
    for section in ("passages", "perplexity", "routing", "llm"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    for role, limits in (("perplexity", ("max_tokens", "batch_size")),
                         ("llm", ("max_input_tokens", "max_new_tokens"))):
        settings = config[role]
        for key in ("model", "revision"):
            if not isinstance(settings.get(key), str) or not settings[key]:
                raise ValueError(f"Missing {role}.{key}")
        if not re.fullmatch(r"[a-f0-9]{40}", settings["revision"]):
            raise ValueError(f"{role}.revision must pin a full model commit hash.")
        for key in limits:
            if type(settings.get(key)) is not int or settings[key] < 1:
                raise ValueError(f"{role}.{key} must be a positive integer.")
    if config["llm"].get("quantization") != "nf4" or config["llm"].get("enable_thinking") is not False:
        raise ValueError("This runner uses NF4 quantization and disabled thinking.")
    if type(config["llm"].get("validation_retries")) is not int or not 0 <= config["llm"]["validation_retries"] <= 1:
        raise ValueError("llm.validation_retries must be zero or one.")
    if type(config["passages"].get("seed")) is not int:
        raise ValueError("passages.seed must be an integer.")
    if type(config["passages"].get("smoke_per_source")) is not int or config["passages"]["smoke_per_source"] < 1:
        raise ValueError("passages.smoke_per_source must be a positive integer.")


def code_hash() -> str:
    """Identify executed source files, including uncommitted local edits."""
    digest = hashlib.sha256()
    for name in ("scripts/run_hybrid.py", "scripts/run_pipeline.py", "src/passages.py",
                 "src/llm_filter.py", "src/language_models.py", "src/deterministic.py",
                 "src/evaluation.py"):
        digest.update(name.encode("utf-8"))
        # Git may check out CRLF on Windows; line-ending conversion is not a code change.
        digest.update((PROJECT_ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment_versions() -> dict:
    """Record installed inference versions without importing the GPU libraries."""
    versions = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def peak_host_memory() -> int | None:
    """Return peak process RAM on Colab/Linux, or None when unavailable."""
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def append_record(handle, record: dict) -> None:
    """Flush each completed inference record so a rerun can resume it."""
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def read_checkpoint(path: Path, fingerprint: str) -> list:
    """Read matching records, discarding only an incomplete trailing write."""
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        raise ValueError("Checkpoint is empty; choose another output or use --overwrite.")
    if not lines[0].endswith(b"\n"):
        raise ValueError("Checkpoint header is incomplete.")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or header.get("kind") != "header":
        raise ValueError("Checkpoint header is invalid.")
    if header.get("fingerprint") != fingerprint:
        raise ValueError("Checkpoint belongs to different data, code, settings or execution scope.")
    records = []
    valid_bytes = len(lines[0])
    for index, line in enumerate(lines[1:], start=1):
        if not line.endswith(b"\n"):
            if index != len(lines) - 1:
                raise ValueError(f"Incomplete checkpoint record at line {index + 1}.")
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
            print("Discarded an incomplete trailing checkpoint record.", flush=True)
            break
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Invalid checkpoint record at line {index + 1}.") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid checkpoint record at line {index + 1}.")
        records.append(record)
        valid_bytes += len(line)
    return records


def write_result(path: Path, result: dict) -> None:
    """Publish a complete JSON only after serialization succeeds."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def score_missing_passages(passages: list, settings: dict, scores: dict, models: dict, checkpoint, cuda) -> None:
    """Score pending passages with Minerva and save each completed score."""
    pending = [passage for passage in passages if passage["id"] not in scores]
    if not pending:
        return
    print(f"Loading Minerva; {len(pending)} passages to score...", flush=True)
    model, tokenizer, metadata = load_model(settings, quantized=False)
    try:
        for batch in perplexity_batches(pending, settings, model, tokenizer):
            for passage_id, score in batch.items():
                scores[passage_id] = score
                append_record(checkpoint, {"kind": "perplexity", "passage_id": passage_id, "result": score})
            print(f"Perplexity: {len(scores)}/{len(passages)}", flush=True)
        metadata["peak_gpu_allocated_bytes"] = cuda.max_memory_allocated()
        models["perplexity"] = metadata
        append_record(checkpoint, {"kind": "model", "role": "perplexity", "metadata": metadata})
    finally:
        del model, tokenizer
        release_cuda_memory()


def review_missing_passages(selected: list, settings: dict, reviews: dict, models: dict, checkpoint, cuda) -> None:
    """Run Qwen for selected passages that lack a saved review."""
    pending = [passage for passage in selected if passage["id"] not in reviews]
    if not pending:
        return
    print(f"Loading Qwen; {len(pending)} passages to review...", flush=True)
    model, tokenizer, metadata = load_model(settings, quantized=True)
    try:
        for passage in pending:
            passage_id = passage["id"]
            review = review_passage(passage, settings, model, tokenizer)
            reviews[passage_id] = review
            append_record(checkpoint, {"kind": "review", "passage_id": passage_id, "result": review})
            print(f"LLM: {len(reviews)}/{len(selected)}; {review['status']}; {passage_id}", flush=True)
        metadata["peak_gpu_allocated_bytes"] = cuda.max_memory_allocated()
        models["llm"] = metadata
        append_record(checkpoint, {"kind": "model", "role": "llm", "metadata": metadata})
    finally:
        del model, tokenizer
        release_cuda_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid.json")
    parser.add_argument("--input", help="First-pass JSON; no raw-data download is required.")
    parser.add_argument("--output", help="New result JSON; checkpoint path is derived from it.")
    parser.add_argument("--smoke-test", action="store_true", help="Test a few passages per collection.")
    parser.add_argument("--reference", help="Select only the operational cores of a reviewed reference.")
    parser.add_argument("--development-all", action="store_true",
                        help="Review every development reference core for offline routing calibration.")
    parser.add_argument("--compare-routing", action="store_true",
                        help="Review the union of fixed with/without-perplexity selections on a reference.")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true", help="Build passages without model dependencies.")
    stage.add_argument("--score-only", action="store_true", help="Calculate perplexity without loading Qwen.")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true", help="Continue matching local checkpoints.")
    recovery.add_argument("--overwrite", action="store_true", help="Replace this run and its checkpoint.")
    args = parser.parse_args()
    try:
        config_path = project_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("Configuration must be an object.")
        if args.input:
            config["input_path"] = args.input
        if args.output:
            config["output_path"] = args.output
        validate_config(config)
        reference, reference_hash = load_reference(project_path(args.reference)) if args.reference else (None, None)
        if args.smoke_test and reference:
            raise ValueError("Choose either a reference subset or an automatic smoke sample.")
        if args.development_all and (reference is None or reference["split"] != "development"):
            raise ValueError("--development-all requires a reviewed development reference, never test data.")
        if args.compare_routing and (reference is None or args.development_all or args.score_only or args.smoke_test):
            raise ValueError("--compare-routing requires a fixed-threshold reference run.")
        if not (args.smoke_test or args.prepare_only or args.score_only or args.development_all):
            select_passages([], {}, config["routing"])
        input_path, output_path = project_path(config["input_path"]), project_path(config["output_path"])
        checkpoint_path = output_path.with_suffix(".checkpoints.jsonl")
        protected = {input_path, config_path}
        if args.reference:
            protected.add(project_path(args.reference))
        if output_path in protected or output_path.suffix.lower() != ".json":
            raise ValueError("Choose a JSON output distinct from input and configuration.")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError("Output exists. Choose a new name or use --overwrite.")
        source, input_hash = load_first_pass(input_path)
        documents = source["documents"]
        passages = build_passages(documents, config["passages"]["core_chars"], config["passages"]["context_chars"])
        full_passage_count = len(passages)
        if reference:
            passages = reference_passages(reference, documents, passages, config["passages"])
            identifiers = {p["document_id"] for p in passages}
            documents = [document for document in documents if document["id"] in identifiers]
        if args.smoke_test:
            passages = smoke_sample(passages, config["passages"]["smoke_per_source"], config["passages"]["seed"])
            identifiers = {p["document_id"] for p in passages}
            documents = [document for document in documents if document["id"] in identifiers]
        if args.development_all:
            scope = "development_all_review"
        elif reference:
            scope = f"reference_{reference['split']}"
        elif args.smoke_test:
            scope = "smoke_test"
        else:
            scope = "extended"
        print(f"Scope: {scope}; {len(passages)}/{full_passage_count} passages; "
              f"{len(documents)}/{len(source['documents'])} documents represented.", flush=True)
        run = {"stage": "hybrid_preparation" if args.prepare_only else "hybrid",
               "scope": scope, "created_utc": datetime.now(timezone.utc).isoformat(),
               "config": config, "input_sha256": input_hash, "input_run": source["run"],
               "code_sha256": code_hash(), "git": git_info(),
               "reference_sha256": reference_hash,
               "input_document_count": len(source["documents"]), "document_count": len(documents),
               "full_passage_count": full_passage_count, "passage_count": len(passages),
               "passages_by_source": dict(Counter(p["source"] for p in passages)),
               "python_version": platform.python_version(), "platform": platform.platform()}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.prepare_only:
            write_result(output_path, {"run": run, "passages": passages})
            print(f"Prepared passages saved to {output_path}")
            return

        # Bind recovery to model versions, input bytes, source code and passage scope.
        identity = {"input": input_hash, "code": run["code_sha256"], "config": config,
                    "scope": scope, "score_only": args.score_only,
                    "compare_routing": args.compare_routing,
                    "reference": reference_hash,
                    "libraries": environment_versions(),
                    "passages": [p["id"] for p in passages]}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        if args.resume and not checkpoint_path.exists():
            raise FileNotFoundError("No checkpoint exists for --resume.")
        if checkpoint_path.exists() and not (args.resume or args.overwrite):
            raise FileExistsError("Checkpoint exists. Use --resume, --overwrite or a new output name.")
        records = read_checkpoint(checkpoint_path, fingerprint) if args.resume else []
        scores = {r["passage_id"]: r["result"] for r in records if r["kind"] == "perplexity"}
        reviews = {r["passage_id"]: r["result"] for r in records if r["kind"] == "review"}
        models = {r["role"]: r["metadata"] for r in records if r["kind"] == "model"}
        torch = require_cuda()
        properties = torch.cuda.get_device_properties(0)
        run.update({"device": torch.cuda.get_device_name(0), "gpu_total_bytes": properties.total_memory,
                    "checkpoint_path": str(checkpoint_path), "resumed": args.resume})
        run["library_versions"] = identity["libraries"]
        started = time.perf_counter()
        with checkpoint_path.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as checkpoint:
            if not args.resume:
                append_record(checkpoint, {"kind": "header", "fingerprint": fingerprint})
            score_missing_passages(passages, config["perplexity"], scores, models, checkpoint, torch.cuda)

            routing = without_perplexity = None
            selected = []
            if args.development_all and not args.score_only:
                selected = passages
            elif not args.score_only:
                routing = select_passages(passages, scores, config["routing"], args.smoke_test)
                selected_ids = {key for key, decision in routing["decisions"].items() if decision["selected"]}
                if args.compare_routing:
                    # Generate one response per passage selected by either policy.
                    without_perplexity = select_passages(passages, scores, config["routing"], use_perplexity=False)
                    selected_ids.update(key for key, decision in without_perplexity["decisions"].items()
                                        if decision["selected"])
                selected = [passage for passage in passages if passage["id"] in selected_ids]

            review_missing_passages(selected, config["llm"], reviews, models, checkpoint, torch.cuda)

        primary_reviews = ({key: review for key, review in reviews.items() if routing["decisions"][key]["selected"]}
                           if routing else reviews)
        merged = merge_reviews(documents, passages, primary_reviews)
        run.update({"stage": "perplexity_only" if args.score_only else "hybrid",
                    "models": models, "invocation_seconds": time.perf_counter() - started,
                    "peak_host_memory_bytes": peak_host_memory(),
                    "reviewed_passage_count": len(reviews),
                    "primary_selected_passage_count": len(primary_reviews),
                    "compare_routing": args.compare_routing,
                    "primary_generation_seconds": sum(r.get("inference_seconds", 0) for r in primary_reviews.values()),
                    "comparison_extra_generation_seconds": sum(r.get("inference_seconds", 0)
                                                               for key, r in reviews.items() if key not in primary_reviews),
                    "review_status_counts": dict(Counter(r["status"] for r in reviews.values())),
                    "generation_attempt_count": sum(r.get("attempt_count", 1) for r in reviews.values()),
                    "merge_policy": "preserve_unflagged_first_pass; complete_replace_flagged; partial_add_only",
                    "perplexity_status_counts": dict(Counter(s["status"] for s in scores.values())),
                    "by_source": summarize_sources(merged),
                    "candidate_count": sum(len(d["entities"]) for d in merged)})
        detailed = [{**passage, "perplexity": scores[passage["id"]],
                     "routing": routing["decisions"][passage["id"]] if routing else None,
                     "routing_without_perplexity": (without_perplexity["decisions"][passage["id"]]
                                                    if without_perplexity else None),
                     "review": reviews.get(passage["id"])} for passage in passages]
        routing_summary = {key: value for key, value in routing.items() if key != "decisions"} if routing else None
        comparison_summary = ({key: value for key, value in without_perplexity.items() if key != "decisions"}
                              if without_perplexity else None)
        write_result(output_path, {"run": run, "routing": routing_summary,
                                   "routing_without_perplexity": comparison_summary,
                                   "passages": detailed, "documents": merged})
        print(f"Saved {output_path}", flush=True)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
