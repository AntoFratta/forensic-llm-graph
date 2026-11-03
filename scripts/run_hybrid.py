"""Review selected first-pass passages using Minerva perplexity and Qwen.

Use --smoke-test for the first GPU check, --prepare-only to inspect passage
construction offline, and --resume to continue matching checkpoint files.
An extended review requires a perplexity threshold chosen on development data.
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
    if result.get("run", {}).get("stage") != "first_pass":
        raise ValueError("Input must be a recorded first-pass run, not another hybrid result.")
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
    if type(config["passages"].get("seed")) is not int:
        raise ValueError("passages.seed must be an integer.")
    if type(config["passages"].get("smoke_per_source")) is not int or config["passages"]["smoke_per_source"] < 1:
        raise ValueError("passages.smoke_per_source must be a positive integer.")


def code_hash() -> str:
    """Identify executed source files, including uncommitted local edits."""
    digest = hashlib.sha256()
    for name in ("scripts/run_hybrid.py", "scripts/run_pipeline.py", "src/passages.py",
                 "src/llm_filter.py", "src/language_models.py"):
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
    header = json.loads(lines[0])
    if header.get("fingerprint") != fingerprint or header.get("kind") != "header":
        raise ValueError("Checkpoint belongs to different data, code, settings or execution scope.")
    records = []
    valid_bytes = len(lines[0])
    for index, line in enumerate(lines[1:], start=1):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete checkpoint line.")
            record = json.loads(line)
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError("Checkpoint contains an invalid record before its end.")
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
            print("Discarded an incomplete trailing checkpoint record.", flush=True)
            break
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid.json")
    parser.add_argument("--input", help="First-pass JSON; no raw-data download is required.")
    parser.add_argument("--output", help="New result JSON; checkpoint path is derived from it.")
    parser.add_argument("--smoke-test", action="store_true", help="Test a few passages per collection.")
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
        if args.input:
            config["input_path"] = args.input
        if args.output:
            config["output_path"] = args.output
        validate_config(config)
        if not (args.smoke_test or args.prepare_only or args.score_only):
            select_passages([], {}, config["routing"])
        input_path, output_path = project_path(config["input_path"]), project_path(config["output_path"])
        checkpoint_path = output_path.with_suffix(".checkpoints.jsonl")
        if output_path in {input_path, config_path} or output_path.suffix.lower() != ".json":
            raise ValueError("Choose a JSON output distinct from input and configuration.")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError("Output exists. Choose a new name or use --overwrite.")
        source, input_hash = load_first_pass(input_path)
        documents = source["documents"]
        passages = build_passages(documents, config["passages"]["core_chars"], config["passages"]["context_chars"])
        full_passage_count = len(passages)
        if args.smoke_test:
            passages = smoke_sample(passages, config["passages"]["smoke_per_source"], config["passages"]["seed"])
            identifiers = {p["document_id"] for p in passages}
            documents = [document for document in documents if document["id"] in identifiers]
        scope = "smoke_test" if args.smoke_test else "extended"
        print(f"Scope: {scope}; {len(passages)}/{full_passage_count} passages; "
              f"{len(documents)}/{len(source['documents'])} documents represented.", flush=True)
        run = {"stage": "hybrid_preparation" if args.prepare_only else "hybrid",
               "scope": scope, "created_utc": datetime.now(timezone.utc).isoformat(),
               "config": config, "input_sha256": input_hash, "input_run": source["run"],
               "code_sha256": code_hash(), "git": git_info(),
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
            remaining = [p for p in passages if p["id"] not in scores]
            if remaining:
                print(f"Loading Minerva; {len(remaining)} passages to score...", flush=True)
                model, tokenizer, metadata = load_model(config["perplexity"], quantized=False)
                try:
                    for batch in perplexity_batches(remaining, config["perplexity"], model, tokenizer):
                        for identifier, score in batch.items():
                            scores[identifier] = score
                            append_record(checkpoint, {"kind": "perplexity", "passage_id": identifier, "result": score})
                        print(f"Perplexity: {len(scores)}/{len(passages)}", flush=True)
                    metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    models["perplexity"] = metadata
                    append_record(checkpoint, {"kind": "model", "role": "perplexity", "metadata": metadata})
                finally:
                    del model, tokenizer
                    release_cuda_memory()
            routing = None if args.score_only else select_passages(passages, scores, config["routing"], args.smoke_test)
            selected = [] if routing is None else [p for p in passages if routing["decisions"][p["id"]]["selected"]]
            remaining = [p for p in selected if p["id"] not in reviews]
            if remaining:
                print(f"Minerva released. Loading Qwen; {len(remaining)} passages to review...", flush=True)
                model, tokenizer, metadata = load_model(config["llm"], quantized=True)
                try:
                    for passage in remaining:
                        review = review_passage(passage, config["llm"], model, tokenizer)
                        reviews[passage["id"]] = review
                        append_record(checkpoint, {"kind": "review", "passage_id": passage["id"], "result": review})
                        print(f"LLM: {len(reviews)}/{len(selected)}; {review['status']}; {passage['id']}", flush=True)
                    metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    models["llm"] = metadata
                    append_record(checkpoint, {"kind": "model", "role": "llm", "metadata": metadata})
                finally:
                    del model, tokenizer
                    release_cuda_memory()

        merged = merge_reviews(documents, passages, reviews)
        run.update({"stage": "perplexity_only" if args.score_only else "hybrid",
                    "models": models, "invocation_seconds": time.perf_counter() - started,
                    "peak_host_memory_bytes": peak_host_memory(),
                    "reviewed_passage_count": len(reviews),
                    "review_status_counts": dict(Counter(r["status"] for r in reviews.values())),
                    "perplexity_status_counts": dict(Counter(s["status"] for s in scores.values())),
                    "by_source": summarize_sources(merged),
                    "candidate_count": sum(len(d["entities"]) for d in merged)})
        detailed = [{**passage, "perplexity": scores[passage["id"]],
                     "routing": routing["decisions"][passage["id"]] if routing else None,
                     "review": reviews.get(passage["id"])} for passage in passages]
        routing_summary = {key: value for key, value in routing.items() if key != "decisions"} if routing else None
        write_result(output_path, {"run": run, "routing": routing_summary,
                                   "passages": detailed, "documents": merged})
        print(f"Saved {output_path}", flush=True)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
