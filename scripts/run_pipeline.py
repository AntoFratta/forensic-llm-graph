"""Run first-pass extraction on prepared documents and record experiment metadata.

The output contains entity candidates for review by the hybrid pipeline.
Source annotations are kept separate from extraction.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from itertools import repeat
from pathlib import Path
import platform
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution from scripts/ without installing the project as a package.
sys.path.insert(0, str(PROJECT_ROOT))

from src.deterministic import extract_entities, load_nlp


def project_path(value: str) -> Path:
    """Resolve relative paths from the repository root."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_documents(path: Path) -> tuple[list, str]:
    """Return validated documents and the SHA-256 hash of the bytes loaded.

    Each document requires a unique non-empty string id and a text field.
    Empty text is allowed, so an extraction failure remains visible in results.
    Additional fields are not interpreted as reference annotations.
    """
    content = path.read_bytes()
    documents = json.loads(content.decode("utf-8-sig"))
    if not isinstance(documents, list) or not documents:
        raise ValueError("Input must be a non-empty JSON array of documents.")

    identifiers = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document must be a JSON object.")
        identifier = document.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Each document must have a non-empty string id.")
        if identifier in identifiers:
            raise ValueError(f"Duplicate document id: {identifier}")
        if not isinstance(document.get("text"), str):
            raise ValueError(f"Document {identifier} must contain a string text field.")
        identifiers.add(identifier)
    return documents, hashlib.sha256(content).hexdigest()


def git_info() -> dict:
    """Return the Git commit and whether the working tree has local changes.

    These values are None when Git or repository history is unavailable.
    A commit with uncommitted changes does not fully identify the code executed.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "uncommitted_changes": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def summarize_sources(documents: list) -> dict:
    """Count candidates by collection, keeping synthetic diagnostics separate."""
    summaries = {}
    for document in documents:
        source = document.get("source", "unknown")
        summary = summaries.setdefault(source, {
            "documents": 0, "candidates": 0, "flagged_candidates": 0, "labels": {},
        })
        summary["documents"] += 1
        for entity in document["entities"]:
            summary["candidates"] += 1
            summary["flagged_candidates"] += int(entity["needs_review"])
            label = entity["label"]
            summary["labels"][label] = summary["labels"].get(label, 0) + 1
    return summaries


def main() -> None:
    """Validate settings, load the model once, extract candidates and save results.

    --rules-only overrides model use and is recorded in the output metadata.
    Existing output files require explicit permission through --overwrite.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output", help="Use a different output path for this run.")
    parser.add_argument("--rules-only", action="store_true", help="Run without the spaCy model.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()

    try:
        config_path = project_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object.")
        for key in ("input_path", "output_path", "spacy_model", "spacy_model_version"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"Configuration requires a non-empty string: {key}")
        if not isinstance(config.get("use_spacy"), bool):
            raise ValueError("use_spacy must be true or false.")
        batch_size = config.get("spacy_batch_size", 8)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("spacy_batch_size must be a positive integer.")
        config["spacy_batch_size"] = batch_size
        if args.output is not None:
            config["output_path"] = args.output

        input_path = project_path(config["input_path"])
        output_path = project_path(config["output_path"])
        if input_path == output_path or output_path == config_path:
            raise ValueError("Output must not replace the input or configuration.")
        if output_path.suffix.lower() != ".json":
            raise ValueError("Output must be a JSON file.")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError("Output already exists. Choose another path or use --overwrite.")

        documents, input_sha256 = load_documents(input_path)
        use_spacy = config["use_spacy"] and not args.rules_only
        nlp = None
        started = time.perf_counter()
        if use_spacy:
            print(f"Loading {config['spacy_model']}...", flush=True)
            nlp = load_nlp(config["spacy_model"])
            if nlp.meta["version"] != config["spacy_model_version"]:
                raise ValueError("Installed model version differs from the configuration.")
        load_seconds = time.perf_counter() - started

        results = []
        started = time.perf_counter()
        # Consume Docs as they are produced instead of retaining the entire NLP corpus.
        parsed_documents = (
            nlp.pipe((document["text"] for document in documents), batch_size=batch_size)
            if use_spacy else repeat(None, len(documents))
        )
        if use_spacy:
            print(f"Processing with spaCy batches of {batch_size}...", flush=True)
        for number, (document, doc) in enumerate(zip(documents, parsed_documents, strict=True), start=1):
            result = {"id": document["id"], "text": document["text"]}
            # Copy provenance only; any input annotations are excluded.
            for key in ("document", "page", "pages", "source", "start", "end", "source_file", "source_sha256",
                        "case_id", "article_id", "url", "title", "crime_category", "publication_date", "event_date",
                        "synthetic", "language", "template_id", "source_revision", "source_url", "source_line"):
                if key in document:
                    result[key] = document[key]
            result["entities"] = extract_entities(document["text"], doc=doc)
            results.append(result)
            if number == 1 or number % 100 == 0 or number == len(documents):
                print(f"Processed {number}/{len(documents)}: {document['id']}", flush=True)
        extraction_seconds = time.perf_counter() - started

        run = {
            "stage": "first_pass",
            "mode": "rules_and_spacy" if use_spacy else "rules_only",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "rules_only_override": args.rules_only,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_sha256": input_sha256,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "spacy_version": version("spacy") if use_spacy else None,
            "model_name": nlp.meta.get("name") if use_spacy else None,
            "model_version": nlp.meta.get("version") if use_spacy else None,
            "spacy_batch_size": batch_size if use_spacy else None,
            "spacy_pipeline": list(nlp.pipe_names) if use_spacy else [],
            "device": "cpu",
            "git": git_info(),
            "model_load_seconds": load_seconds,
            "extraction_seconds": extraction_seconds,
            "document_count": len(results),
            "candidate_count": sum(len(r["entities"]) for r in results),
            "flagged_candidate_count": sum(e["needs_review"] for r in results for e in r["entities"]),
            "by_source": summarize_sources(results),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"run": run, "documents": results}, ensure_ascii=False, indent=2),
            encoding="utf-8", newline="\n",
        )
        print(f"Saved {run['candidate_count']} candidates to {output_path}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
