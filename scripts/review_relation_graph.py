"""Prepare and run a bounded Qwen review of passages with relation cues."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_relation_graph import graph_code_hash, load_ner
from scripts.run_hybrid import validate_config, write_result
from scripts.run_pipeline import project_path
from src.language_models import load_model
from src.relation_review import review_relations, select_relation_passages, text_hash


def prepare(input_path: Path, reference_path: Path | None, limit: int) -> dict:
    if limit < 1:
        raise ValueError("The passage budget must be positive.")
    run, documents, input_hash = load_ner(input_path)
    settings = run.get("config", {}).get("passages", {})
    units = None
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
        if reference["input_sha256"] != input_hash:
            raise ValueError("The reference must use the same saved NER input.")
        units = reference["units"]
    passages, eligible = select_relation_passages(documents, settings.get("core_chars", 800),
                                                 settings.get("context_chars", 160), limit, units)
    return {"run": {"stage": "relation_review_plan", "input_sha256": input_hash,
                    "code_sha256": graph_code_hash(with_review=True),
                    "scope": "reference_diagnostic" if units is not None else "corpus",
                    "eligible_passages": eligible, "selected_passages": len(passages)},
            "passages": [{**p, "text_sha256": text_hash(p["text"])} for p in passages]}


def run_review(plan_path: Path, config_path: Path) -> dict:
    content = plan_path.read_bytes()
    plan = json.loads(content.decode("utf-8-sig"))
    run = plan["run"]
    if run.get("stage") != "relation_review_plan" or run.get("code_sha256") != graph_code_hash(with_review=True):
        raise ValueError("Prepare the review plan with the same graph code before running Qwen.")
    passages = plan["passages"]
    if len(passages) != run["selected_passages"] or len({p["id"] for p in passages}) != len(passages):
        raise ValueError("Invalid passage list in review plan.")
    for p in passages:
        start, first, last, end = (p[k] for k in ("start", "core_start", "core_end", "end"))
        if (any(type(v) is not int for v in (start, first, last, end))
                or not 0 <= start <= first < last <= end or len(p["text"]) != end - start
                or text_hash(p["text"]) != p["text_sha256"]):
            raise ValueError(f"Invalid passage in review plan: {p['id']}.")

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    settings = config["llm"]
    reviews = []
    model_info = {"model": settings["model"], "revision": settings["revision"]}
    if passages:
        model, tokenizer, model_info = load_model(settings, quantized=True)
        for number, passage in enumerate(passages, 1):
            review = review_relations(passage, settings, model, tokenizer)
            reviews.append({**{key: val for key, val in passage.items() if key != "text"}, **review})
            print(f"Reviewed {number}/{len(passages)}: {passage['id']} ({review['status']})", flush=True)
    return {"run": {**run, "stage": "relation_review", "plan_sha256": hashlib.sha256(content).hexdigest(),
                    "model": model_info, "settings": settings,
                    "statuses": dict(Counter(row["status"] for row in reviews))},
            "reviews": reviews}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("prepare", help="Select passages locally and write a model input file.")
    plan.add_argument("--input", required=True, help="Saved NER JSON.")
    plan.add_argument("--reference", help="Optional reference JSON. Only passage text and offsets are used.")
    plan.add_argument("--max-passages", type=int, default=100)
    run = commands.add_parser("run", help="Review a prepared plan on a CUDA GPU.")
    run.add_argument("--plan", required=True)
    run.add_argument("--config", default="configs/hybrid.json")
    for command in (plan, run):
        command.add_argument("--output", required=True)
        command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        output = project_path(args.output)
        if output.suffix.lower() != ".json":
            raise ValueError("Choose a JSON output.")
        if output.exists() and not args.overwrite:
            raise FileExistsError("Output already exists. Choose another path or use --overwrite.")
        if args.command == "prepare":
            input_path = project_path(args.input)
            reference_path = project_path(args.reference) if args.reference else None
            if output in (input_path, reference_path):
                raise ValueError("Output must be distinct from the input and reference.")
            result = prepare(input_path, reference_path, args.max_passages)
        else:
            plan_path, config_path = project_path(args.plan), project_path(args.config)
            if output in (plan_path, config_path):
                raise ValueError("Output must be distinct from the plan and configuration.")
            result = run_review(plan_path, config_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_result(output, result)
        run = result["run"]
        print(f"Selected {run['selected_passages']} of {run['eligible_passages']} eligible passages.")
        print(f"Saved {output}")
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
