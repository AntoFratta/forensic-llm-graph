"""Evaluate reviewed entity spans and optionally calibrate routing on development."""

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hybrid import code_hash, load_first_pass, write_result
from scripts.run_pipeline import project_path
from src.evaluation import _metrics, load_reference, reference_passages, score_reference
from src.llm_filter import merge_reviews, parse_review
from src.passages import build_passages, select_passages


# The published runner added input and checkpoint checks without changing inference or routing.
COMPATIBLE_EVALUATION_CODE_HASHES = (
    "ff6db752bfb62265a56982b217c2a33755d81e67b72d854b98f55dd6ec53de4e",
    "584b73f513f40f2895099039dfcfc12f6336b5aa7d7a4c1d9f66f61e857f890c",
)


def overall_metrics(result: dict) -> dict:
    """Pool exact-match counts across all reviewed passages."""
    units = result["units"]
    counts = Counter(
        tp=sum(unit["tp"] for unit in units),
        fp=sum(len(unit["false_positives"]) for unit in units),
        fn=sum(len(unit["false_negatives"]) for unit in units),
    )
    return _metrics(counts)


def routing_trial(reference, documents, passages, scores, reviews, settings, use_perplexity=True):
    """Reuse recorded responses to compare routing policies without new generation."""
    routing = select_passages(passages, scores, settings, use_perplexity=use_perplexity)
    selected = {key: reviews[key] for key, decision in routing["decisions"].items() if decision["selected"]}
    if any(review is None for review in selected.values()):
        raise ValueError("A policy selected an unreviewed passage; generate the routing union with --compare-routing.")
    merged = merge_reviews(documents, passages, selected)
    return {"threshold": routing["threshold"], "threshold_mode": routing["threshold_mode"],
            "selected_count": len(selected), "budget": routing["budget"],
            "selected_passage_ids": sorted(selected), "metrics": score_reference(reference, merged)}


def calibrate(reference, documents, passages, settings):
    """Compare three development quantiles and a threshold above all observed scores.

    Select by micro F1 on REDIT, DICE and ArchivioAntimafia, then fewer reviews and the higher threshold.
    Synthetic examples remain separate diagnostics. This small search does not
    establish that the chosen threshold generalizes to other text distributions.
    """
    if reference["split"] != "development":
        raise ValueError("Routing calibration is restricted to development references.")
    scores = {p["id"]: p["perplexity"] for p in passages}
    reviews = {p["id"]: p["review"] for p in passages}
    if any(review is None for review in reviews.values()):
        raise ValueError("Calibration requires a recorded review attempt for every development passage.")
    values = sorted(score["perplexity"] for score in scores.values()
                    if score["status"] == "ok" and score["perplexity"] is not None)
    if not values:
        raise ValueError("No valid development perplexities are available.")
    fallback = values[max(0, math.ceil(len(values) * settings["diagnostic_percentile"]) - 1)]
    thresholds = {values[max(0, math.ceil(len(values) * q) - 1)] for q in (0.5, 0.75, 0.9)}
    thresholds.add(fallback)
    thresholds.add(math.nextafter(values[-1], math.inf))
    trials = [routing_trial(reference, documents, passages, scores, reviews,
                            {**settings, "perplexity_threshold": threshold}) for threshold in sorted(thresholds)]

    def objective(trial):
        source_counts = trial["metrics"]["groups"].get("source_documents", {}).get("micro", {})
        value = source_counts.get("f1")
        return (-1 if value is None else value, -trial["selected_count"], trial["threshold"])

    quality_varies = len({objective(trial)[0] for trial in trials}) > 1
    chosen = (max(trials, key=objective) if quality_varies else
              next(trial for trial in trials if trial["threshold"] == fallback))
    without_ppl = routing_trial(reference, documents, passages, scores, reviews, settings, use_perplexity=False)
    return {"objective": "micro F1 on REDIT, DICE and ArchivioAntimafia; ties: fewer reviews, higher threshold",
            "budget_scope": "the source-balanced annotated reference subset, not the full corpus",
            "threshold_quality_varies_on_development": quality_varies,
            "distinct_selected_sets": len({tuple(trial["selected_passage_ids"]) for trial in trials}),
            "threshold_selection": ("development_F1_search" if quality_varies else
                                    "fixed_development_percentile_fallback; no measured F1 preference"),
            "chosen_threshold": chosen["threshold"], "chosen": chosen,
            "without_perplexity": without_ppl, "trials": trials}


def replay_validation(hybrid: dict, documents: list) -> dict:
    """Revalidate saved development responses only if generation code is unchanged.

    Git preserves the original prompt and generation implementation. Comparing
    them prevents reusing old responses as evidence for a different prompt or
    model backend. Model outputs and generation attempts remain untouched on disk.
    """
    run = hybrid["run"]
    commit = run.get("git", {}).get("commit", "")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Replay requires the original generation commit in local Git history.")
    for name in ("src/language_models.py", "src/llm_filter.py"):
        previous = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=PROJECT_ROOT,
                                  capture_output=True, text=True, encoding="utf-8")
        if previous.returncode:
            raise ValueError("Original generation source is unavailable; do not replay unverifiable responses.")
        current = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        if name.endswith("language_models.py"):
            unchanged = previous.stdout == current
        else:
            def prompt_parts(source):
                return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                        if (isinstance(node, ast.FunctionDef) and node.name == "review_messages")
                        or (isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets))]
            unchanged = prompt_parts(previous.stdout) == prompt_parts(current)
        if not unchanged:
            raise ValueError("Prompt or generation implementation changed; a fresh model run is required.")
    before = Counter(p["review"]["status"] for p in hybrid["passages"])
    for passage in hybrid["passages"]:
        review = passage["review"]
        if review.get("raw_response") and review["status"] in {"ok", "partial", "invalid_response"}:
            review["original_validation_status"] = review["status"]
            review["original_validation_errors"] = review.get("validation_errors", [])
            review.update(parse_review(review["raw_response"], passage, run["config"]["llm"]["model"]))
    reviews = {p["id"]: p["review"] for p in hybrid["passages"]}
    hybrid["documents"] = merge_reviews(documents, hybrid["passages"], reviews)
    return {"original_generation_code_sha256": run["code_sha256"],
            "evaluated_code_sha256": code_hash(), "generation_commit": commit,
            "generation_source_and_prompt_verified": True, "new_model_generations": 0,
            "review_status_before": dict(before),
            "review_status_after": dict(Counter(r["status"] for r in reviews.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True, help="Recorded first-pass JSON.")
    parser.add_argument("--hybrid", help="Optional model output on exactly these reference passages.")
    parser.add_argument("--calibrate-routing", action="store_true")
    parser.add_argument("--replay-validation", action="store_true",
                        help="Revalidate recorded development responses after parser/fusion fixes, without inference.")
    parser.add_argument("--calibrated-config", help="Save a configuration with the chosen development threshold.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        output_path = project_path(args.output)
        config_output = project_path(args.calibrated_config) if args.calibrated_config else None
        inputs = {project_path(p) for p in (args.reference, args.baseline, args.hybrid) if p}
        for target in (output_path, config_output):
            if target and (target in inputs or target.suffix != ".json"):
                raise ValueError("Use JSON outputs distinct from every input.")
            if target and target.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {target}; choose another name or use --overwrite.")
        if output_path == config_output:
            raise ValueError("Evaluation and calibrated configuration require distinct paths.")
        if args.calibrate_routing and not args.hybrid:
            raise ValueError("Calibration requires --hybrid development model output.")
        if args.replay_validation and not args.hybrid:
            raise ValueError("Validation replay requires --hybrid development output.")
        if config_output and not args.calibrate_routing:
            raise ValueError("--calibrated-config requires --calibrate-routing.")
        reference, reference_hash = load_reference(project_path(args.reference))
        if (args.calibrate_routing or args.replay_validation) and reference["split"] != "development":
            raise ValueError("Never calibrate on test data.")
        baseline, baseline_hash = load_first_pass(project_path(args.baseline))
        identifiers = {u["document_id"] for u in reference["units"]}
        documents = [d for d in baseline["documents"] if d["id"] in identifiers]
        report = {"stage": "reference_evaluation", "split": reference["split"],
                  "reference_sha256": reference_hash, "baseline_sha256": baseline_hash,
                  "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                  "metrics": {"first_pass": score_reference(reference, documents)},
                  "limitations": ["Small source-balanced reference, not population-representative accuracy.",
                                  "Undefined metrics are null; inspect category support before interpreting F1."]}
        if args.hybrid:
            hybrid_path = project_path(args.hybrid)
            hybrid = json.loads(hybrid_path.read_text(encoding="utf-8-sig"))
            run = hybrid["run"]
            if (run.get("input_sha256") != baseline_hash or run.get("reference_sha256") != reference_hash
                    or run.get("stage") != "hybrid"):
                raise ValueError("Model output does not match the baseline, reference or executed pipeline code.")
            recorded_code, current_code = run.get("code_sha256"), code_hash()
            if (recorded_code != current_code
                    and (recorded_code, current_code) != COMPATIBLE_EVALUATION_CODE_HASHES
                    and not args.replay_validation):
                raise ValueError("Pipeline code changed; rerun the models or explicitly verify a development validation replay.")
            expected = {u["id"] for u in reference["units"]}
            if {p["id"] for p in hybrid["passages"]} != expected or len(hybrid["passages"]) != len(expected):
                raise ValueError("Model output must cover exactly the reference passage IDs.")
            operational = build_passages(documents, **{key: run["config"]["passages"][key]
                                                       for key in ("core_chars", "context_chars")})
            reference_passages(reference, documents, operational, run["config"]["passages"])
            if args.replay_validation:
                if run["scope"] != "development_all_review":
                    raise ValueError("Replay is restricted to all-review development runs.")
                report["validation_replay"] = replay_validation(hybrid, documents)
            report["hybrid_sha256"] = hashlib.sha256(hybrid_path.read_bytes()).hexdigest()
            mode = "all_review_development_diagnostic" if run["scope"] == "development_all_review" else "hybrid"
            report["metrics"][mode] = score_reference(reference, hybrid["documents"])
            if run.get("compare_routing"):
                without_ppl = routing_trial(
                    reference, documents, hybrid["passages"],
                    {p["id"]: p["perplexity"] for p in hybrid["passages"]},
                    {p["id"]: p["review"] for p in hybrid["passages"]},
                    run["config"]["routing"], use_perplexity=False)
                report["metrics"]["without_perplexity"] = without_ppl["metrics"]
                report["fixed_routing_comparison"] = {key: value for key, value in without_ppl.items() if key != "metrics"}
            if args.calibrate_routing:
                if run["scope"] != "development_all_review":
                    raise ValueError("Calibration requires a --development-all run, not selectively observed responses.")
                result = calibrate(reference, documents, hybrid["passages"], run["config"]["routing"])
                report["calibration"] = result
                report["metrics"]["hybrid_selected"] = result["chosen"]["metrics"]
                report["metrics"]["without_perplexity"] = result["without_perplexity"]["metrics"]
                if config_output:
                    config = deepcopy(run["config"])
                    config["routing"]["perplexity_threshold"] = result["chosen_threshold"]
                    config["output_path"] = "outputs/hybrid_reference.json"
                    config["calibration"] = {"split": "development", "reference_sha256": reference_hash,
                                             "objective": result["objective"],
                                             "threshold_selection": result["threshold_selection"],
                                             "pipeline_code_sha256": code_hash()}
                    config_output.parent.mkdir(parents=True, exist_ok=True)
                    write_result(config_output, config)
        for result in report["metrics"].values():
            result["overall"] = overall_metrics(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_result(output_path, report)
        for mode, result in report["metrics"].items():
            print(f"{mode}: {result['overall']}")
        if "calibration" in report:
            print("Development threshold:", report["calibration"]["chosen_threshold"])
        print("Saved", output_path)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
