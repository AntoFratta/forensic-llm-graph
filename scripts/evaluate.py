"""Evaluate reviewed entity spans and optionally calibrate routing on development."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hybrid import code_hash, load_first_pass, write_result
from scripts.run_pipeline import project_path
from src.evaluation import load_reference, reference_passages, score_reference
from src.llm_filter import merge_reviews
from src.passages import build_passages, select_passages


def routing_trial(reference, documents, passages, scores, reviews, settings, use_perplexity=True):
    """Reuse recorded responses to compare routing policies without new generation."""
    routing = select_passages(passages, scores, settings, use_perplexity=use_perplexity)
    selected = {key: reviews[key] for key, decision in routing["decisions"].items() if decision["selected"]}
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
    thresholds = {values[max(0, math.ceil(len(values) * q) - 1)] for q in (0.5, 0.75, 0.9)}
    thresholds.add(math.nextafter(values[-1], math.inf))
    trials = [routing_trial(reference, documents, passages, scores, reviews,
                            {**settings, "perplexity_threshold": threshold}) for threshold in sorted(thresholds)]

    def objective(trial):
        source_counts = trial["metrics"]["groups"].get("source_documents", {}).get("micro", {})
        value = source_counts.get("f1")
        return (-1 if value is None else value, -trial["selected_count"], trial["threshold"])

    chosen = max(trials, key=objective)
    without_ppl = routing_trial(reference, documents, passages, scores, reviews, settings, use_perplexity=False)
    return {"objective": "micro F1 on REDIT, DICE and ArchivioAntimafia; ties: fewer reviews, higher threshold",
            "budget_scope": "the source-balanced annotated reference subset, not the full corpus",
            "chosen_threshold": chosen["threshold"], "chosen": chosen,
            "without_perplexity": without_ppl, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True, help="Recorded first-pass JSON.")
    parser.add_argument("--hybrid", help="Optional model output on exactly these reference passages.")
    parser.add_argument("--calibrate-routing", action="store_true")
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
        if config_output and not args.calibrate_routing:
            raise ValueError("--calibrated-config requires --calibrate-routing.")
        reference, reference_hash = load_reference(project_path(args.reference))
        if args.calibrate_routing and reference["split"] != "development":
            raise ValueError("Never calibrate on test data.")
        baseline, baseline_hash = load_first_pass(project_path(args.baseline))
        identifiers = {u["document_id"] for u in reference["units"]}
        documents = [d for d in baseline["documents"] if d["id"] in identifiers]
        report = {"stage": "reference_evaluation", "split": reference["split"],
                  "reference_sha256": reference_hash, "baseline_sha256": baseline_hash,
                  "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                  "metrics": {"first_pass": score_reference(reference, documents)},
                  "limitations": ["Small source-balanced reference, not population-representative accuracy.",
                                  "REDIT forms, news and judicial texts are separated from generated synthetic diagnostics.",
                                  "Undefined metrics are null; inspect category support before interpreting F1."]}
        if args.hybrid:
            hybrid_path = project_path(args.hybrid)
            hybrid = json.loads(hybrid_path.read_text(encoding="utf-8-sig"))
            run = hybrid["run"]
            if (run.get("input_sha256") != baseline_hash or run.get("reference_sha256") != reference_hash
                    or run.get("code_sha256") != code_hash() or run.get("stage") != "hybrid"):
                raise ValueError("Model output does not match the baseline, reference or executed pipeline code.")
            expected = {u["id"] for u in reference["units"]}
            if {p["id"] for p in hybrid["passages"]} != expected or len(hybrid["passages"]) != len(expected):
                raise ValueError("Model output must cover exactly the reference passage IDs.")
            operational = build_passages(documents, **{key: run["config"]["passages"][key]
                                                       for key in ("core_chars", "context_chars")})
            reference_passages(reference, documents, operational, run["config"]["passages"])
            report["hybrid_sha256"] = hashlib.sha256(hybrid_path.read_bytes()).hexdigest()
            mode = "all_review_development_diagnostic" if run["scope"] == "development_all_review" else "hybrid"
            report["metrics"][mode] = score_reference(reference, hybrid["documents"])
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
                    config["output_path"] = "outputs/10_hybrid_reference_test.json"
                    config["calibration"] = {"split": "development", "reference_sha256": reference_hash,
                                             "evaluation_output": str(output_path), "objective": result["objective"]}
                    config_output.parent.mkdir(parents=True, exist_ok=True)
                    write_result(config_output, config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_result(output_path, report)
        for mode, result in report["metrics"].items():
            print(mode)
            for group in ("source_documents", "synthetic"):
                if group in result["groups"]:
                    print(" ", group, result["groups"][group]["micro"])
        if "calibration" in report:
            print("Development threshold:", report["calibration"]["chosen_threshold"])
        print("Saved", output_path)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
