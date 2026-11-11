"""Score explicit graph relations against human-reviewed text excerpts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hybrid import write_result
from scripts.run_pipeline import project_path
from src.evaluation import _metrics
from src.relation_graph import SUBJECT_LABELS


def _relation_key(source: dict, predicate: dict, target: dict) -> tuple:
    return (source["start"], source["end"], source["label"],
            predicate["start"], predicate["end"],
            target["start"], target["end"], target["label"])


def _reviewed_relations(unit: dict) -> set:
    """Require manually accepted annotations, including for empty excerpts."""
    if unit.get("annotation_status") != "reviewed":
        raise ValueError(f"Reference excerpt {unit['id']} still needs review.")
    gold = set()
    for relation in unit["relations"]:
        source, predicate, target = (relation[name] for name in ("source", "predicate", "target"))
        for span in (source, predicate, target):
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int
                    or not unit["start"] <= start < end <= unit["end"]):
                raise ValueError(f"Invalid relation span in {unit['id']}.")
        if (source["label"] not in SUBJECT_LABELS or target["label"] not in SUBJECT_LABELS
                or (source["end"] > target["start"] and target["end"] > source["start"])):
            raise ValueError(f"Invalid subject spans in {unit['id']}.")
        key = _relation_key(source, predicate, target)
        if key in gold:
            raise ValueError(f"Duplicate reference relation in {unit['id']}.")
        gold.add(key)
    return gold


def score_graph(graph: dict, reference: dict, diagnostic: bool = False) -> dict:
    """Score direction, endpoint boundaries and labels, and predicate boundaries."""
    run = graph["run"]
    if run.get("stage") != "relation_graph" or run.get("input_sha256") != reference["input_sha256"]:
        raise ValueError("The graph and reference must use the same saved NER output.")
    if (not diagnostic and "graph_code_sha256" in reference
            and run.get("code_sha256") != reference["graph_code_sha256"]):
        raise ValueError("The graph was built with different code from the frozen reference.")
    units = reference["units"]
    if not units:
        raise ValueError("The reference must contain at least one reviewed excerpt.")

    nodes = {node["id"]: node for node in graph["nodes"]}
    documents = {node["document_id"] for node in nodes.values() if node["type"] == "DOCUMENT"}
    predictions = defaultdict(set)
    for edge in graph["edges"]:
        if edge["type"] != "EXPLICIT_RELATION":
            continue
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        for evidence in edge["evidence"]:
            predictions[evidence["document_id"]].add(_relation_key(
                {**evidence["source_mention"], "label": source["label"]},
                {"start": evidence["predicate_start"], "end": evidence["predicate_end"]},
                {**evidence["target_mention"], "label": target["label"]},
            ))

    totals, results = Counter(), []
    ids, boundaries = set(), defaultdict(list)
    for unit in units:
        start, end, document_id = unit["start"], unit["end"], unit["document_id"]
        if (unit["id"] in ids or document_id not in documents or type(start) is not int
                or type(end) is not int or not 0 <= start < end or len(unit["text"]) != end - start):
            raise ValueError(f"Invalid reference excerpt: {unit['id']}.")
        if any(start < other_end and other_start < end for other_start, other_end in boundaries[document_id]):
            raise ValueError(f"Reference excerpts overlap in {document_id}.")
        ids.add(unit["id"])
        boundaries[document_id].append((start, end))
        gold = _reviewed_relations(unit)
        predicted = {key for key in predictions[document_id]
                     if start <= min(key[0], key[3], key[5]) and max(key[1], key[4], key[6]) <= end}
        missing, extra = gold - predicted, predicted - gold
        counts = Counter(tp=len(gold & predicted), fp=len(extra), fn=len(missing))
        totals.update(counts)
        results.append({"id": unit["id"], **counts, "false_positives": sorted(extra),
                        "false_negatives": sorted(missing)})

    summary = _metrics(totals)
    summary["gold_relations"] = summary.pop("gold_mentions")
    summary["predicted_relations"] = summary.pop("predicted_mentions")
    return {"scope": "development_diagnostic" if diagnostic else reference["scope"],
            "matching": "exact_directed_endpoints_and_predicate",
            "unit_count": len(units), "overall": summary, "units": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="outputs/relation_graph.json")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default="outputs/relation_evaluation.json")
    parser.add_argument("--diagnostic", action="store_true", help="Use known excerpts to diagnose a changed method.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        graph_path, reference_path = project_path(args.graph), project_path(args.reference)
        output = project_path(args.output)
        if output in (graph_path, reference_path) or output.suffix.lower() != ".json":
            raise ValueError("Choose a JSON output distinct from both inputs.")
        if output.exists() and not args.overwrite:
            raise FileExistsError("Output already exists. Choose another path or use --overwrite.")
        graph = json.loads(graph_path.read_text(encoding="utf-8-sig"))
        reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
        result = score_graph(graph, reference, diagnostic=args.diagnostic)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_result(output, result)
        print(result["overall"])
        print(f"Saved {output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
