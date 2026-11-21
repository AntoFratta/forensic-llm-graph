"""Prepare relation reviews, build a graph from saved NER and evaluate it."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hybrid import validate_config, write_result
from scripts.run_pipeline import project_path
from src.evaluation import _metrics
from src.graph_queries import inspect_graph, score_context
from src.language_models import load_model
from src.relation_graph import SUBJECT_LABELS, build_graph
from src.relation_review import (
    CONTEXT_MAX_NEW_TOKENS,
    load_review_relations,
    review_relations,
    select_context_passages,
    select_relation_passages,
    text_hash,
)


def graph_code_hash(with_review: bool = False) -> str:
    """Identify the extraction code used for an evaluation run."""
    digest = hashlib.sha256()
    files = ["src/explicit_relations.py", "src/relation_graph.py", "src/passages.py", "src/deterministic.py",
             "src/context_facts.py", "src/document_links.py", "src/llm_filter.py"]
    if with_review:
        files.append("src/relation_review.py")
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update((PROJECT_ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_ner(path: Path) -> tuple[dict, list, str]:
    """Read saved predictions and retain a hash for matching plans and reviews."""
    content = path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    data = json.loads(content.decode("utf-8-sig"))
    if (not isinstance(data, dict) or not isinstance(data.get("run"), dict)
            or not isinstance(data.get("documents"), list) or not data["documents"]):
        raise ValueError("Expected a saved NER run with metadata and documents.")
    run = data["run"]
    if run.get("stage") not in ("first_pass", "hybrid"):
        raise ValueError("Expected first-pass or hybrid NER predictions.")
    config = run.get("config", {})
    if not isinstance(config, dict) or not isinstance(config.get("passages", {}), dict):
        raise ValueError("Invalid NER configuration metadata.")
    return run, data["documents"], fingerprint


def prepare(input_path: Path, reference_path: Path | None, limit: int, *, context: bool = False) -> dict:
    """Select passages and record the input and code used to produce the plan."""
    if limit < 1:
        raise ValueError("The passage budget must be positive.")
    run, documents, input_hash = load_ner(input_path)
    settings = run.get("config", {}).get("passages", {})
    core, padding = settings.get("core_chars", 800), settings.get("context_chars", 160)
    units = None
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
        if reference["input_sha256"] != input_hash:
            raise ValueError("The reference must use the same saved NER input.")
        units = reference["units"]
    if context:
        passages, eligible = select_context_passages(documents, core, padding, limit, units)
    else:
        passages, eligible = select_relation_passages(documents, core, padding, limit, units)
    return {"run": {"stage": "relation_review_plan", "input_sha256": input_hash,
                    "code_sha256": graph_code_hash(with_review=True),
                    "profile": "context" if context else "legacy",
                    "max_passages": limit,
                    "scope": "reference_diagnostic" if units is not None else "corpus",
                    "eligible_passages": eligible, "selected_passages": len(passages)},
            "passages": [{**p, "text_sha256": text_hash(p["text"])} for p in passages]}


def run_review(plan_path: Path, config_path: Path, *, checkpoint_path: Path | None = None,
               check_first: int = 0) -> dict:
    """Validate the saved plan before loading Qwen and recording its responses."""
    content = plan_path.read_bytes()
    plan = json.loads(content.decode("utf-8-sig"))
    run = plan["run"]
    if run.get("stage") != "relation_review_plan" or run.get("code_sha256") != graph_code_hash(with_review=True):
        raise ValueError("Prepare the review plan with the same graph code before running Qwen.")
    if check_first < 0:
        raise ValueError("The initial review check must be non-negative.")
    passages = plan["passages"]
    if len(passages) != run["selected_passages"] or len({p["id"] for p in passages}) != len(passages):
        raise ValueError("Invalid passage list in review plan.")
    for passage in passages:
        start, core_start, core_end, end = (passage[key] for key in ("start", "core_start", "core_end", "end"))
        if (any(type(offset) is not int for offset in (start, core_start, core_end, end))
                or not 0 <= start <= core_start < core_end <= end
                or len(passage["text"]) != end - start
                or text_hash(passage["text"]) != passage["text_sha256"]):
            raise ValueError(f"Invalid passage in review plan: {passage['id']}.")

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    settings = dict(config["llm"])
    if run.get("profile") == "context":
        settings["max_new_tokens"] = min(settings["max_new_tokens"], CONTEXT_MAX_NEW_TOKENS)
    plan_hash = hashlib.sha256(content).hexdigest()
    model_info = {"model": settings["model"], "revision": settings["revision"]}
    metadata = {**run, "stage": "relation_review", "plan_sha256": plan_hash, "settings": settings}
    reviews = []
    if checkpoint_path is not None and checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        reviews = saved["reviews"]
        if (any(saved["run"].get(key) != value for key, value in metadata.items())
                or [row["id"] for row in reviews] != [p["id"] for p in passages[:len(reviews)]]):
            raise ValueError("The review checkpoint does not match this plan and configuration.")
        model_info = saved["run"]["model"]
        print(f"Resuming with {len(reviews)}/{len(passages)} passages already reviewed.", flush=True)

    def result() -> dict:
        statuses = dict(Counter(row["status"] for row in reviews))
        return {"run": {**metadata, "model": model_info, "statuses": statuses}, "reviews": reviews}

    initial_count = min(check_first, len(passages))

    def check_initial_reviews() -> None:
        if (initial_count and len(reviews) >= initial_count
                and not any(row.get("accepted_count", 0) for row in reviews[:initial_count])):
            raise RuntimeError(f"No Qwen proposal was accepted in the first {initial_count} passages. "
                               f"Partial reviews: {checkpoint_path}.")

    check_initial_reviews()
    if len(reviews) < len(passages):
        model, tokenizer, model_info = load_model(settings, quantized=True)
        for number, passage in enumerate(passages[len(reviews):], len(reviews) + 1):
            started = perf_counter()
            review = review_relations(passage, settings, model, tokenizer)
            seconds = round(perf_counter() - started, 2)
            reviews.append({**{key: val for key, val in passage.items() if key != "text"}, **review,
                            "review_seconds": seconds})
            if checkpoint_path is not None:
                write_result(checkpoint_path, result())
            accepted = review.get("accepted_count", 0)
            print(f"Reviewed {number}/{len(passages)}: {passage['id']} "
                  f"({review['status']}, {accepted} accepted, {seconds:.1f}s)", flush=True)
            if number == initial_count:
                check_initial_reviews()
    return result()


def build_saved_graph(input_path: Path, review_path: Path | None, revalidate_reviews: bool,
                      *, context: bool = False, links_path: Path | None = None) -> dict:
    """Build a graph and record the fingerprints of the predictions and optional reviews."""
    if revalidate_reviews and review_path is None:
        raise ValueError("Revalidation requires --relation-reviews.")

    run, documents, input_sha256 = load_ner(input_path)
    core_chars = run.get("config", {}).get("passages", {}).get("core_chars", 800)
    reviewed, facts, review_metadata = {}, {}, {}
    if review_path is not None:
        content = review_path.read_bytes()
        review = json.loads(content.decode("utf-8-sig"))
        context = context or review["run"].get("profile") == "context"
        same_code = review["run"].get("code_sha256") == graph_code_hash(with_review=True)
        if not revalidate_reviews and not same_code:
            raise ValueError("The relation review was produced with different graph code.")
        reviewed = load_review_relations(review, documents, input_sha256, revalidate=revalidate_reviews,
                                         facts_by_document=facts, allow_positional=same_code)
        review_metadata = {"relation_review_sha256": hashlib.sha256(content).hexdigest(),
                           "relation_review": review["run"],
                           "relation_review_revalidated": revalidate_reviews,
                           "historical_positional_facts_allowed": same_code}

    link_config = {}
    if context:
        link_path = links_path or PROJECT_ROOT / "configs/graph_links.json"
        configuration = link_path.read_bytes()
        link_config = json.loads(configuration.decode("utf-8-sig"))
        review_metadata["link_config_sha256"] = hashlib.sha256(configuration).hexdigest()
    graph = build_graph(documents, core_chars, reviewed, context=context, reviewed_facts=facts, link_config=link_config)
    node_counts = Counter(node["type"] for node in graph["nodes"])
    edge_counts = Counter(edge["type"] for edge in graph["edges"])
    graph["run"] = {
        "stage": "relation_graph",
        "profile": "context" if context else "legacy",
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "code_sha256": graph_code_hash(with_review=review_path is not None),
        "input_stage": run["stage"],
        "input_scope": run.get("scope"),
        "core_chars": core_chars,
        "subject_scope": "document",
        "explicit_relation_method": ("rules_and_qwen" if review_path else
                                     "context_rules" if context else "role_representation_and_employment_patterns"),
        "descriptive_subject_count": sum(node.get("mention_kind") == "description" for node in graph["nodes"]),
        "nodes_by_type": dict(node_counts),
        "edges_by_type": dict(edge_counts),
        **review_metadata,
    }
    return graph


def _relation_key(source: dict, predicate: dict, target: dict) -> tuple:
    """Represent an occurrence by its directed endpoints and predicate span."""
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
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("prepare", help="Select passages and save a Qwen review plan.")
    plan.add_argument("--input", required=True, help="Saved first-pass or hybrid NER JSON.")
    plan.add_argument("--reference", help="Optional reference JSON. Uses only passage text and offsets.")
    plan.add_argument("--max-passages", type=int, default=100)
    plan.add_argument("--context", action="store_true", help="Select passages for broader contextual facts.")
    plan.add_argument("--output", required=True)

    review = commands.add_parser("review", help="Run Qwen on a prepared plan using CUDA.")
    review.add_argument("--plan", required=True)
    review.add_argument("--config", default="configs/hybrid.json")
    review.add_argument("--check-first", type=int, default=0,
                        help="Stop if none of the initial passages produces a supported model proposal.")
    review.add_argument("--output", required=True)

    build = commands.add_parser("build", help="Build the graph from saved predictions on CPU.")
    build.add_argument("--input", required=True, help="Saved first-pass or hybrid NER JSON.")
    build.add_argument("--relation-reviews", help="Saved Qwen responses to add to the graph.")
    build.add_argument("--revalidate-reviews", action="store_true",
                       help="Reapply current validation to saved complete Qwen attempts.")
    build.add_argument("--context", action="store_true", help="Include contextual facts and cross-document links.")
    build.add_argument("--links", help="Shared case and event references; defaults to configs/graph_links.json.")
    build.add_argument("--output", default="outputs/relation_graph.json")

    evaluate = commands.add_parser("evaluate", help="Score explicit edges against reviewed excerpts on CPU.")
    evaluate.add_argument("--graph", default="outputs/relation_graph.json")
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--diagnostic", action="store_true",
                          help="Use known excerpts to diagnose a changed method.")
    evaluate.add_argument("--output", default="outputs/relation_evaluation.json")

    score = commands.add_parser("score-context", help="Score contextual facts and reviewed document pairs.")
    score.add_argument("--graph", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)

    inspect = commands.add_parser("inspect", help="Export a document or shared node with original evidence.")
    inspect.add_argument("--graph", required=True)
    inspect.add_argument("--input", required=True, help="The same saved NER used for this graph.")
    inspect.add_argument("--node", required=True, help="A document or graph node identifier.")
    inspect.add_argument("--edge-type", action="append", default=[], help="Keep this edge type (repeatable).")
    inspect.add_argument("--output", required=True)

    for command in (plan, review, build, evaluate, score, inspect):
        command.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()

    try:
        output = project_path(args.output)
        input_names = ("input", "reference", "plan", "config", "relation_reviews", "graph", "links")
        input_paths = {
            name: project_path(value)
            for name in input_names
            if (value := getattr(args, name, None))
        }
        if output in input_paths.values() or output.suffix.lower() != ".json":
            raise ValueError("Choose a JSON output distinct from every input.")
        if output.exists() and not args.overwrite:
            raise FileExistsError("Output already exists. Choose another path or use --overwrite.")

        checkpoint = None
        if args.command == "prepare":
            result = prepare(input_paths["input"], input_paths.get("reference"), args.max_passages,
                             context=args.context)
        elif args.command == "review":
            checkpoint = output.with_suffix(".partial.json")
            if checkpoint in input_paths.values():
                raise ValueError("The review checkpoint path conflicts with an input file.")
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite:
                checkpoint.unlink(missing_ok=True)
            result = run_review(input_paths["plan"], input_paths["config"],
                                checkpoint_path=checkpoint, check_first=args.check_first)
        elif args.command == "build":
            result = build_saved_graph(input_paths["input"], input_paths.get("relation_reviews"),
                                       args.revalidate_reviews, context=args.context or bool(args.links),
                                       links_path=input_paths.get("links"))
        else:
            graph = json.loads(input_paths["graph"].read_text(encoding="utf-8-sig"))
            if args.command == "inspect":
                _, documents, fingerprint = load_ner(input_paths["input"])
                if graph["run"]["input_sha256"] != fingerprint:
                    raise ValueError("Inspect requires the same saved NER used to build the graph.")
                result = inspect_graph(graph, documents, args.node, args.edge_type)
            else:
                reference = json.loads(input_paths["reference"].read_text(encoding="utf-8-sig"))
                if args.command == "score-context":
                    if graph["run"].get("profile") != "context":
                        raise ValueError("An expanded context graph is required.")
                    if reference.get("input_sha256") != graph["run"]["input_sha256"]:
                        raise ValueError("The context reference must match the graph's saved NER input.")
                    result = score_context(graph, reference)
                else:
                    result = score_graph(graph, reference, diagnostic=args.diagnostic)

        output.parent.mkdir(parents=True, exist_ok=True)
        write_result(output, result)
        if checkpoint is not None:
            checkpoint.unlink(missing_ok=True)
        if args.command in {"prepare", "review"}:
            run = result["run"]
            print(f"Selected {run['selected_passages']} of {run['eligible_passages']} eligible passages.")
        elif args.command == "build":
            print(f"Nodes: {result['run']['nodes_by_type']}")
            print(f"Edges: {result['run']['edges_by_type']}")
        elif args.command == "evaluate":
            print(result["overall"])
        elif args.command == "score-context":
            print(result["facts"]["overall"])
        print(f"Saved {output}")
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
