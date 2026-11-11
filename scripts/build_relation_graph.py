"""Build a subject and document graph from saved NER predictions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hybrid import write_result
from scripts.run_pipeline import project_path
from src.relation_graph import build_graph
from src.relation_review import load_review_relations


def graph_code_hash(with_review: bool = False) -> str:
    """Identify the extraction code used for an evaluation run."""
    digest = hashlib.sha256()
    files = ["src/explicit_relations.py", "src/relation_graph.py", "src/passages.py"]
    if with_review:
        files.append("src/relation_review.py")
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update((PROJECT_ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_ner(path: Path) -> tuple[dict, list, str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Saved first-pass or hybrid NER JSON.")
    parser.add_argument("--output", default="outputs/relation_graph.json")
    parser.add_argument("--relation-reviews", help="Saved Qwen responses to add to the graph.")
    parser.add_argument("--revalidate-reviews", action="store_true",
                        help="Reapply current validation to saved complete Qwen attempts.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing graph.")
    args = parser.parse_args()

    try:
        input_path, output_path = project_path(args.input), project_path(args.output)
        review_path = project_path(args.relation_reviews) if args.relation_reviews else None
        if args.revalidate_reviews and review_path is None:
            raise ValueError("Revalidation requires --relation-reviews.")
        if output_path in (input_path, review_path) or output_path.suffix.lower() != ".json":
            raise ValueError("Choose a JSON output distinct from the input.")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError("Output already exists. Choose another path or use --overwrite.")

        run, documents, input_sha256 = load_ner(input_path)
        core_chars = run.get("config", {}).get("passages", {}).get("core_chars", 800)
        reviewed, review_metadata = {}, {}
        if review_path is not None:
            content = review_path.read_bytes()
            review = json.loads(content.decode("utf-8-sig"))
            if (not args.revalidate_reviews
                    and review["run"].get("code_sha256") != graph_code_hash(with_review=True)):
                raise ValueError("The relation review was produced with different graph code.")
            reviewed = load_review_relations(review, documents, input_sha256, revalidate=args.revalidate_reviews)
            review_metadata = {"relation_review_sha256": hashlib.sha256(content).hexdigest(),
                               "relation_review": review["run"],
                               "relation_review_revalidated": args.revalidate_reviews}

        graph = build_graph(documents, core_chars, reviewed)
        node_counts = Counter(node["type"] for node in graph["nodes"])
        edge_counts = Counter(edge["type"] for edge in graph["edges"])
        graph["run"] = {
            "stage": "relation_graph",
            "input_path": str(input_path),
            "input_sha256": input_sha256,
            "code_sha256": graph_code_hash(with_review=review_path is not None),
            "input_stage": run["stage"],
            "input_scope": run.get("scope"),
            "core_chars": core_chars,
            "subject_scope": "document",
            "explicit_relation_method": "rules_and_qwen" if review_path else "role_representation_and_employment_patterns",
            "descriptive_subject_count": sum(node.get("mention_kind") == "description" for node in graph["nodes"]),
            "nodes_by_type": dict(node_counts),
            "edges_by_type": dict(edge_counts),
            **review_metadata,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_result(output_path, graph)
        print(f"Nodes: {dict(node_counts)}")
        print(f"Edges: {dict(edge_counts)}")
        print(f"Saved {output_path}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
