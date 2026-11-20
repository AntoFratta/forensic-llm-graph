"""Read local graph evidence and score annotated facts and document pairs."""

from collections import Counter, defaultdict

from src.evaluation import _metrics


def inspect_graph(graph: dict, documents: list, identifier: str, edge_types: list) -> dict:
    nodes = {n["id"]: n for n in graph["nodes"]}
    target = nodes.get(identifier) or nodes.get(f"document:{identifier}")
    if target is None:
        raise ValueError(f"Graph node not found: {identifier}.")
    if target["type"] == "DOCUMENT":
        local = {n["id"] for n in nodes.values() if n.get("document_id") == target["document_id"]}
    else:
        local = {target["id"]}
    texts = {d["id"]: d["text"] for d in documents}
    edges = []
    visible = local.copy()
    for edge in graph["edges"]:
        if edge_types and edge["type"] not in edge_types:
            continue
        if edge["source"] not in local and edge["target"] not in local:
            continue
        visible.update((edge["source"], edge["target"]))
        quotes = []
        for evidence in edge["evidence"]:
            row = dict(evidence)
            if "start" in row:
                row["quote"] = texts[row["document_id"]][row["start"]:row["end"]]
            quotes.append(row)
        edges.append({**edge, "evidence": quotes})
    candidates = [row for row in graph.get("link_candidates", []) if target.get("document_id") in row["documents"]]
    return {"focus": target, "nodes": [nodes[node_id] for node_id in sorted(visible)], "edges": edges,
            "link_candidates": candidates}


def _fact_signature(kind: str, start: int, end: int, assertion: str, arguments: dict) -> tuple:
    return (kind, start, end, assertion,
            tuple(sorted((role, e["start"], e["end"], e["label"]) for role, e in arguments.items())))


def _differences(predicted: set, expected: set) -> dict:
    return {"tp": len(predicted & expected), "fp": len(predicted - expected), "fn": len(expected - predicted)}


def _counts_and_metrics(counts: Counter, item_name: str) -> dict:
    result = _metrics(counts)
    result[f"gold_{item_name}"] = result.pop("gold_mentions")
    result[f"predicted_{item_name}"] = result.pop("predicted_mentions")
    return result


def score_context(graph: dict, reference: dict) -> dict:
    """Score exact fact and argument spans, plus same-hub membership for reviewed pairs."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    document_ids = {n["document_id"] for n in nodes.values() if n["type"] == "DOCUMENT"}
    occurrence_facts, event_arguments, hubs = defaultdict(set), defaultdict(dict), defaultdict(set)
    hub_kind = {"REFERS_TO_CASE": "case", "REFERS_TO_EVENT": "event", "REFERS_TO_IDENTITY": "identity"}
    for edge in graph["edges"]:
        kind, source, target = edge["type"], nodes[edge["source"]], nodes[edge["target"]]
        if kind == "CONTEXT_RELATION":
            for ev in edge["evidence"]:
                args = {"subject": {**ev["source_mention"], "label": source["label"]},
                        "object": {**ev["target_mention"], "label": target["label"]}}
                occurrence_facts[ev["document_id"]].add(_fact_signature(
                    edge["relation"], ev["predicate_start"], ev["predicate_end"], ev["assertion"], args))
        elif kind == "FACT_ARGUMENT":
            ev = edge["evidence"][0]
            event_arguments[source["id"]][edge["role"]] = {**ev["mention"], "label": target["label"]}
        elif kind in hub_kind:
            hubs[(hub_kind[kind], target["id"])].add(source["document_id"])
    for edge in graph["edges"]:
        if edge["type"] == "STATED_IN":
            fact = nodes[edge["source"]]
            ev = edge["evidence"][0]
            occurrence_facts[ev["document_id"]].add(_fact_signature(
                fact["label"], ev["predicate_start"], ev["predicate_end"], ev["assertion"],
                event_arguments[fact["id"]]))

    counts, unit_scores = Counter(), []
    observed = defaultdict(list)
    for unit in reference.get("units", []):
        doc, start, end = unit["document_id"], unit["start"], unit["end"]
        if doc not in document_ids or type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError(f"Invalid context reference: {unit['id']}.")
        if any(start < other_end and other_start < end for other_start, other_end in observed[doc]):
            raise ValueError(f"Context reference excerpts overlap in {doc}.")
        observed[doc].append((start, end))
        expected = {_fact_signature(f["kind"], f["predicate_start"], f["predicate_end"], f["assertion"],
                                    f["arguments"]) for f in unit["facts"]}
        if any(not start <= key[1] < key[2] <= end for key in expected):
            raise ValueError(f"Reference predicate outside excerpt: {unit['id']}.")
        predicted = {key for key in occurrence_facts[doc] if start <= key[1] < key[2] <= end}
        delta = _differences(predicted, expected)
        counts.update(delta)
        unit_scores.append({"id": unit["id"], **delta})

    labelled_pairs, link_counts, pair_scores = set(), defaultdict(Counter), []
    for p in reference.get("document_pairs", []):
        kind, docs, expected = p["kind"], tuple(sorted(p["documents"])), p["linked"]
        key = (kind, docs)
        if (kind not in hub_kind.values() or len(docs) != 2 or docs[0] == docs[1]
                or not set(docs) <= document_ids or type(expected) is not bool or key in labelled_pairs):
            raise ValueError(f"Invalid reviewed document pair: {key}.")
        labelled_pairs.add(key)
        predicted = any(k == kind and set(docs) <= members for (k, _), members in hubs.items())
        label = ("tp" if expected else "fp") if predicted else ("fn" if expected else "tn")
        link_counts[kind][label] += 1
        pair_scores.append({**p, "predicted_linked": predicted, "result": label})
    if not unit_scores and not pair_scores:
        raise ValueError("Provide annotated facts or document pairs to score.")
    return {"facts": {"overall": _counts_and_metrics(counts, "facts"), "units": unit_scores},
            "document_links": {"by_kind": {k: {**_counts_and_metrics(v, "links"), "tn": v["tn"]}
                                          for k, v in link_counts.items()},
                               "pairs": pair_scores}}
