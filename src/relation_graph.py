"""Build a graph of subjects, documents and relationships grounded in text."""

from itertools import combinations
import unicodedata

from src.context_facts import context_mentions, extract_facts, fact_key, DIRECT
from src.document_links import add_document_links
from src.explicit_relations import extract_explicit_relations, is_subject_candidate
from src.passages import build_passages


SUBJECT_LABELS = {"PERSON", "COMPANY"}


def _span_key(mention: dict) -> tuple:
    return mention["start"], mention["end"], mention["label"]


def _subject_mentions(document: dict) -> list:
    """Return grounded, unique subject mentions in document order."""
    text = document["text"]
    unique = {}
    for entity in document["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("label"), str):
            raise ValueError(f"Invalid entity in {document['id']}.")
        if entity["label"] not in SUBJECT_LABELS:
            continue
        start, end = entity.get("start"), entity.get("end")
        value = entity.get("text")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(value, str) or not value.strip() or value != text[start:end]):
            raise ValueError(f"Subject mention does not match the text in {document['id']}.")
        if not isinstance(entity.get("needs_review"), bool):
            raise ValueError(f"Subject mention lacks review status in {document['id']}.")
        if not is_subject_candidate(value, entity["label"]):
            continue
        unique[_span_key(entity)] = entity
    return [unique[key] for key in sorted(unique)]


def _relation_mentions(named: list, relations: list) -> list:
    """Include grounded endpoints and suppress names nested in their descriptions."""
    mentions = {
        _span_key(mention): mention
        for relation in relations
        for mention in (relation["source"], relation["target"])
    }
    descriptions = [mention for mention in mentions.values() if mention.get("mention_kind") == "description"]
    for mention in named:
        if not any(description["label"] == mention["label"]
                   and description["start"] <= mention["start"]
                   and mention["end"] <= description["end"]
                   for description in descriptions):
            mentions.setdefault(_span_key(mention), mention)
    return sorted(mentions.values(), key=_span_key)


def _add_explicit_edges(text: str, relations: list, mentions: dict, edges: list) -> None:
    grouped, evidence_by_span = {}, {}
    for relation in relations:
        left = mentions.get(_span_key(relation["source"]))
        right = mentions.get(_span_key(relation["target"]))
        if left is None or right is None or left["subject"] == right["subject"]:
            continue
        start, end = relation["predicate_start"], relation["predicate_end"]
        predicate = " ".join(text[start:end].lower().split())
        key = (left["subject"], right["subject"], predicate)
        if key not in grouped:
            edge = {"type": "EXPLICIT_RELATION", "source": key[0], "target": key[1],
                    "relation": predicate, "evidence": []}
            grouped[key] = edge
            edges.append(edge)
        evidence_key = (_span_key(relation["source"]), start, end, _span_key(relation["target"]))
        if evidence_key in evidence_by_span:
            if relation.get("method") == "qwen":
                evidence_by_span[evidence_key]["qwen_reviewed"] = True
            continue
        evidence = {
            "document_id": left["document_id"], "start": min(left["start"], start, right["start"]),
            "end": max(left["end"], end, right["end"]),
            "predicate_start": start, "predicate_end": end,
            "source_mention": {"start": left["start"], "end": left["end"],
                               "needs_review": left["needs_review"]},
            "target_mention": {"start": right["start"], "end": right["end"],
                               "needs_review": right["needs_review"]},
        }
        if relation.get("method") == "qwen":
            evidence["method"] = "qwen"
        grouped[key]["evidence"].append(evidence)
        evidence_by_span[evidence_key] = evidence


def _add_context_facts(facts: list, mentions: dict, document_id: str, text: str,
                       nodes: list, edges: list, identifiers: list) -> None:
    direct = {kind for kind, *_ in DIRECT}
    for index, fact in enumerate(facts):
        start, end = fact["predicate_start"], fact["predicate_end"]
        predicate = " ".join(text[start:end].split())
        evidence = {"document_id": document_id, "start": fact["start"], "end": fact["end"],
                    "predicate_start": start, "predicate_end": end,
                    "assertion": fact["assertion"], "method": fact["method"]}
        if fact.get("qwen_reviewed"):
            evidence["qwen_reviewed"] = True
        arguments = {role: mentions[_span_key(entity)] for role, entity in fact["arguments"].items()}
        if fact["kind"] in direct:
            source, target = (arguments[k] for k in ("subject", "object"))
            if source["subject"] == target["subject"]:
                continue
            edge_evidence = {**evidence, "source_mention": {k: source[k] for k in ("start", "end")},
                             "target_mention": {k: target[k] for k in ("start", "end")}}
            if speaker := arguments.get("speaker"):
                edge_evidence["speaker"] = {k: speaker[k] for k in ("subject", "start", "end")}
            edges.append({"type": "CONTEXT_RELATION", "source": source["subject"], "target": target["subject"],
                          "relation": fact["kind"], "predicate": predicate, "evidence": [edge_evidence]})
            if fact["kind"] == "identifier" and fact["assertion"] == "asserted":
                identifiers.append((source["subject"], fact, document_id))
            continue

        event = f"fact:{document_id}:{start}:{end}:{index}"
        nodes.append({"id": event, "type": "FACT", "document_id": document_id, "label": fact["kind"],
                      "name": predicate, "assertion": fact["assertion"]})
        edges.append({"type": "STATED_IN", "source": event, "target": f"document:{document_id}",
                      "evidence": [evidence]})
        for role, mention in arguments.items():
            edges.append({"type": "FACT_ARGUMENT", "source": event, "target": mention["subject"],
                          "role": role, "evidence": [{**evidence, "mention": {
                              "start": mention["start"], "end": mention["end"],
                          }}]})


def build_graph(documents: list, core_chars: int = 800, reviewed_relations: dict | None = None,
                *, context: bool = False, reviewed_facts: dict | None = None,
                link_config: dict | None = None) -> dict:
    """Build the graph from the final entities of a saved NER run.

    Group named mentions by spelling and label within each document. Descriptions
    are tied to individual occurrences so two unnamed people are never merged
    merely because their descriptions are identical.
    """
    if not isinstance(documents, list) or not documents:
        raise ValueError("Expected a non-empty list of NER documents.")
    if type(core_chars) is not int or core_chars < 100:
        raise ValueError("Passage length must be an integer of at least 100 characters.")

    nodes, edges, seen_documents, identifiers = [], [], set(), []
    reviewed_relations = reviewed_relations or {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each NER document must be an object.")
        identifier = document.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen_documents:
            raise ValueError("Document identifiers must be non-empty and unique.")
        if not isinstance(document.get("text"), str) or not isinstance(document.get("entities"), list):
            raise ValueError(f"Document {identifier} requires text and entities.")
        seen_documents.add(identifier)
        document_node = f"document:{identifier}"
        nodes.append({
            "id": document_node,
            "type": "DOCUMENT",
            "document_id": identifier,
            **{key: document[key] for key in ("source", "document", "title", "source_file", "url",
                                             "case_id", "synthetic")
               if key in document},
        })

        subjects, mentions = {}, {}
        named = _subject_mentions(document)
        relations = extract_explicit_relations(document["text"], named)
        reviewed = reviewed_relations.get(identifier, ())
        by_span = {_span_key(entity): entity for entity in _relation_mentions(named, relations)}
        for relation in reviewed:
            for entity in (relation["source"], relation["target"]):
                by_span.setdefault(_span_key(entity), entity)
        facts = {}
        if context:
            contextual = context_mentions(document)
            descriptions = [e for e in by_span.values() if e.get("mention_kind") == "description"]
            contextual_by_span = by_span.copy()
            for entity in contextual:
                key = _span_key(entity)
                if entity["label"] in SUBJECT_LABELS and not is_subject_candidate(entity["text"], entity["label"]):
                    continue
                # Preserve full descriptions used by the legacy relations.
                if any(e["label"] == entity["label"] and e["start"] <= entity["start"]
                       and entity["end"] <= e["end"] and _span_key(e) != key for e in descriptions):
                    continue
                contextual_by_span.setdefault(key, entity)
                if entity.get("source") != "rule:context":
                    by_span.setdefault(key, entity)
            for fact in extract_facts(document, list(contextual_by_span.values())):
                facts[fact_key(fact)] = fact
                for entity in fact["arguments"].values():
                    by_span.setdefault(_span_key(entity), entity)
            for fact in (reviewed_facts or {}).get(identifier, ()):
                key = fact_key(fact)
                if key in facts:
                    facts[key]["qwen_reviewed"] = True
                else:
                    facts[key] = fact
                    for entity in fact["arguments"].values():
                        by_span.setdefault(_span_key(entity), entity)
        for _, entity in sorted(by_span.items()):
            start, end, label = entity["start"], entity["end"], entity["label"]
            kind = entity.get("mention_kind", "name")
            if kind == "description" or label in {"TIME", "ROLE"}:
                key = (kind, start, end, label)
            else:
                name = unicodedata.normalize("NFC", " ".join(entity["text"].split()).casefold())
                if context and label == "COMPANY":
                    name = name.rstrip(".")
                key = (kind, label, name)
            if key not in subjects:
                subject = f"subject:{identifier}:{start}:{end}:{label}"
                nodes.append({
                    "id": subject, "type": "SUBJECT" if label in SUBJECT_LABELS else "ENTITY", "label": label,
                    "name": entity["text"], "mention_kind": kind, "document_id": identifier,
                })
                edge = {"type": "MENTIONED_IN", "source": subject, "target": document_node, "evidence": []}
                edges.append(edge)
                subjects[key] = edge
            edge = subjects[key]
            evidence = {
                "document_id": identifier, "start": start, "end": end,
                "needs_review": entity["needs_review"],
            }
            edge["evidence"].append(evidence)
            mentions[_span_key(entity)] = {**evidence, "subject": edge["source"]}

        _add_explicit_edges(document["text"], [*relations, *reviewed], mentions, edges)
        if context:
            _add_context_facts(list(facts.values()), mentions, identifier, document["text"], nodes, edges, identifiers)
        co_occurrences = {}
        passage_document = {"id": identifier, "text": document["text"], "entities": list(mentions.values())}
        for passage in build_passages([passage_document], core_chars, context_chars=0):
            # Prefer unflagged evidence and require two distinct text spans.
            witnesses = {}
            for mention in passage["candidates"]:
                witnesses.setdefault(mention["subject"], []).append(mention)
            for candidates in witnesses.values():
                candidates.sort(key=lambda mention: mention["needs_review"])
            for source, target in combinations(sorted(witnesses), 2):
                pair = next(
                    ((left, right) for left in witnesses[source] for right in witnesses[target]
                     if left["end"] <= right["start"] or right["end"] <= left["start"]),
                    None,
                )
                if pair is None:
                    continue
                key = (source, target)
                if key not in co_occurrences:
                    edge = {"type": "CO_OCCURS_WITH", "source": source, "target": target, "evidence": []}
                    edges.append(edge)
                    co_occurrences[key] = edge
                co_occurrences[key]["evidence"].append({
                    "document_id": identifier,
                    "passage_id": passage["id"],
                    "start": passage["start"],
                    "end": passage["end"],
                    "mentions": [
                        {key: mention[key] for key in ("start", "end", "needs_review")}
                        for mention in pair
                    ],
                })
    graph = {"nodes": nodes, "edges": edges}
    if context:
        add_document_links(documents, graph, link_config or {}, identifiers)
    return graph
