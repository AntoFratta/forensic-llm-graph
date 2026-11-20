"""Connect local document graphs using explicit case references and identifiers."""

from collections import defaultdict
from datetime import date
from hashlib import sha256
from itertools import combinations
import re
import unicodedata


_DOCKET = re.compile(
    r"\b(?:Procura della Repubblica|Tribunale)\s+di\s+"
    r"(?P<city>[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+){0,3})"
    r"[\s,:(]*(?:(?:procedimento|proc\.)\s+)?(?:n\.\s*)?"
    r"(?P<number>\d{1,8})\s*/\s*(?P<year>19\d{2}|20\d{2})\s+R\.?\s*G\.?\s*N\.?\s*R\.?\b"
)


def _evidence(document_id: str, start: int, end: int) -> dict:
    return {"document_id": document_id, "start": start, "end": end}


def _find_anchor(text: str, anchor: str):
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, anchor.split())) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE)


def _data_scope(document: dict) -> str:
    if document.get("synthetic") or document.get("source") in {"redit", "rizzo_synthetic"}:
        return "synthetic"
    return "source"


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def _publication_date(document: dict):
    value = document.get("publication_date")
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _candidate_pairs(documents: list, memberships: dict) -> list:
    """Surface shared-name and title cues separately from confirmed graph edges."""
    names, titles, dates = defaultdict(dict), defaultdict(list), {}
    for doc in documents:
        identifier, scope = doc["id"], _data_scope(doc)
        dates[identifier] = _publication_date(doc)
        if title := _normalized(doc.get("title") or ""):
            titles[(scope, title)].append(identifier)
        if not dates[identifier]:
            continue
        for e in doc["entities"]:
            if e["label"] not in {"PERSON", "COMPANY"} or e.get("needs_review"):
                continue
            value = _normalized(e["text"])
            if len(value.split()) >= 2 and len(value) >= 9:
                names[(scope, e["label"], value)].setdefault(identifier, _evidence(identifier, e["start"], e["end"]))
    candidates = defaultdict(list)
    def already_linked(a, b):
        return bool(memberships.get(a, set()) & memberships.get(b, set()))

    for group in titles.values():
        if len(group) > 30:
            continue
        for pair in combinations(sorted(group), 2):
            if not already_linked(*pair):
                candidates[pair].append({"signal": "same_normalized_title", "documents": list(pair)})
    by_pair = defaultdict(list)
    for (_, label, value), evidence in names.items():
        if not 2 <= len(evidence) <= 30:
            continue
        for a, b in combinations(sorted(evidence), 2):
            if not already_linked(a, b) and abs((dates[a] - dates[b]).days) <= 7:
                by_pair[(a, b)].append({"signal": "shared_name", "label": label, "value": value,
                                         "evidence": [evidence[a], evidence[b]]})
    for pair, signals in by_pair.items():
        if len(signals) >= 2:
            candidates[pair].extend(signals)
    return [{"documents": list(pair), "signals": signals} for pair, signals in sorted(candidates.items())]


def add_document_links(documents: list, graph: dict, link_config: dict, identifiers: list) -> None:
    """Keep local nodes distinct and create hubs only for supported shared references."""
    by_id = {d["id"]: d for d in documents}
    groups = defaultdict(dict)
    for document in documents:
        if case_id := document.get("case_id"):
            groups[("CASE", _data_scope(document), f"metadata:{case_id}")][document["id"]] = {
                "document_id": document["id"], "case_id": case_id, "basis": "source_metadata",
            }
        for match in _DOCKET.finditer(document["text"]):
            city, number, year = (match.group(key) for key in ("city", "number", "year"))
            key = f"rgnr:{_normalized(city)}:{int(number)}:{year}"
            doc = document["id"]
            groups[("CASE", _data_scope(document), key)].setdefault(doc, {
                **_evidence(doc, *match.span()), "basis": "explicit_jurisdiction_and_docket",
            })

    names = {}
    known_links = set()
    for spec in link_config.get("references", []):
        kind, key, name = spec["type"], spec["id"], spec["name"]
        if kind not in {"CASE", "SHARED_EVENT"} or not key or (kind, key) in known_links:
            raise ValueError(f"Invalid shared reference: {key}.")
        known_links.add((kind, key))
        names[(kind, key)] = name
        scoped = spec.get("documents")
        candidates = documents if scoped is None else (by_id[doc_id] for doc_id in scoped if doc_id in by_id)
        anchors = spec["anchors"]
        if not anchors or not all(isinstance(a, str) and a.strip() for a in anchors):
            raise ValueError(f"Shared reference {key} needs source anchors.")
        for document in candidates:
            if scoped is None and document.get("source") not in spec.get("sources", ()):
                continue
            text = document["text"]
            match = next((m for anchor in anchors if (m := _find_anchor(text, anchor))), None)
            if match:
                doc = document["id"]
                groups[(kind, _data_scope(document), key)][doc] = {
                    **_evidence(doc, *match.span()), "basis": "case_or_event_reference",
                }

    memberships = defaultdict(set)
    for (kind, scope, key), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        node_id = f"{kind.lower()}:{scope}:{key}"
        graph["nodes"].append({"id": node_id, "type": kind, "name": names.get((kind, key), key),
                               "data_scope": scope})
        edge_type = "REFERS_TO_CASE" if kind == "CASE" else "REFERS_TO_EVENT"
        for doc, evidence in sorted(members.items()):
            memberships[doc].add(node_id)
            graph["edges"].append({"type": edge_type, "source": f"document:{doc}", "target": node_id,
                                   "evidence": [evidence]})

    identities = defaultdict(list)
    for subject, fact, doc in identifiers:
        holder, value = fact["arguments"]["subject"], fact["arguments"]["object"]
        if holder.get("mention_kind") == "description" or holder.get("needs_review"):
            continue
        normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value["text"])).upper()
        if value["label"] == "TAX_CODE" and holder["label"] == "PERSON":
            if not re.fullmatch(r"[A-Z]{6}[A-Z0-9]{10}", normalized):
                continue
        elif value["label"] == "VAT_NUMBER" and holder["label"] == "COMPANY":
            if not re.fullmatch(r"\d{11}", normalized):
                continue
        else:
            continue
        name = " ".join(unicodedata.normalize("NFC", holder["text"]).casefold().split())
        identities[(_data_scope(by_id[doc]), holder["label"], value["label"], normalized)].append(
            (subject, name, holder["text"], _evidence(doc, fact["start"], fact["end"]))
        )

    for (scope, label, identifier_type, value), occurrences in sorted(identities.items()):
        if len({item[3]["document_id"] for item in occurrences}) < 2 or len({item[1] for item in occurrences}) != 1:
            continue
        digest = sha256(f"{scope}:{identifier_type}:{value}".encode()).hexdigest()
        node_id = f"identity:{digest}"
        graph["nodes"].append({"id": node_id, "type": "IDENTITY", "label": label,
                               "name": occurrences[0][2], "data_scope": scope})
        by_subject = defaultdict(list)
        for subject, _, _, evidence in occurrences:
            by_subject[subject].append({**evidence, "basis": "explicit_shared_identifier",
                                        "identifier_type": identifier_type})
        for subject, evidence in sorted(by_subject.items()):
            graph["edges"].append({"type": "REFERS_TO_IDENTITY", "source": subject, "target": node_id,
                                   "evidence": evidence})

    # Explicitly reviewed identities are scoped to a documented common case.
    mentions = {edge["source"]: edge["evidence"] for edge in graph["edges"] if edge["type"] == "MENTIONED_IN"}
    subjects = defaultdict(list)
    for node in graph["nodes"]:
        if node["type"] == "SUBJECT" and node["mention_kind"] == "name":
            subjects[(node["document_id"], node["label"], _normalized(node["name"]))].append(node)
    declared = set()
    for spec in link_config.get("reviewed_subjects", []):
        name, label, key = spec["name"], spec["label"], spec["id"]
        if not key or key in declared or label not in {"PERSON", "COMPANY"}:
            raise ValueError(f"Invalid reviewed subject: {key}.")
        declared.add(key)
        for scope in ("source", "synthetic"):
            case_node = f"case:{scope}:{spec['case_id']}"
            each = {}
            for doc in spec["documents"]:
                if case_node not in memberships.get(doc, ()):
                    continue
                for subject in subjects[(doc, label, _normalized(name))]:
                    each[subject["id"]] = [{**item, "basis": "reviewed_subject_in_shared_case",
                                            "case_node": case_node} for item in mentions[subject["id"]]]
            if len({items[0]["document_id"] for items in each.values()}) < 2:
                continue
            node_id = f"identity:{scope}:reviewed:{key}"
            graph["nodes"].append({"id": node_id, "type": "IDENTITY", "label": label, "name": name,
                                   "data_scope": scope})
            for subject, evidence in sorted(each.items()):
                graph["edges"].append({"type": "REFERS_TO_IDENTITY", "source": subject, "target": node_id,
                                       "evidence": evidence})
    graph["link_candidates"] = _candidate_pairs(documents, memberships)
