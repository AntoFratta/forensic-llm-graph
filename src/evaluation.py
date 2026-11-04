"""Read reviewed references and score exact entity spans in their owned cores."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from src.llm_filter import LABELS


def load_reference(path: Path) -> tuple[dict, str]:
    """Require an explicit human review and preserve document-global offsets.

    Reference labels are used for evaluation only. Passage selection below
    reads identities and boundaries, never entity labels or annotation notes.
    """
    content = path.read_bytes()
    reference = json.loads(content.decode("utf-8-sig"))
    if (not isinstance(reference, dict) or reference.get("schema_version") != 1
            or reference.get("split") not in {"development", "test"}
            or not isinstance(reference.get("units"), list) or not reference["units"]):
        raise ValueError("Expected a version-1 development or test reference with units.")
    if reference["split"] == "test" and reference.get("split_frozen") is not True:
        raise ValueError("Freeze the test grouping and selection before evaluating it.")
    seen = set()
    cores = defaultdict(list)
    for unit in reference["units"]:
        if (unit.get("annotation_status") != "reviewed" or not unit.get("human_review")
                or any(issue.get("status") == "open" for issue in unit.get("issues", []))):
            raise ValueError("Every reference unit must be human-reviewed, without open issues.")
        if unit.get("split") != reference["split"]:
            raise ValueError("Do not mix development and test reference units.")
        if not isinstance(unit.get("id"), str) or unit["id"] in seen:
            raise ValueError("Reference unit IDs must be unique strings.")
        seen.add(unit["id"])
        if not isinstance(unit.get("document_id"), str) or not isinstance(unit.get("text"), str):
            raise ValueError("Reference units require document IDs and original text.")
        offsets = [unit.get(key) for key in ("start", "core_start", "core_end", "end")]
        if (any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] < offsets[2] <= offsets[3]
                or len(unit["text"]) != unit["end"] - unit["start"]):
            raise ValueError("Invalid reference window or core boundaries.")
        if hashlib.sha256(unit["text"].encode()).hexdigest() != unit.get("text_sha256"):
            raise ValueError("Reference text hash does not match its original text.")
        cores[unit["document_id"]].append((unit["core_start"], unit["core_end"]))
        if not isinstance(unit.get("entities"), list):
            raise ValueError("Reference entities must be a list, including verified negatives.")
        keys = set()
        for entity in unit["entities"]:
            start, end = entity.get("start"), entity.get("end")
            if (type(start) is not int or type(end) is not int
                    or not unit["core_start"] <= start < unit["core_end"]
                    or not start < end <= unit["end"] or entity.get("label") not in LABELS
                    or unit["text"][start - unit["start"]:end - unit["start"]] != entity.get("text")):
                raise ValueError("Invalid reference entity span or category.")
            key = (start, end, entity["label"])
            if key in keys:
                raise ValueError("Duplicate entity in a reference unit.")
            keys.add(key)
    for intervals in cores.values():
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Reference cores must not overlap within a document.")
    return reference, hashlib.sha256(content).hexdigest()


def reference_passages(reference: dict, documents: list, passages: list, settings: dict) -> list:
    """Select operational windows matching the checked reference cores.

    Existing candidates come solely from the recorded first pass. Context may
    extend beyond the reference window to keep a detected entity intact.
    """
    if any(reference.get(key) != settings[key] for key in ("core_chars", "context_chars")):
        raise ValueError("Reference and operational passage sizes must agree before scoring perplexity.")
    by_document = {document["id"]: document for document in documents}
    by_passage = {passage["id"]: passage for passage in passages}
    selected = []
    for unit in reference["units"]:
        document = by_document.get(unit["document_id"])
        if (document is None or document.get("source") != unit.get("source")
                or document["text"][unit["start"]:unit["end"]] != unit["text"]):
            raise ValueError("Reference text or source differs from the first-pass document.")
        passage = by_passage.get(unit["id"])
        if (passage is None or passage["core_start"] != unit["core_start"]
                or passage["core_end"] != unit["core_end"]):
            raise ValueError("Reference cores do not match the operational passage partition.")
        selected.append(passage)
    return selected


def _metrics(counts: Counter) -> dict:
    tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn, "gold_mentions": tp + fn,
            "predicted_mentions": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def score_reference(reference: dict, documents: list) -> dict:
    """Count exact (start, end, label) matches; score each core once.

    A boundary or label error contributes both a false positive and a false
    negative. Synthetic diagnostics are reported separately from REDIT forms, news and judicial texts.
    Missing predictions are errors, not grounds to exclude a reviewed unit.
    """
    by_document = {document["id"]: document for document in documents}
    buckets = defaultdict(lambda: defaultdict(Counter))
    unit_results = []
    for unit in reference["units"]:
        document = by_document.get(unit["document_id"])
        if document is None:
            raise ValueError(f"Missing prediction document: {unit['document_id']}")
        if document["text"][unit["start"]:unit["end"]] != unit["text"]:
            raise ValueError("Predictions and reference must refer to identical original text.")
        gold = {(e["start"], e["end"], e["label"]) for e in unit["entities"]}
        predicted = set()
        for entity in document["entities"]:
            start, end = entity["start"], entity["end"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(document["text"])
                    or document["text"][start:end] != entity["text"] or entity["label"] not in LABELS):
                raise ValueError("Prediction entities require valid original-text spans and categories.")
            if unit["core_start"] <= start < unit["core_end"]:
                predicted.add((start, end, entity["label"]))
        matches, extra, missing = gold & predicted, predicted - gold, gold - predicted
        group = "synthetic" if unit["source"] == "rizzo_synthetic" else "source_documents"
        for metric, values in (("tp", matches), ("fp", extra), ("fn", missing)):
            for _, _, label in values:
                for bucket in (group, "source:" + unit["source"]):
                    buckets[bucket][label][metric] += 1
        # Retain groups even for a genuinely empty, correctly negative unit.
        buckets[group]
        buckets["source:" + unit["source"]]

        def describe(values):
            return [{"start": start, "end": end, "label": label,
                     "text": document["text"][start:end]} for start, end, label in sorted(values)]

        unit_results.append({"reference_id": unit["reference_id"], "passage_id": unit["id"],
                             "source": unit["source"], "tp": len(matches),
                             "false_positives": describe(extra), "false_negatives": describe(missing)})
    summary = {}
    for name, labels in sorted(buckets.items()):
        total = Counter()
        for counts in labels.values():
            total.update(counts)
        summary[name] = {"unit_count": sum(
            (u["source"] == name[7:] if name.startswith("source:")
             else (u["source"] == "rizzo_synthetic") == (name == "synthetic"))
            for u in reference["units"]),
            "micro": _metrics(total),
            "by_category": {label: _metrics(labels[label]) for label in sorted(LABELS)}}
    return {"matching": "exact_start_end_label; entity_owned_by_start_offset",
            "split": reference["split"], "unit_count": len(reference["units"]),
            "group_definition": "source_documents = REDIT forms, DICE news and ArchivioAntimafia; synthetic = Rizzo generated texts. REDIT uses fictitious personal values.",
            "groups": summary, "units": unit_results}
