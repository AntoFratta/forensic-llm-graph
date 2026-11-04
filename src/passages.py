"""Build original-text passages and select candidates for language-model review."""

from collections import defaultdict
import math
import random
import re


def build_passages(documents: list, core_chars: int, context_chars: int) -> list:
    """Partition text into disjoint cores with context on both sides.

    An entity belongs to the core containing its start offset. Context can appear
    in neighbouring passages, but ownership prevents duplicate merge decisions.
    A word boundary is preferred near the target size. A final remainder shorter
    than a quarter of the target is absorbed into the preceding core, avoiding
    nearly empty review windows. Text is never normalized.
    Context is extended when an owned first-pass candidate would otherwise be
    cut; model token limits are checked separately rather than silently truncating.
    """
    if type(core_chars) is not int or core_chars < 100:
        raise ValueError("core_chars must be an integer of at least 100.")
    if type(context_chars) is not int or context_chars < 0:
        raise ValueError("context_chars must be a non-negative integer.")
    passages = []
    for document in documents:
        text = document["text"]
        core_start = 0
        while core_start < len(text):
            core_end = min(core_start + core_chars, len(text))
            if core_end < len(text):
                boundary = next((position for position in range(core_end, core_start + core_chars // 2, -1)
                                 if text[position].isspace()), None)
                if boundary is not None:
                    core_end = boundary
                if len(text) - core_end < core_chars // 4:
                    core_end = len(text)
            owned = [entity for entity in document["entities"]
                     if core_start <= entity["start"] < core_end]
            start = max(0, core_start - context_chars)
            end = min(len(text), core_end + context_chars)
            if owned:
                end = max(end, max(entity["end"] for entity in owned))
            passages.append({
                "id": f"{document['id']}@{core_start}:{core_end}",
                "document_id": document["id"], "source": document.get("source", "unknown"),
                "start": start, "end": end, "core_start": core_start, "core_end": core_end,
                "text": text[start:end], "candidates": owned,
            })
            core_start = core_end
    return passages


def smoke_sample(passages: list, per_source: int, seed: int) -> list:
    """Sample a few passages per collection, preferring distinct documents.

    This is a reproducible technical check, not a development/test split or an
    annotated evaluation sample. Selection does not inspect reference labels.
    """
    if type(per_source) is not int or per_source < 1:
        raise ValueError("smoke_per_source must be a positive integer.")
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for passage in passages:
        groups[passage["source"]][passage["document_id"]].append(passage)
    selected = set()
    for source in sorted(groups):
        document_ids = sorted(groups[source])
        rng.shuffle(document_ids)
        choices = [rng.choice(groups[source][identifier]) for identifier in document_ids[:per_source]]
        if len(choices) < per_source:
            used = {passage["id"] for passage in choices}
            remaining = [p for identifier in document_ids for p in groups[source][identifier]
                         if p["id"] not in used]
            rng.shuffle(remaining)
            choices.extend(remaining[:per_source - len(choices)])
        selected.update(passage["id"] for passage in choices)
    return [passage for passage in passages if passage["id"] in selected]


def select_passages(passages: list, scores: dict, settings: dict,
                    smoke_test: bool = False, use_perplexity: bool = True) -> dict:
    """Combine heuristic review signals and perplexity under an explicit budget.

    A fixed threshold is required for an extended run. The sample percentile is
    allowed only for the technical smoke test and must not become a test-set
    calibration. Review priorities are heuristic, not calibrated confidence.
    """
    threshold = settings["perplexity_threshold"] if use_perplexity else None
    threshold_mode = "fixed" if use_perplexity else "disabled"
    if not use_perplexity:
        pass
    elif threshold is None:
        if not smoke_test:
            raise ValueError("Set a development-calibrated perplexity_threshold before an extended run.")
        values = sorted(score["perplexity"] for score in scores.values()
                        if score["status"] == "ok" and score["perplexity"] is not None)
        fraction = settings["diagnostic_percentile"]
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction < 1:
            raise ValueError("diagnostic_percentile must be between zero and one.")
        threshold = values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None
        threshold_mode = "smoke_sample_percentile"
    elif isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("perplexity_threshold must be positive and finite.")

    fraction = settings["max_review_fraction"]
    maximum = settings["max_review_passages"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("max_review_fraction must be greater than zero and at most one.")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_review_passages must be a positive integer.")
    strong_signals = {"overlapping_candidates", "possible_form_fields_in_name",
                      "name_without_proper_noun", "legal_reference_as_name",
                      "common_form_term_as_name", "unexpected_digit_count"}
    decisions = {}
    for passage in passages:
        score = scores[passage["id"]]
        reasons = []
        flags = {reason for entity in passage["candidates"] for reason in entity["review_reasons"]}
        priority = 0
        if flags:
            reasons.append("first_pass_review_flags")
            priority += 1
        if flags & strong_signals:
            reasons.append("conflicting_or_suspect_candidates")
            priority += 3
        if not passage["candidates"] and sum(char.isalpha() for char in passage["text"]) >= 80:
            reasons.append("possible_omission")
            priority += 1
        if use_perplexity and score["status"] != "ok":
            reasons.append("perplexity_unavailable")
            priority += 1
        elif use_perplexity and threshold is not None and score["perplexity"] >= threshold:
            reasons.append("high_perplexity")
            priority += 2
        decisions[passage["id"]] = {"eligible": bool(reasons), "selected": False,
                                     "reasons": reasons, "priority": priority}
    eligible = [p for p in passages if decisions[p["id"]]["eligible"]]
    eligible.sort(key=lambda p: (-decisions[p["id"]]["priority"],
                                 -(scores[p["id"]].get("mean_nll") or 0) if use_perplexity else 0,
                                 p["id"]))
    budget = min(maximum, math.ceil(len(passages) * fraction))
    for passage in eligible[:budget]:
        decisions[passage["id"]]["selected"] = True
    if smoke_test:
        # Exercise both models on every sampled passage, without claiming selective performance.
        for decision in decisions.values():
            decision["would_select"] = decision["selected"]
            decision["selected"] = True
            decision["reasons"] = [*decision["reasons"], "smoke_test_coverage"]
    return {"threshold": threshold, "threshold_mode": threshold_mode,
            "budget": budget, "decisions": decisions}
