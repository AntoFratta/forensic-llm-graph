"""Compact wire format for graph proposals. All indexes are local to one passage."""

import re

from src.context_facts import CLAUSE_BOUNDARY, ground_proposal


_NUMBERED_PARAGRAPH = re.compile(r"\n(?=[ \t]*\d+[.)][ \t]+)")


def clauses(text):
    """Return original half-open spans, omitting only separators and outer whitespace."""
    separators = sorted(match.span() for pattern in (CLAUSE_BOUNDARY, _NUMBERED_PARAGRAPH)
                        for match in pattern.finditer(text))
    result = []
    start = 0
    for end, next_start in [*separators, (len(text), len(text))]:
        part = text[start:end]
        if part.strip():
            result.append((start + len(part) - len(part.lstrip()), end - len(part) + len(part.rstrip())))
        start = max(start, next_start)
    return result


def _numbered(sequence, number, name):
    if type(number) is not int or not 1 <= number <= len(sequence):
        raise ValueError(f"Select a supplied {name} number (starting at 1).")
    return sequence[number - 1]


def _participant(ref, passage, bounds, *, endpoint=False):
    span = None
    kind = None
    if type(ref) is int:
        entry = _numbered(passage["context_mentions"], ref, "mention")
        text, label = entry["text"], entry["label"]
        span = entry["start"], entry["end"]
        lo, hi = bounds
        base = passage["start"]
        if (not base + lo <= span[0] < span[1] <= base + hi
                or passage["text"][span[0] - base:span[1] - base] != text):
            raise ValueError("The referenced mention must occur in the predicate clause.")
        kind = entry.get("mention_kind")
    elif (isinstance(ref, list) and len(ref) in ((2, 3) if endpoint else (2,))
          and all(isinstance(field, str) and field.strip() for field in ref)):
        text, label = ref[:2]
        if len(ref) == 3:
            kind = ref[2]
    else:
        raise ValueError('Use a mention number or ["literal text","LABEL"] for a participant.')
    value = {"text": text, "label": label}
    if endpoint:
        supported_name = span and {"start": span[0], "end": span[1], "label": label} in passage.get("known_names", [])
        value["kind"] = kind or ("name" if text[:1].isupper() or supported_name else "description")
    return value, span


def ground_fact(row, passage, parts, locate, *, predicate_span=None):
    if not isinstance(row, list) or len(row) != 5 or not isinstance(row[4], dict):
        raise ValueError("A fact needs [clause, type, predicate, assertion, arguments].")
    number, kind, predicate, assertion, refs = row
    lo, hi = _numbered(parts, number, "clause")
    quote = passage["text"][lo:hi]
    arguments, spans = {}, {}
    for role, ref in refs.items():
        arguments[role], spans[role] = _participant(ref, passage, (lo, hi))
    local = {**passage, "text": quote, "start": passage["start"] + lo}
    item = {"kind": kind, "predicate": predicate, "assertion": assertion, "evidence": quote, "arguments": arguments}
    return ground_proposal(item, local, locate, argument_spans=spans, predicate_span=predicate_span)


def expand_role(row, passage, parts, cues):
    if not isinstance(row, list) or len(row) != 3:
        raise ValueError("A role relation needs [cue, source, target].")
    cue = _numbered(cues, row[0], "cue")
    bounds = next(((lo, hi) for lo, hi in parts if lo <= cue[0] < cue[1] <= hi), None)
    if bounds is None:
        raise ValueError("The selected cue must belong to one clause.")
    source, source_span = _participant(row[1], passage, bounds, endpoint=True)
    target, target_span = _participant(row[2], passage, bounds, endpoint=True)
    return {"source": source, "target": target}, bounds, cue, source_span, target_span
