"""Extract explicit role relationships with evidence from the document text."""

import re


_ARTICLE = r"(?:(?:un|una|uno|il|la|lo)\s+|l['’])"
_PLACE_WORD = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[-'’][A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+)*"
_PERSON_NOUN = (
    r"(?:\d{1,3}\s*-?\s*enne|(?:minor|maggior|diciott|diciannov|vent|trent|"
    r"quarant|cinquant|sessant|settant|ottant|novant)enne|"
    r"uomo|donna|ragazz[oa]|giovane|anzian[oa]|imputat[oa]|indagat[oa]|"
    r"testimone|dipendente|client[ea]|consulente|dirigente|medic[oa]|operai[oa]|soci[oa])"
)
_ORIGIN = (
    rf"(?:(?:\s+(?i:residente\s+a|originari[oa]\s+di)|"
    rf"(?i:(?:(?<=enne)|(?<=uomo)|(?<=donna)|(?<=ragazzo)|(?<=ragazza)))\s+(?i:di))\s+{_PLACE_WORD}"
    rf"(?:\s+{_PLACE_WORD}){{0,2}})?"
)
_COMPANY_NOUN = (
    r"(?:centro\s+(?:benessere|estetico|commerciale|fitness)|"
    r"rivendita(?:\s+di(?:\s+(?!(?:e|ed|o|che|ha)\b)(?-i:[a-zà-öø-ÿ]+)){1,2})?|"
    r"(?:società|azienda|ditta|impresa|cooperativa)"
    r"(?:\s+(?:edile|agricola|commerciale|assicurativa|finanziaria|locale|privata)\b)?)\b"
)
_PERSON_DESCRIPTION = re.compile(rf"\b(?i:{_ARTICLE}{_PERSON_NOUN})\b{_ORIGIN}")
_COMPANY_DESCRIPTION = re.compile(rf"\b{_ARTICLE}{_COMPANY_NOUN}", re.IGNORECASE)
_BARE_DESCRIPTION = {
    "PERSON": re.compile(rf"(?i:{_PERSON_NOUN})\b{_ORIGIN}(?=\s*(?:[.,;:!?)»”]|$))"),
    "COMPANY": re.compile(rf"{_COMPANY_NOUN}(?=\s*(?:[.,;:!?)»”]|$))", re.IGNORECASE),
}
_RELATION = re.compile(
    r"\b(?:(?P<employment>lavor(?:a|ava|ò|erà)\s+(?:presso|per))\b\s*|"
    r"(?P<role>(?:ex\s+)?(?:(?P<legal>avvocat[oa]|difensor[ea])|"
    r"(?:amministratore|amministratrice)(?:\s+(?:delegat[oa]|unic[oa]))?|legale\s+rappresentante|"
    r"rappresentante\s+legale|presidente|direttor[ea]|direttrice|dipendente))"
    r"\s+(?P<link>dell['’]|(?:di|del|della|dello|dei|degli|delle)\b)\s*)",
    re.IGNORECASE,
)
_SUBJECT_LINK = re.compile(
    r"\s*,?\s*(?:(?:è|era|fu|è\s+stat[oa])\s+)?"
    r"(?:(?:il|la|lo|un|una|uno)\s+|l['’])?",
    re.IGNORECASE,
)
_TARGET_LINK = re.compile(r'\s*["«“]?\s*')
_COORDINATION = re.compile(r'\s*["»”]?\s+(?:e|ed)\s+', re.IGNORECASE)
_ROLE_ARTICLE = re.compile(rf"\b{_ARTICLE}$", re.IGNORECASE)
_NON_ASSERTED_ROLE = re.compile(
    r"\b(?:non|chi)\s+(?:(?:è|era|fu|sia|sarebbe)(?:\s+stat[oa])?\s+)?$"
    r"|\b(?:sarebbe|potrebbe\s+essere)\s+$",
    re.IGNORECASE,
)


def _description(match: re.Match, label: str) -> dict:
    return {
        "start": match.start(), "end": match.end(), "text": match.group(),
        "label": label, "mention_kind": "description", "needs_review": False,
    }


def _descriptions(text: str) -> list:
    """Find local descriptions that can serve as relation endpoints."""
    mentions = []
    for pattern, label in ((_PERSON_DESCRIPTION, "PERSON"), (_COMPANY_DESCRIPTION, "COMPANY")):
        for match in pattern.finditer(text):
            mentions.append(_description(match, label))
    return mentions


def _subject(text: str, mentions: list, start: int) -> dict | None:
    candidates = [
        mention for mention in mentions
        if mention["label"] == "PERSON" and 0 <= start - mention["end"] <= 32
        and _SUBJECT_LINK.fullmatch(text[mention["end"]:start])
    ]
    return min(candidates, key=lambda mention: (
        mention.get("mention_kind") != "description", -mention["end"], mention["start"],
    ), default=None)


def _role_subject(text: str, start: int, end: int) -> dict | None:
    """Use a role as a local subject when it heads a noun phrase."""
    article = _ROLE_ARTICLE.search(text, max(0, start - 6), start)
    if article is None:
        return None
    first = article.start()
    if _NON_ASSERTED_ROLE.search(text, max(0, first - 32), first):
        return None
    return {"start": first, "end": end, "text": text[first:end], "label": "PERSON",
            "mention_kind": "description", "needs_review": False}


def _target(text: str, mentions: list, start: int, labels: set) -> dict | None:
    candidates = [
        mention for mention in mentions
        if mention["label"] in labels and 0 <= mention["start"] - start <= 8
        and _TARGET_LINK.fullmatch(text[start:mention["start"]])
    ]
    return min(candidates, key=lambda mention: (
        mention["start"], -mention["end"], mention.get("mention_kind") != "description",
    ), default=None)


def _targets(text: str, mentions: list, match: re.Match, labels: set):
    target = _target(text, mentions, match.end(), labels)
    link = match.group("link") or ""
    if link.casefold() in {"del", "della", "dello", "dell'", "dell’"}:
        # In "dell'azienda" the article is part of the preposition.
        for label in sorted(labels):
            descriptor = _BARE_DESCRIPTION[label].match(text, match.end())
            if descriptor and (target is None or (target["start"], target["end"]) == descriptor.span()):
                target = _description(descriptor, label)
                break
    while target is not None:
        yield target
        conjunction = _COORDINATION.match(text, target["end"])
        if conjunction is None:
            break
        target = _target(text, mentions, conjunction.end(), labels)


def extract_explicit_relations(text: str, named_mentions: list) -> list:
    """Extract directed relations for supported role and representation phrases.

    Endpoints must be adjacent to the predicate through the expected syntax.
    Descriptions are retained only when they participate in a relation.
    """
    predicates = list(_RELATION.finditer(text))
    if not predicates:
        return []
    mentions = [*named_mentions, *_descriptions(text)]
    relations = []
    for match in predicates:
        source = _subject(text, mentions, match.start())
        predicate_start, predicate_end = match.span("role" if match.group("role") else "employment")
        if source is None and match.group("role"):
            source = _role_subject(text, predicate_start, predicate_end)
        if source is None:
            continue
        targets = {"PERSON", "COMPANY"} if match.group("legal") else {"COMPANY"}
        for target in _targets(text, mentions, match, targets):
            relations.append({
                "source": source, "target": target,
                "predicate_start": predicate_start, "predicate_end": predicate_end,
            })
    return relations
