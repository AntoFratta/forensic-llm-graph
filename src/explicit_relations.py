"""Extract explicit role relationships with evidence from the document text."""

import re
import unicodedata

from src.deterministic import _COMPANY_SUFFIX


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
    r"farmacia|pasticceria|filiale\s+di\s+banca|caseificio|raffineria|"
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
    r"\b(?:(?P<employment>lavor(?:a|ava|ò|erà)\s+(?:presso|per|in))\b\s*|"
    r"(?P<assistance>difensor[ea]|avvocat[oa])(?=\s+(?:Avv\.|di\s+fiducia\b))|"
    r"(?P<role>(?:ex\s+)?(?:(?P<legal>avvocat[oa](?:\s+difensor[ea])?|difensor[ea])|"
    r"amministrat(?:ore|rice)(?:\s+(?:delegat[oa]|unic[oa]|di\s+fatto))?|legale\s+rappresentante|"
    r"rappresentante\s+legale|presidente|direttor[ea]|direttrice|dipendente|titolare|liquidator[ea]))"
    r"(?:\s+(?P<link>dell['’]|(?:di|del|della|dello|dei|degli|delle)\b)\s*|"
    r"(?=\s+(?:e\s+poi\s+liquidator[ea]\b|con\s+(?:il\s+fratello|la\s+sorella)\b))))",
    re.IGNORECASE,
)
_SUBJECT_LINK = re.compile(
    r"\s*[,\(–-]?\s*(?:(?:che(?:\s+pure)?|quale|come|è|era|fu|è\s+stat[oa])\s+)?"
    r"(?:(?:il|la|lo|un|una|uno)\s+|l['’])?(?:semplice\s+)?"
    r"(?:amministrat(?:ore|rice)\s+(?:unic[oa]\s+)?e\s+poi\s+)?",
    re.IGNORECASE,
)
_TARGET_LINK = re.compile(r'\s*["«“]?\s*')
_COMPANY_TARGET_LINK = re.compile(r'\s*(?:società|ditta|azienda|impresa)\s+["«“]?\s*', re.I)
_LEGAL_TARGET_LIST = re.compile(r'\s*["«“]?\s*[^.;:!?\n]{2,56}\s+(?:e|ed)\s+["«“]?\s*', re.I)
_INITIAL_PAIR = r"[A-Z]\.[ \t]*[A-Z]\."
_INITIALS = re.compile(rf"{_INITIAL_PAIR}(?!\w|[ \t]*[A-Z]\.)")
_INITIAL_LIST_PREFIX = re.compile(rf"\s*(?:{_INITIAL_PAIR}\s*(?:,\s*|(?:e|ed)\s+))+")
_INITIAL_LIST_COMMA = re.compile(rf"\s*,\s*(?={_INITIALS.pattern})")
_COORDINATION = re.compile(r'\s*["»”]?\s+(?:e|ed)\s+', re.IGNORECASE)
_ROLE_ARTICLE = re.compile(rf"\b{_ARTICLE}$", re.IGNORECASE)
_NON_ASSERTED_ROLE = re.compile(
    r"\b(?:non|chi)\s+(?:(?:è|era|fu|sia|sarebbe)(?:\s+stat[oa])?\s+)?$"
    r"|\b(?:sarebbe|potrebbe\s+essere)\s+$",
    re.IGNORECASE,
)
_NON_COMPANY = re.compile(
    rf"^(?:{_ARTICLE})?(?:comune|provincia|regione|ministero|tribunale|corte|prefettura|questura|procura|stato|"
    r"consiglio|collegio(?:\s+giudicante|$)|comitato|commissione|ordine|csm|"
    r"azienda\s+(?:sanitaria|ospedaliera)|ausl|asl|dipartimento|pronto(?:\s+soccorso|$)|distretto|"
    r"ufficio\s+diocesano|affari\s+penali|(?:specifica\s+)?posizione\s+di\s+garanzia|"
    r"federazione|associazione|onlus|asd|fondazione|polisportiva|aero\s+club|lions|arcigay|"
    r"pubblica\s+assistenza|federconsumatori|confesercenti|confagricoltura|confcommercio|confindustria|"
    r"banca\s+del\s+tempo|camera\s+penale|scuderia\s+ferrari\s+club|"
    r"servizi[oi]\s+(?:di\s+)?(?:(?:bike|car|scooter)[ -]?)?sharing|"
    r"sindac[oa]|presidente|avvocat[oa]|difensor[ea]|(?:uno\s+dei\s+)?(?:due\s+)?imputat[oaie])\b",
    re.I,
)
_NON_PERSON = re.compile(r"^(?:fiducia\s+di\b|(?:egli|essi|esse)$)", re.I)
_AFTER_ROLE = re.compile(r"\s+e\s+poi\s+liquidator[ea]\s+(?:dell['’]|della\s+|del\s+|di\s+)\s*", re.I)
_CO_HOLDER = re.compile(
    r"\s+con\s+(?:il\s+fratello|la\s+sorella)\s+[^,;:!?\n]{1,60}?\s+(?:dell['’]|della\s+|del\s+|di\s+)\s*", re.I,
)
_APPOSITION = re.compile(r'\s*["»”]?\s*[,\(–-]\s*(?:(?:avv\.|avvocat[oa]|dott\.|sig\.)\s*)?', re.I)
_APPOSITION_END = re.compile(r"\s*[,\)–—-]")
_ROLE_OWNER = re.compile(rf"\s*{_ARTICLE}(?:ex\s+)?", re.I)
_DEFENCE_TITLE = re.compile(r"\s+(?:(?:Avv\.|avvocat[oa])\s+)?", re.I)
_ASSISTED_BY = re.compile(r"\bassistit[oa]\s+(?:dall['’]|dal\s+|dalla\s+)$", re.I)
_COUNSEL_FOR = re.compile(r"\s*,\s*a\s+mezzo\s+del(?:lo)?\s+(?:proprio\s+)?\Z", re.I)
_COUNSEL_CLIENT_LINK = re.compile(
    r"\s+di\s+(?:fi|ﬁ)ducia\s+(?P<link>dell['’]|(?:di|del|della|dello|dei|degli|delle)\b)\s*", re.I,
)


def is_subject_candidate(text: str, label: str) -> bool:
    """Screen obvious category errors without changing the mention or its offsets."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    if label == "PERSON":
        return not _NON_PERSON.match(normalized)
    return not _NON_COMPANY.match(normalized) or bool(_COMPANY_SUFFIX.search(normalized))


def predicate_span(match: re.Match) -> tuple[int, int]:
    if match.group("assistance"):
        return match.span("assistance")
    return match.span("role" if match.group("role") else "employment")


def _target_position(text: str, match: re.Match) -> tuple[int, str] | None:
    if match.group("legal") or match.group("assistance"):
        # Keep the cue unchanged when the legal role includes a qualifier before its client.
        qualified = _COUNSEL_CLIENT_LINK.match(text, predicate_span(match)[1])
        if qualified:
            return qualified.end(), qualified["link"]
    if match.group("assistance"):
        return None
    after = _AFTER_ROLE.match(text, match.end()) or _CO_HOLDER.match(text, match.end())
    return (after.end() if after else match.end()), match.group("link") or ""


def _target_labels(match: re.Match) -> set:
    return {"PERSON", "COMPANY"} if match.group("legal") or match.group("assistance") else {"COMPANY"}


def _target_link(text: str, label: str) -> bool:
    return bool(_TARGET_LINK.fullmatch(text)
                or label == "COMPANY" and _COMPANY_TARGET_LINK.fullmatch(text))


def relation_is_supported(text: str, match: re.Match, source: dict, target: dict) -> bool:
    """Check that the proposed ends attach to this predicate in the right direction."""
    if source["label"] != "PERSON" or target["label"] not in _target_labels(match):
        return False
    if not all(is_subject_candidate(endpoint["text"], endpoint["label"]) for endpoint in (source, target)):
        return False
    left, right = predicate_span(match)
    article = _ROLE_ARTICLE.search(text, max(0, left - 6), left)
    prefix_end = article.start() if article else left
    if _NON_ASSERTED_ROLE.search(text[max(0, prefix_end - 32):prefix_end]):
        return False
    target_position = _target_position(text, match)
    if target_position is None:
        if target["end"] <= left < right <= source["start"]:
            return bool(_COUNSEL_FOR.fullmatch(text[target["end"]:left])
                        and _DEFENCE_TITLE.fullmatch(text[right:source["start"]]))
        return bool(right <= source["start"] < source["end"] <= target["start"]
                    and _ASSISTED_BY.search(text[max(0, left - 35):left])
                    and re.fullmatch(r"\s+di\s+fiducia\s+", text[right:source["start"]], re.I)
                    and _APPOSITION.fullmatch(text[source["end"]:target["start"]]))
    target_start, _ = target_position
    if target["start"] < target_start:
        return False
    gap = text[target_start:target["start"]]
    legal = match.group("legal") or match.group("assistance")
    coordinated = legal and (_LEGAL_TARGET_LIST.fullmatch(gap) or (
        target["label"] == "PERSON" and _INITIALS.fullmatch(target["text"]) and _INITIAL_LIST_PREFIX.fullmatch(gap)
    ))
    if not (_target_link(gap, target["label"]) or coordinated):
        return False
    if source["end"] <= left and _SUBJECT_LINK.fullmatch(text[source["end"]:left]):
        return True
    if source["start"] <= left < right == source["end"] and _ROLE_OWNER.fullmatch(text[source["start"]:left]):
        return not _NON_ASSERTED_ROLE.search(text[max(0, source["start"] - 32):source["start"]])
    if not match.group("employment") and target["end"] <= source["start"]:
        return bool(_APPOSITION.fullmatch(text[target["end"]:source["start"]]))
    return bool(match.group("role") and match.group("role").casefold() == "titolare"
                and right <= source["start"] < source["end"] <= target_start
                and re.fullmatch(r"\s+con\s+(?:il\s+fratello|la\s+sorella)\s+", text[right:source["start"]], re.I)
                and re.fullmatch(r"\s+(?:dell['’]|della\s+|del\s+|di\s+)\s*", text[source["end"]:target_start], re.I))


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
        if mention["label"] in labels and 0 <= mention["start"] - start <= 24
        and is_subject_candidate(mention["text"], mention["label"])
        and _target_link(text[start:mention["start"]], mention["label"])
    ]
    target = min(candidates, key=lambda mention: (
        mention["start"], -mention["end"], mention.get("mention_kind") != "description",
    ), default=None)
    if target is None and "PERSON" in labels and (initials := _INITIALS.match(text, start)):
        return {**_description(initials, "PERSON"), "mention_kind": "name"}
    return target


def _targets(text: str, mentions: list, target_position: tuple[int, str], labels: set):
    target_start, link = target_position
    target = _target(text, mentions, target_start, labels)
    if link.casefold() in {"del", "della", "dello", "dell'", "dell’"}:
        # In "dell'azienda" the article is part of the preposition.
        for label in sorted(labels):
            descriptor = _BARE_DESCRIPTION[label].match(text, target_start)
            if descriptor and (target is None or (target["start"], target["end"]) == descriptor.span()):
                target = _description(descriptor, label)
                break
    while target is not None:
        yield target
        conjunction = _COORDINATION.match(text, target["end"])
        if conjunction is None and target["label"] == "PERSON" and _INITIALS.fullmatch(target["text"]):
            conjunction = _INITIAL_LIST_COMMA.match(text, target["end"])
        if conjunction is None:
            break
        target = _target(text, mentions, conjunction.end(), labels)


def _counsel_endpoints(text: str, mentions: list, match: re.Match) -> tuple[dict, dict] | None:
    start, end = predicate_span(match)
    introduction = _COUNSEL_FOR.search(text, max(0, start - 64), start)
    title = _DEFENCE_TITLE.match(text, end)
    if introduction is None or introduction.end() != start or title is None:
        return None
    client = _subject(text, mentions, introduction.start())
    lawyer = _target(text, mentions, title.end(), {"PERSON"})
    if client is not None and lawyer is not None and relation_is_supported(text, match, lawyer, client):
        return lawyer, client
    return None


def _apposed_role_holder(text: str, named: list, target: dict) -> dict | None:
    introduction = _APPOSITION.match(text, target["end"])
    if introduction is None:
        return None
    person = _target(text, named, introduction.end(), {"PERSON"})
    if person is not None and not _INITIALS.fullmatch(person["text"]) and _APPOSITION_END.match(text, person["end"]):
        return person
    return None


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
        predicate_start, predicate_end = predicate_span(match)
        target_position = _target_position(text, match)
        if target_position is None:
            endpoints = _counsel_endpoints(text, mentions, match)
            if endpoints is None:
                continue
            source, client = endpoints
            targets = (client,)
        else:
            source = _subject(text, mentions, match.start())
            if source is None and not match.group("employment"):
                source = _role_subject(text, predicate_start, predicate_end)
            if source is None:
                continue
            targets = _targets(text, mentions, target_position, _target_labels(match))
        for target in targets:
            holder = source
            if source.get("mention_kind") == "description" and source["end"] == predicate_end:
                holder = _apposed_role_holder(text, named_mentions, target) or source
            relations.append({
                "source": holder, "target": target,
                "predicate_start": predicate_start, "predicate_end": predicate_end,
            })
    return relations
