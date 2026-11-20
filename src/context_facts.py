"""Extract document-local facts and validate model proposals against original spans."""

import re

from src.explicit_relations import is_subject_candidate
from src.llm_filter import LABELS


SUBJECTS = {"PERSON", "COMPANY"}
NAME = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[’'-][A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+)?"
PERSON_DESCRIPTION = re.compile(
    r"\b(?:il|la|un|una|l['’])\s*(?:\d{2}enne(?:\s+"
    r"(?!(?:a|al|con|da|dal|dapprima|di|e|era|ha|in|per|residente|domiciliat[oa]|si|è)\b)[a-zà-öø-ÿ]+)?|"
    r"(?:uomo|donna|ragazzo|ragazza|operaio|operaia|testimone|imputat[oa]|indagat[oa])"
    r"(?:\s+di\s+\d{2}\s+anni)?|[a-zà-öø-ÿ]+\s+di\s+\d{2}\s+anni)\b", re.IGNORECASE,
)
ACCOUNT_DESCRIPTION = re.compile(r"\b(?:(?:il|un|sul|al)\s+)?conto\s+corrente\b", re.IGNORECASE)
ACCOUNT_ATTRIBUTION = r"(?:è\s+|era\s+|risulta\s+)?(?:co)?intestat[oa]\s+(?:ad|alla|all['’]|al|a)"
ACCOUNT_PARTY = re.compile(r"\b(?:parte\s+(?:venditrice|acquirente|locatrice|conduttrice)|"
                           r"convenut[oa]|ricorrente|resistente)\b", re.IGNORECASE)
PERSON_TITLE = r"(?:sig(?:\.ra)?|signor[ae]?|dott(?:\.ssa)?)\.?"
ACCOUNT_NAMED_HOLDER = re.compile(
    rf"\b(?i:{ACCOUNT_ATTRIBUTION})\s+(?:(?i:{PERSON_TITLE})\s+(?P<person>{NAME}(?:\s+{NAME}){{1,3}})|"
    rf"(?i:società|ditta|impresa)\s+(?P<company>{NAME}(?:\s+{NAME}){{0,3}}))"
)
ABBREVIATED_COMPANY = re.compile(r"\bs\s*\.?\s*(?:r\.?\s*l(?:\.?\s*s)?|p\.?\s*a|n\.?\s*c|a\.?\s*s)$", re.IGNORECASE)
ORGANIZATION = re.compile(
    rf"\b(?:Tribunale|Procura(?:\s+della\s+Repubblica)?|Corte\s+d['’]Appello|"
    rf"Corte\s+di\s+Appello|Comune|Questura|Carabinieri)\s+d(?:i\s+|ell['’]){NAME}(?:\s+{NAME})?\b"
)
AUTHORITY_DESCRIPTION = re.compile(
    r"\b(?:la\s+polizia(?:\s+(?:locale|stradale|giudiziaria))?|i\s+carabinieri|"
    r"la\s+guardia\s+di\s+finanza)\b", re.IGNORECASE,
)
TRANSFER_OBJECT = re.compile(
    r"\b(?:(?:il|la|lo|i|gli|le|un|una)\s+)?(?:document[oi]|fascicol[oi]|atti|contratt[oi]|"
    r"chiav[ei]|assegn[oi])\b", re.IGNORECASE,
)
AMOUNT = re.compile(r"(?<!\w)(?:€\s*\d[\d.,]*|\d[\d.,]*\s*(?:euro\b|€))", re.IGNORECASE)
PAYMENT_COMPLEMENT = re.compile(
    r"\s+(?:(?:già|anche|direttamente)\s+)*(?:€|\d[\d.,]*\s*(?:euro\b|€)|"
    r"(?:(?:il|lo|la|l['’]|un|una|i|le|gli)\s*)?(?:euro|denaro|soldi|somma|somme|importo|canone|"
    r"rata|rate|bonific[oi]|assegn[oi]|stipendio|acconto|rimbors[oi]|sald[oi]|fondi)\b|"
    r"(?:con|tramite|mediante)\s+(?:(?:un|due|tre|\d+)\s+)?bonific[oi]\b)", re.IGNORECASE,
)
PAYABLE_OBJECT = re.compile(r"\s+(?:(?:il|lo|la|un|una|i|le|gli)\s+)?(?:pedaggio|debit[oi]|fattur[ae])\b",
                            re.IGNORECASE)
CLAUSE_BOUNDARY = re.compile(r"[;!?]|\n[^\S\n]*\n|(?<!\b[A-Za-z])(?<!\d)"
                             r"(?<!\b(?i:sig|avv))(?<!\b(?i:dott))(?<!\b(?i:sig\.ra))(?<!\b(?i:dott\.ssa))"
                             r"\.(?=\s+[A-Z]|\s*$)")
REPORTED = re.compile(r"\b(?:ha\s+(?:dichiarato|riferito|detto|affermato)|"
                      r"(?:dichiara|riferisce)\s+che|secondo\s+[^,;]{2,50}(?=,))", re.IGNORECASE)
HYPOTHETICAL = re.compile(r"\b(?:(?<!\banche )se|forse|ipotizz\w+|presunt\w+|avrebbe(?:ro)?|sarebbe(?:ro)?)\b",
                          re.IGNORECASE)
NEGATED = re.compile(r"\b(?:non|mai|senza)\b", re.IGNORECASE)

# The order encodes which attachment is being checked, including passive wording.
DIRECT = (
    ("family", r"(?:(?:è|era)\s+)?(?:(?:il|la)\s+)?(?:figli[oa]|padre|madre|fratello|sorella|moglie|marito|coniuge)\s+di",
     SUBJECTS, {"PERSON"}, False),
    ("residence", r"(?:è\s+|era\s+)?(?:residente|domiciliat[oa]|abita|abitava)\s+(?:a|in|nel|nella)",
     {"PERSON"}, {"LOCATION"}, False),
    ("registered_office", r"(?:(?:ha|aveva)\s+(?:la\s+)?sede(?:\s+legale)?|con\s+sede(?:\s+legale)?)\s+(?:a|in|presso)",
     {"COMPANY", "ORGANIZATION"}, {"LOCATION"}, False),
    ("account_holder", ACCOUNT_ATTRIBUTION, {"BANK_ACCOUNT"}, SUBJECTS | {"PARTY"}, True),
    ("account_holder", r"(?:è\s+|era\s+)?(?:co)?intestatari[oa]\s+(?:di|del|dell['’])(?:\s+un)?(?:\s+conto(?:\s+corrente)?)?(?:\s+con)?(?:\s+IBAN)?\s*[:=]?",
     SUBJECTS, {"BANK_ACCOUNT"}, False),
    ("contact", r"(?:(?:ha|aveva|usa|utilizza)\s+)?(?:l['’])?(?:indirizzo\s+)?(?:e-?mail|recapito|telefono|cellulare|PEC)\s*[:=]?",
     SUBJECTS, {"EMAIL", "PHONE"}, False),
    ("contact", r"(?:è\s+|era\s+)?(?:intestat[oa]\s+(?:a|alla|al)|in\s+uso\s+a|appartiene\s+a)",
     {"EMAIL", "PHONE"}, SUBJECTS, True),
    ("identifier", r"(?:con\s+)?(?:codice\s+fiscale|partita\s+IVA|P\.\s*IVA)\s*[:=]?",
     SUBJECTS, {"TAX_CODE", "VAT_NUMBER"}, False),
)
def _direct_patterns(entries):
    return tuple((kind, re.compile(rf"(?<!\w)(?:{pattern})(?:(?<=['’])|(?!\w))", re.IGNORECASE), left, right, reverse)
                 for kind, pattern, left, right, reverse in entries)


DIRECT = _direct_patterns(DIRECT)
REVIEW_DIRECT = _direct_patterns((
    ("family", r"(?:è|era)\s+sposat[oa]\s+con", {"PERSON"}, {"PERSON"}, False),
    ("residence", r"(?:vive|viveva)\s+(?:a|in|nel|nella)", {"PERSON"}, {"LOCATION"}, False),
    ("registered_office", r"(?:ha|aveva)\s+la\s+sede\s+legale\s+ubicata\s+(?:a|in)",
     {"COMPANY", "ORGANIZATION"}, {"LOCATION"}, False),
    ("account_holder", r"appart(?:iene|eneva)\s+(?:ad|alla|all['’]|al|a)",
     {"BANK_ACCOUNT"}, SUBJECTS | {"PARTY"}, True),
    ("contact", r"(?:è\s+|era\s+)?raggiungibile\s+(?:al\s+numero|all['’]indirizzo)",
     SUBJECTS, {"PHONE", "EMAIL"}, False),
    ("identifier", r"(?:è\s+|era\s+)?titolare\s+(?:del\s+codice\s+fiscale|della\s+partita\s+IVA)\s*[:=]?",
     SUBJECTS, {"TAX_CODE", "VAT_NUMBER"}, False),
))

EVENTS = {
    "meeting": r"ha(?:nno)?\s+incontrat[oa]|incontrò|incontrava|si\s+(?:è|sono)\s+incontrat[oaie]\s+con",
    "communication": r"ha(?:nno)?\s+(?:parlato\s+con|telefonato\s+a|contattato|chiamato)|telefonò\s+a",
    "procedure": r"(?:(?:è|fu|era)\s+stat[oa]\s+|(?:è|fu|era)\s+)(?:(?:infatti|poi|quindi)\s+)?(?:arrestat[oa]|denunciat[oa]|"
                 r"interrogat[oa]|condannat[oa]|assolt[oa])|ha(?:nno)?\s+(?:arrestato|condannato|assolto|interrogato)",
    "payment": r"ha(?:nno)?\s+(?:pagato|versato|accreditato|trasferito|ricevuto)|versò|ha\s+effettuato\s+un\s+bonifico",
    "delivery": r"ha(?:nno)?\s+(?:consegnato|ricevuto|inviato)|consegnò|inviò",
    "agreement": r"ha(?:nno)?\s+(?:concluso|stipulato|firmato)\s+(?:(?:un|il)\s+)?(?:contratto|accordo)\s+con",
    "birth": r"(?:(?:è|era)\s+)?nat[oa]\s+(?:a|in)",
    "statement": r"ha(?:nno)?\s+(?:dichiarato|riferito|testimoniato|affermato|negato)|dichiara|riferisce",
}
EVENTS = {kind: re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for kind, pattern in EVENTS.items()}
REVIEW_EVENTS = {
    "meeting": r"aveva(?:no)?\s+incontrat[oa]|si\s+(?:è|sono)\s+riunit[oaie]\s+con",
    "communication": r"ha(?:nno)?\s+(?:scritt[oa]\s+a|comunicato\s+con)|aveva(?:no)?\s+parlato\s+con",
    "procedure": r"ha(?:nno)?\s+(?:denunciato|identificato|catturato|rintracciato)",
    "payment": r"ha(?:nno)?\s+(?:riscosso|incassato|rimborsato)",
    "delivery": r"ha(?:nno)?\s+(?:trasmesso|recapitato)",
    "agreement": r"ha(?:nno)?\s+sottoscritto\s+(?:(?:un|il)\s+)?(?:contratto|accordo)\s+con",
    "birth": r"(?:è|era)\s+venut[oa]\s+al\s+mondo\s+(?:a|in)",
    "statement": r"ha(?:nno)?\s+(?:precisato|raccontato|riferit[oa])\s+che",
}
REVIEW_EVENTS = {kind: re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for kind, pattern in REVIEW_EVENTS.items()}
AGREEMENT_PARTIES = re.compile(r"\b(?:accordo|contratto(?:\s+di\s+(?:subappalto|locazione|cessione))?)"
                               r"(?:\s+(?:intercorso|concluso|stipulato|siglato))?\s+tra\b", re.IGNORECASE)
SIGNED_AGREEMENT = re.compile(
    rf"(?P<predicate>(?i:\bscrittura\s+privata\s+tra))\s+(?i:(?:il|la)\s+{PERSON_TITLE})\s+"
    rf"(?P<person>{NAME}(?:\s+{NAME}){{1,3}})\s*,[^;!?]{{0,240}}?\b(?i:e\s+la\s+(?:società|ditta|impresa))\s+"
    rf"(?P<organization>{NAME}(?:\s+{NAME}){{0,4}})\s*,[^;!?]{{0,220}}?\b(?i:si\s+conviene)\b"
)
OPEN_VERB = re.compile(r"\b(?:ha|hanno|aveva|avevano)\s+[a-zà-öø-ÿ]+(?:ato|uto|ito)\b", re.IGNORECASE)
GRAPH_LABELS = LABELS | {"ORGANIZATION", "OBJECT", "AMOUNT", "PARTY"}


def mention_key(entity: dict) -> tuple:
    return entity["start"], entity["end"], entity["label"]


def context_mentions(document: dict) -> list:
    """Keep NER spans and add only explicit descriptions used by graph rules."""
    text = document["text"]
    result = {}
    for e in document["entities"]:
        if e["label"] not in GRAPH_LABELS:
            continue
        start, end = e["start"], e["end"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not e["text"].strip() or text[start:end] != e["text"]):
            raise ValueError(f"Entity span does not match document {document['id']}.")
        if e["label"] in SUBJECTS and not is_subject_candidate(e["text"], e["label"]):
            continue
        if e["label"] == "COMPANY" and text[end:end + 1] == "." and ABBREVIATED_COMPANY.search(e["text"]):
            e = {**e, "text": text[start:end + 1], "end": end + 1}
        result[mention_key(e)] = e
    for regex, label, kind in ((PERSON_DESCRIPTION, "PERSON", "description"),
                               (ACCOUNT_DESCRIPTION, "BANK_ACCOUNT", "description"),
                               (ACCOUNT_PARTY, "PARTY", "description"),
                               (ORGANIZATION, "ORGANIZATION", "name"),
                               (AUTHORITY_DESCRIPTION, "ORGANIZATION", "description"),
                               (TRANSFER_OBJECT, "OBJECT", "description"),
                               (AMOUNT, "AMOUNT", "description")):
        for match in regex.finditer(text):
            entity = {"start": match.start(), "end": match.end(), "text": match.group(), "label": label,
                      "mention_kind": kind, "source": "rule:context", "needs_review": False}
            result.setdefault(mention_key(entity), entity)
    for match in ACCOUNT_NAMED_HOLDER.finditer(text):
        name = match.lastgroup
        start, end = match.span(name)
        label = name.upper()
        if any(e["label"] == label and e["start"] == start and e["end"] > end for e in result.values()):
            continue
        entity = {"start": start, "end": end, "text": text[start:end], "label": label,
                  "mention_kind": "name", "source": "rule:account_context", "needs_review": False}
        result[mention_key(entity)] = entity
    return sorted(result.values(), key=mention_key)


def _bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(0, start - 180)
    for match in CLAUSE_BOUNDARY.finditer(text, left, start):
        left = match.end()
    right = CLAUSE_BOUNDARY.search(text, end, min(len(text), end + 240))
    return left, right.start() if right else min(len(text), end + 240)


def _mention(text: str, entities: list, start: int, end: int, labels: set, *, before: bool,
             connector: str) -> dict | None:
    """Find an adjacent endpoint only if its intervening wording fits the relation."""
    candidates = []
    for e in entities:
        if e["label"] not in labels:
            continue
        if not start <= e["start"] < e["end"] <= end:
            continue
        gap = text[e["end"]:end] if before else text[start:e["start"]]
        if re.fullmatch(connector, gap, re.IGNORECASE):
            candidates.append(e)
    if not candidates:
        return None
    # A full span takes precedence over a nested abbreviation at the same boundary.
    return min(candidates, key=lambda e: (e.get("source") != "rule:account_context", e.get("needs_review", False),
                                         -(e["end"] - e["start"]), e["start"]))


_LEFT = r"[\s,]*(?:non\s+)?(?:(?:è|era|fu)\s+)?"
_RIGHT = r"[\s:,-]*(?:(?:il|la|un|una)\s+|l['’])?"
_EVENT_LEFT = rf"{_LEFT}(?:(?:dapprima|poi|infine|successivamente|quindi|già)\s+)?"
_REVIEW_LEFT = rf"(?:{_LEFT}|\s*,\s*[^,;.!?\n]{{1,48}},\s*(?:non\s+)?)"
_REVIEW_EVENT_LEFT = rf"(?:{_EVENT_LEFT}|{_REVIEW_LEFT})"
_REVIEW_RIGHT = rf"(?:(?:[\s,]+(?:poi|anche|ieri|successivamente|direttamente)\b)?{_RIGHT})"


def _assertion(text: str, lo: int, start: int, hi: int) -> str:
    lead = text[lo:start]
    if NEGATED.search(lead[-36:]):
        return "negated"
    if text[hi:hi + 1] == "?" or HYPOTHETICAL.search(lead):
        return "hypothetical"
    return "reported" if REPORTED.search(lead) else "asserted"


def _fact(kind: str, span: tuple[int, int], bounds: tuple[int, int], arguments: dict, text: str,
          entities: list) -> dict:
    start, end = span
    lo, hi = bounds
    assertion = _assertion(text, lo, start, hi)
    marker = REPORTED.search(text, lo, start)
    if marker and "speaker" not in arguments:
        if marker.group().lower().startswith("secondo "):
            speaker = _mention(text, entities, marker.start() + len("secondo "), marker.end(),
                               SUBJECTS | {"ORGANIZATION"}, before=False, connector=_RIGHT)
        else:
            speaker = _mention(text, entities, lo, marker.start(), SUBJECTS | {"ORGANIZATION"},
                               before=True, connector=_LEFT)
        if speaker:
            arguments["speaker"] = speaker
    return {"kind": kind, "predicate_start": start, "predicate_end": end,
            "start": lo, "end": hi, "assertion": assertion,
            "arguments": arguments, "method": "rules"}


def fact_key(fact: dict) -> tuple:
    return (fact["kind"], fact["predicate_start"], fact["predicate_end"],
            tuple(sorted((role, *mention_key(e)) for role, e in fact["arguments"].items())))


def _event_arguments(text: str, entities: list, start: int, end: int, lo: int, hi: int,
                     kind: str, *, review: bool = False) -> dict:
    left_connector = _REVIEW_EVENT_LEFT if review else _EVENT_LEFT
    right_connector = _REVIEW_RIGHT if review else _RIGHT
    labels = {"PERSON"} if kind == "birth" else SUBJECTS | {"ORGANIZATION"}
    actor = _mention(text, entities, lo, start, labels, before=True, connector=left_connector)
    if not actor or actor["text"].rstrip().endswith(("-", "–", "—")):
        return {}
    arguments = {"subject": actor}
    if kind == "statement":
        arguments["speaker"] = arguments.pop("subject")
        return arguments
    participant_labels = SUBJECTS | {"ORGANIZATION", "BANK_ACCOUNT"}
    following = end
    if kind == "delivery":
        item = _mention(text, entities, following, hi, {"OBJECT"}, before=False, connector=right_connector)
        if item:
            arguments["item"] = item
            following = item["end"]
    if kind == "payment":
        verb = text[start:end].casefold().split()[-1]
        payable = verb in {"pagato", "rimborsato"} and PAYABLE_OBJECT.match(text, following, hi)
        if not (verb == "bonifico" or PAYMENT_COMPLEMENT.match(text, following, hi) or payable):
            return {}
        amount = _mention(text, entities, following, hi, {"AMOUNT"}, before=False, connector=right_connector)
        if amount:
            arguments["amount"] = amount
            following = amount["end"]
    other = _mention(text, entities, following, hi, participant_labels,
                     before=False, connector=rf"{right_connector}(?:(?:a|ad|da|con|presso|sul|dal|al)\s+)?")
    if other:
        arguments["object"] = other
    if kind in {"meeting", "communication", "delivery", "agreement", "action"} and not other:
        return {}
    following = other["end"] if other else following
    if other:
        index = 2
        while extra := _mention(text, entities, following, hi, participant_labels, before=False,
                                 connector=r"\s*(?:,|e\s+|ed\s+)\s*"):
            arguments[f"object_{index}"] = extra
            following = extra["end"]
            index += 1
    place_connector = r"\s+" if kind == "birth" else r"[\s,]*(?:a|ad|in|presso|nel|nella)\s+"
    relative_residence = other and re.search(r"\b(?:residente|domiciliat[oa])$", other["text"], re.IGNORECASE)
    where = None if relative_residence else _mention(text, entities, following, hi, {"LOCATION"},
                                                    before=False, connector=place_connector)
    if where:
        arguments["place"] = where
        following = where["end"]
    when = _mention(text, entities, following, hi, {"TIME"}, before=False,
                    connector=r"[\s,]*(?:(?:il|nel|alle|all['’]|in\s+data)\s+)?")
    if when:
        arguments["time"] = when
    if kind == "birth" and not where:
        return {}
    return arguments


def _agreement_arguments(text: str, entities: list, end: int, hi: int) -> dict:
    labels = SUBJECTS | {"ORGANIZATION", "PARTY"}
    participant_prefix = rf"{_RIGHT}(?:(?:società|ditta|impresa|{PERSON_TITLE})\s+)?"
    first = _mention(text, entities, end, hi, labels, before=False, connector=participant_prefix)
    if first:
        second = _mention(text, entities, first["end"], hi, labels, before=False,
                          connector=rf"\s+(?:e|ed){participant_prefix}")
        if second:
            return {"subject": first, "object": second}
    return {}


def _signed_agreements(text: str, entities: list):
    for match in SIGNED_AGREEMENT.finditer(text):
        arguments = {}
        for role, label in (("subject", "PERSON"), ("object", "ORGANIZATION")):
            start, end = match.span(label.lower())
            arguments[role] = {"start": start, "end": end, "text": text[start:end], "label": label,
                               "mention_kind": "name", "source": "rule:context", "needs_review": False}
        yield _fact("agreement", match.span("predicate"), match.span(), arguments, text, entities)


def extract_facts(document: dict, entities: list, *, review: bool = False) -> list:
    """Extract supported directed attributes and events over the whole document."""
    text = document["text"]
    facts = {}
    direct = (*DIRECT, *REVIEW_DIRECT) if review else DIRECT
    events = (*EVENTS.items(), *REVIEW_EVENTS.items(), ("agreement", AGREEMENT_PARTIES)) if review else EVENTS.items()
    left_connector = _REVIEW_LEFT if review else _LEFT
    right_connector = _REVIEW_RIGHT if review else _RIGHT
    for kind, regex, before, after, reverse in direct:
        for predicate in regex.finditer(text):
            start, end = predicate.span()
            lo, hi = _bounds(text, start, end)
            left = _mention(text, entities, lo, start, before, before=True, connector=left_connector)
            connector = right_connector
            if kind == "account_holder" and reverse:
                connector = rf"{right_connector}(?:(?:società|ditta|impresa|{PERSON_TITLE})\s+)?"
            right = _mention(text, entities, end, hi, after, before=False, connector=connector)
            if left and right:
                source, target = (right, left) if reverse else (left, right)
                fact = _fact(kind, (start, end), (lo, hi), {"subject": source, "object": target}, text, entities)
                facts[fact_key(fact)] = fact
    for kind, pattern in events:
        for predicate in pattern.finditer(text):
            start, end = predicate.span()
            lo, hi = _bounds(text, start, end)
            arguments = (_agreement_arguments(text, entities, end, hi) if pattern is AGREEMENT_PARTIES else
                         _event_arguments(text, entities, start, end, lo, hi, kind, review=review))
            if review and pattern is REVIEW_EVENTS.get(kind) and kind == "procedure" and "object" not in arguments:
                continue
            if arguments:
                fact = _fact(kind, (start, end), (lo, hi), arguments, text, entities)
                facts[fact_key(fact)] = fact
    if review:
        for fact in _signed_agreements(text, entities):
            facts[fact_key(fact)] = fact
    return list(facts.values())


def context_cues(text: str) -> int:
    """Count passages with possible facts, including actions outside the rule families."""
    return sum(bool(p.search(text)) for _, p, *_ in (*DIRECT, *REVIEW_DIRECT)) + sum(
        bool(p.search(text)) for p in (*EVENTS.values(), *REVIEW_EVENTS.values(),
                                      AGREEMENT_PARTIES, SIGNED_AGREEMENT, OPEN_VERB)
    )


def _passage_entities(passage: dict) -> list:
    base = passage["start"]
    return [{**e, "start": e["start"] - base, "end": e["end"] - base}
            for e in passage["context_mentions"]]


def _reviewed_fact(fact: dict, base: int) -> dict:
    return {**fact, "method": "qwen", "start": fact["start"] + base, "end": fact["end"] + base,
            "predicate_start": fact["predicate_start"] + base, "predicate_end": fact["predicate_end"] + base,
            "arguments": {role: {**e, "start": e["start"] + base, "end": e["end"] + base}
                          for role, e in fact["arguments"].items()}}


def reviewable_actions(passage: dict) -> list:
    """Offer open predicates with grounded actors and objects for model confirmation."""
    text, base = passage["text"], passage["start"]
    core_start, core_end = passage["core_start"] - base, passage["core_end"] - base
    entities = _passage_entities(passage)
    actions = []
    for match in OPEN_VERB.finditer(text, core_start):
        start, end = match.span()
        if start >= core_end:
            break
        if any(pattern.match(text, start) for pattern in EVENTS.values()):
            continue
        lo, hi = _bounds(text, start, end)
        args = _event_arguments(text, entities, start, end, lo, hi, "action")
        if args:
            actions.append(_reviewed_fact(_fact("action", (start, end), (lo, hi), args, text, entities), base))
    return actions


MAX_FACT_CANDIDATES = 4


def reviewable_facts(passage: dict) -> list:
    """Prioritize additional attachments and retain variety within each passage."""
    base = passage["start"]
    text = {"text": passage["text"]}
    entities = _passage_entities(passage)
    rules = {fact_key(fact): fact for fact in extract_facts(text, entities)}
    expanded = {fact_key(fact): fact for fact in extract_facts(text, entities, review=True)}
    facts = {}
    for key, fact in (rules | expanded).items():
        if passage["core_start"] <= fact["predicate_start"] + base < passage["core_end"]:
            facts[key] = {**_reviewed_fact(fact, base), "additional_candidate": key not in rules}
    typed_positions = {fact["predicate_start"] for fact in facts.values()}
    candidates = [*facts.values(), *({**fact, "additional_candidate": True} for fact in reviewable_actions(passage)
                                     if fact["predicate_start"] not in typed_positions)]
    candidates.sort(key=lambda fact: (not fact["additional_candidate"], fact["predicate_start"], fact["kind"]))
    ordered, remaining, seen = [], [], set()
    for fact in candidates:
        if fact["kind"] in seen:
            remaining.append(fact)
        else:
            seen.add(fact["kind"])
            ordered.append(fact)
    return (ordered + remaining)[:MAX_FACT_CANDIDATES]


def parse_fact(item: dict, passage: dict, locate) -> dict:
    """Use Qwen quotations to locate a clause and verify its attachments with the rules."""
    if not isinstance(item, dict) or set(item) != {"evidence", "predicate", "arguments"}:
        raise ValueError("A fact requires an evidence quotation, predicate and arguments.")
    supplied = item["arguments"]
    if not isinstance(supplied, dict) or not supplied or not all(isinstance(v, str) for v in supplied.values()):
        raise ValueError("Fact arguments must map roles to source quotations.")
    text = passage["text"]
    quote_start, quote_end = locate(item["evidence"], text)
    quote = text[quote_start:quote_end]
    start, end = locate(item["predicate"], quote)
    start += quote_start
    end += quote_start
    base = passage["start"]
    if not passage["core_start"] <= start + base < passage["core_end"]:
        raise ValueError("The fact predicate must start inside the passage core.")
    lo, hi = _bounds(text, start, end)
    if quote_start < lo or (quote_end > hi and not re.fullmatch(r"[.!?]\s*", text[hi:quote_end])):
        raise ValueError("The fact evidence must be in one clause.")
    entities = _passage_entities(passage)
    relative = {"text": text}
    found = [f for f in extract_facts(relative, entities)
             if (f["predicate_start"], f["predicate_end"]) == (start, end)]
    if not found and OPEN_VERB.fullmatch(text[start:end]):
        arguments = _event_arguments(text, entities, start, end, lo, hi, "action")
        if arguments:
            found.append(_fact("action", (start, end), (lo, hi), arguments, text, entities))
    coordinates = {role: tuple(quote_start + n for n in locate(value, quote))
                   for role, value in supplied.items()}
    for fact in found:
        args = fact["arguments"]
        participants = {role for role in args if role in {"subject", "object"}} or {"speaker"}
        if not participants <= coordinates.keys() or not all(
            role in args and (args[role]["start"], args[role]["end"]) == offsets
            for role, offsets in coordinates.items()
        ):
            continue
        # Keep a rule-grounded argument even if the model omits an optional attribute.
        return _reviewed_fact(fact, base)
    raise ValueError("The fact arguments or direction are unsupported by the selected wording.")
