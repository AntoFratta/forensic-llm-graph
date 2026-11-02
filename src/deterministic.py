"""First-pass entity recognition for Italian forensic documents.

Rules identify structured mentions, while a spaCy model supplies named entities.
The module preserves original text and offsets. It returns candidates for later
review rather than final LLM decisions.
"""

import re


def _pattern(expression: str) -> re.Pattern:
    """Compile a case-insensitive rule once for reuse across documents."""
    return re.compile(expression, re.IGNORECASE)


# Structured identifiers are matched as written; no checksum validation is applied.
_EMAIL = re.compile(
    r"(?<![a-z0-9._+%@-])[a-z0-9_+%-]+(?:\.[a-z0-9_+%-]+)*@"
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}(?![a-z0-9_@-])",
    re.IGNORECASE | re.ASCII,
)
_TAX_CODE = _pattern(
    r"\b[A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST]"
    r"[0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]\b"
)
_VAT = _pattern(
    r"\b(?:partita\s+i\.?\s*v\.?\s*a\.?|p\.?\s*i\.?\s*v\.?\s*a\.?)"
    r"\s*[:=]?\s*(?P<value>\d(?:[ .]?\d){7,12})(?!\d)"
)
_PHONE = _pattern(
    r"\b(?:telefono(?:\s+fisso)?|telefonico|tel\.?|cellulare|cell\.?|fax)"
    r"(?:\s*/\s*(?:cellulare|telefono))?\s*(?:n\.?|numero)?\s*[:=]?\s*"
    r"(?P<value>\+?\d(?:[\d /().-]*\d)?)"
)
# Italian IBANs may contain spaces; other countries are supported in compact form.
_ITALIAN_IBAN = re.compile(r"\bIT\s*\d\s*\d\s*[A-Z](?:[ \t]*[A-Z0-9]){22}\b", re.I | re.ASCII)
_COMPACT_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.I | re.ASCII)
_ACCOUNT = _pattern(
    r"\b(?:c\s*/\s*c|conto\s+corrente|conto\s+bancario)"
    r"(?:\s+(?:postale|bancario))?\s*(?:n(?:umero)?\.?)?\s*[:=]?\s*"
    r"(?P<value>\d(?:[ /-]?\d){3,19})(?!\d)"
)
# Each numeric date uses the same separator between all three components.
_NUMERIC_DATE = "(?:" + "|".join(
    rf"\d{{1,2}}\s*{separator}\s*\d{{1,2}}\s*{separator}\s*(?:\d{{4}}|\d{{2}})"
    for separator in ("/", "-", r"\.")
) + ")"
_MONTH = r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)"
_DATE = rf"(?:{_NUMERIC_DATE}|\d{{1,2}}(?:\s*°)?\s+{_MONTH}\s+\d{{4}})"
_DATE_RANGE = _pattern(rf"\b(?P<value>{_DATE}\s+(?:al|a|[-–])\s+{_DATE})(?!\d)")
_DATES = _pattern(rf"\b{_DATE}(?!\d)")
_HOUR = r"(?:[01]?\d|2[0-3])\s*[:.]\s*[0-5]\d"
_HOURS = _pattern(rf"\b(?:ore|orari(?:o)?)\s*[:=]?\s*(?P<value>{_HOUR}(?:\s*[-–]\s*{_HOUR})?)(?!\d)")
_YEARS = _pattern(r"(?<![\w/])(?:1\d{3}|20\d{2})\b(?!\s*/\s*\d)")
_SHORT_YEARS = _pattern(r"(?<!\w)['’]\d{2}\b")
_RELATIVE_TIME = _pattern(
    r"\b(?:ieri|oggi|domani|stasera|stamattina|stanotte|l'altroieri|"
    r"(?:settimana|mese|anno)\s+(?:scors[oa]|prossim[oa])|"
    r"(?:\d+|un[oa]?|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici)"
    r"\s+(?:giorn[oi]|mes[ei]|ann[oi]|or[ae]|settimane)\s+(?:fa|prima|dopo))\b"
)
# This initial role vocabulary is intentionally context-sensitive and is flagged.
_ROLES = _pattern(
    r"\b(?:amministrator[ei](?:\s+(?:unic[oi]|delegat[oi]))?|"
    r"legale\s+rappresentante|rappresentante\s+legale|"
    r"procurator[ei](?:\s+della\s+Repubblica)?(?:\s+aggiunt[oi])?|"
    r"pubblic[oi]\s+minister[oi]|giudice(?:\s+per\s+le\s+indagini\s+preliminari)?|"
    r"presidente|direttor[ei]|dirigente|responsabile(?:\s+del\s+servizio)?|"
    r"sindaco|titolare|imprenditor[ei]|consulente|magistrat[oi]|"
    r"collaboratore\s+di\s+giustizia|carabinier[ei]|teste)\b|\b(?:P\.\s*M\.|Avv\.)"
)
_STREET_NAME = r"(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÿ’'-]*|di|del|della|delle|degli|dei|dell')"
_ADDRESS = re.compile(
    rf"\b(?:[Vv]ia|[Cc]orso|[Pp]iazza|[Vv]iale|[Vv]icolo|[Ll]argo)"
    rf"\s+{_STREET_NAME}(?:\s+{_STREET_NAME}){{0,7}}"
    r"\s*,?\s*(?:n\.\s*)?\d+(?:\s*/\s*[A-Z])?"
    r"(?:\s*,?\s*C\.?A\.?P\.?\s*\d{5})?\b"
)
_COMPANY_SUFFIX = _pattern(
    r"\b(?:s\s*\.?\s*r\s*\.?\s*l\s*\.?\s*s?|"
    r"s\s*\.?\s*p\s*\.?\s*a|s\s*\.?\s*n\s*\.?\s*c)"
    r"\b\.?[\s\"'”»]*$"
)
_ADMINISTRATIVE_PLACE = _pattern(
    r"^(?:comune|provincia|regione|città(?:\s+metropolitana)?)"
    r"\s+(?:di\s+|del\s+|della\s+|dell['’]\s*)"
)
_GENERIC_PLACE = _pattern(r"^(?:comune|provincia|regione|città|località|paese)$")
_LEGAL_REFERENCE = _pattern(
    r"^(?:d\s*\.\s*lgs\.?|d\s*\.\s*p\s*\.\s*r\.?|"
    r"d\s*\.\s*l\.?|art(?:t)?\.?|legge|decreto)$"
)
# Form vocabulary is a review signal, not a list of names to discard.
_FORM_TERM = _pattern(
    r"^(?:chiede|richiede|dichiara|autorizza|comunica|attesta|certifica|"
    r"richiesta|dichiarazione|oggetto|firma|giorn[oi]|mes[ei]|ann[oi]|"
    r"data|nat[oa]|residente|domicilio|indirizzo)$"
)
_NAME_WORD = r"[A-ZÀ-ÖØ-Þ][\wÀ-ÿ’'-]+"
_FORM_NAME_WORD = rf"(?!(?i:nome|cognome)\b){_NAME_WORD}"
_FORM_NAME_END = (
    r"(?=\s+(?i:M\s+F\b|F\s+M\b|nat[oa](?:\s*/\s*[oa])?\b|"
    r"codice\s+fiscale\b|data\s+di\s+nascita\b|sesso\b)|\s*[,;]|\s*$)"
)
_FORM_PERSON = re.compile(
    r"\b(?i:sottoscritt[oa](?:\s*/\s*[oa])?)\s*:?\s*"
    rf"(?P<value>{_FORM_NAME_WORD}(?:\s+{_FORM_NAME_WORD}){{1,5}}?)"
    r"(?=\s+(?i:M\s+F\b|F\s+M\b|nat[oa](?:\s*/\s*[oa])?\b|codice\s+fiscale\b))"
)
# Paired personal fields provide context even without a preceding 'sottoscritto'.
_FORM_FIELDS = [
    re.compile(
        rf"\b(?i:{first})\s*[:=]?\s*"
        rf"(?P<first>{_FORM_NAME_WORD}(?:\s+{_FORM_NAME_WORD}){{0,5}}?)"
        rf"\s+(?i:{second})\s*[:=]?\s*"
        rf"(?P<second>{_FORM_NAME_WORD}(?:\s+{_FORM_NAME_WORD}){{0,5}}?)"
        + _FORM_NAME_END
    )
    for first, second in (("cognome", "nome"), ("nome", "cognome"))
]
_PERSON_FIELD_LABEL = _pattern(r"\b(?:nome|cognome)\b")
_RESIDENCE_PREFIX = _pattern(
    r"^(?:residente|domiciliat[oa])\s+(?:in|a)\s+"
)
_SEX_FIELDS = re.compile(r"\s+(?:M\s+F|F\s+M)$")


def load_nlp(model_name: str):
    """Load and return an installed Italian spaCy pipeline with NER enabled.

    The calling script reuses this pipeline across documents. Installation is
    handled separately, including on Colab; a missing model stops execution.
    """
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("Install spaCy before enabling the NLP component.") from exc
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(f"spaCy model {model_name!r} is not installed.") from exc
    if "ner" not in nlp.pipe_names or nlp.lang != "it":
        raise ValueError("Select an Italian spaCy pipeline with a trained NER component.")
    return nlp


def _entity(text: str, start: int, end: int, label: str, source: str,
            reasons=()) -> dict:
    """Build a candidate using inclusive start and exclusive end offsets.

    format_valid is None when no digit-count check applies. subtype distinguishes
    IBANs from other account identifiers; neither field establishes existence.
    """
    return {
        "text": text[start:end], "label": label, "start": start, "end": end,
        "source": source, "needs_review": bool(reasons),
        "review_reasons": list(reasons), "format_valid": None, "subtype": None,
    }


def _add_matches(result: list, text: str, pattern: re.Pattern, label: str,
                 rule: str, reasons=()) -> None:
    """Append rule matches, selecting the value group when context is present."""
    for match in pattern.finditer(text):
        group = "value" if "value" in pattern.groupindex else 0
        start, end = match.span(group)
        result.append(_entity(text, start, end, label, f"rule:{rule}", reasons))


def _overlaps(first: dict, second: dict) -> bool:
    """Check whether two half-open character spans share any characters."""
    return first["start"] < second["end"] and second["start"] < first["end"]


def _flag(entity: dict, reason: str) -> None:
    """Mark a candidate for review without repeating the same reason."""
    entity["needs_review"] = True
    if reason not in entity["review_reasons"]:
        entity["review_reasons"].append(reason)


def _extract_rules(text: str) -> list:
    """Extract structured mentions, addresses and role candidates.

    Context words introduce phone, VAT and account numbers but are excluded from
    their spans. Unusual digit counts are flagged rather than discarded.
    """
    result = []
    for pattern, label, rule in [
        (_EMAIL, "EMAIL", "email"), (_TAX_CODE, "TAX_CODE", "tax_code_format"),
        (_ITALIAN_IBAN, "BANK_ACCOUNT", "italian_iban"),
        (_COMPACT_IBAN, "BANK_ACCOUNT", "compact_iban"),
        (_ACCOUNT, "BANK_ACCOUNT", "account_context"),
    ]:
        before = len(result)
        _add_matches(result, text, pattern, label, rule)
        for candidate in result[before:]:
            if label == "BANK_ACCOUNT":
                if "iban" in rule:
                    candidate["subtype"] = "iban"
                else:
                    prefix = text[max(0, candidate["start"]-50):candidate["start"]]
                    if re.search(r"postale\s*(?:n(?:umero)?\.?)?\s*[:=]?\s*$", prefix, re.I):
                        candidate["subtype"] = "postal_account"
                    else:
                        candidate["subtype"] = "account_number"
    for pattern, label, rule in [(_VAT, "VAT_NUMBER", "vat_context"), (_PHONE, "PHONE", "phone_context")]:
        before = len(result)
        _add_matches(result, text, pattern, label, rule)
        for candidate in result[before:]:
            digits = re.sub(r"\D", "", candidate["text"])
            if label == "VAT_NUMBER":
                valid = len(digits) == 11
            else:
                valid = 6 <= len(digits) <= 15
            candidate["format_valid"] = valid
            if not valid:
                _flag(candidate, "unexpected_digit_count")
            if label == "VAT_NUMBER":
                prefix = text[max(0, candidate["start"]-55):candidate["start"]]
                if re.search(r"(?:codice\s+fiscale|c\.?\s*f\.?)\s*/", prefix, re.I):
                    _flag(candidate, "tax_code_or_vat_ambiguous")
    _add_matches(result, text, _ADDRESS, "LOCATION", "street_address", ("address_boundary_heuristic",))
    _add_matches(result, text, _ROLES, "ROLE", "role_lexicon", ("role_context_unverified",))
    _add_matches(result, text, _FORM_PERSON, "PERSON", "person_form_context", ("person_form_context",))
    for pattern in _FORM_FIELDS:
        for match in pattern.finditer(text):
            for field in ("first", "second"):
                start, end = match.span(field)
                result.append(_entity(text, start, end, "PERSON", "rule:person_form_context",
                                      ("person_form_context",)))
    return result


def _extract_times(text: str, entities: list) -> list:
    """Extract dates, times and selected relative expressions from original text.

    Previously detected identifiers are protected to avoid interpreting their
    digits as dates. More specific time spans precede years and contained dates.
    Month/day ambiguity does not affect the TIME label or trigger normalization.
    """
    protected = [e for e in entities if e["label"] in {"EMAIL", "PHONE", "TAX_CODE", "VAT_NUMBER", "BANK_ACCOUNT"}]
    times = []
    for pattern, rule, reasons in [
        (_HOURS, "time_of_day", ()), (_DATE_RANGE, "date_range", ()),
        (_DATES, "date", ()), (_YEARS, "year", ("year_context_unverified",)),
        (_SHORT_YEARS, "abbreviated_year", ("abbreviated_year",)),
        (_RELATIVE_TIME, "relative_time", ("relative_time_context",)),
    ]:
        batch = []
        _add_matches(batch, text, pattern, "TIME", rule, reasons)
        for candidate in batch:
            if any(_overlaps(candidate, e) for e in protected):
                continue
            # Keep the complete interval instead of adding its individual dates.
            if any(e["start"] <= candidate["start"] and candidate["end"] <= e["end"] for e in times):
                continue
            if rule == "date" and re.fullmatch(_NUMERIC_DATE, candidate["text"]):
                parts = re.split(r"\s*[/.-]\s*", candidate["text"])
                first, second = int(parts[0]), int(parts[1])
                # Either day/month order is accepted for mention recognition.
                day_first = 1 <= first <= 31 and 1 <= second <= 12
                month_first = 1 <= first <= 12 and 1 <= second <= 31
                if not day_first and not month_first:
                    _flag(candidate, "unusual_date_format")
            times.append(candidate)
    return times


def _extract_nlp(text: str, doc) -> list:
    """Map spaCy entity spans to project labels without changing their offsets.

    ORG includes institutions as well as companies. Spans without a recognized
    legal suffix retain a tentative COMPANY label and an explicit review flag.
    POS tags supply additional review signals, not automatic rejection rules.
    """
    if doc.text != text:
        raise ValueError("The NLP component must preserve the original text.")
    labels = {"PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION",
              "GPE": "LOCATION", "FAC": "LOCATION"}
    has_pos = doc.has_annotation("POS")
    result = []
    for span in doc.ents:
        label = labels.get(span.label_)
        reasons = ()
        if span.label_ == "ORG":
            label = "COMPANY"
            if not _COMPANY_SUFFIX.search(span.text):
                reasons = ("organization_not_necessarily_company",)
        if not label:
            continue
        candidate = _entity(text, span.start_char, span.end_char, label, "spacy:ner", reasons)
        candidate["nlp_label"] = span.label_
        if label in {"PERSON", "LOCATION"}:
            if has_pos and not any(token.pos_ == "PROPN" for token in span):
                _flag(candidate, "name_without_proper_noun")
            if _LEGAL_REFERENCE.fullmatch(span.text.strip()):
                _flag(candidate, "legal_reference_as_name")
            if _FORM_TERM.fullmatch(span.text.strip()):
                _flag(candidate, "common_form_term_as_name")
            if re.search(r"\s+[MF](?:\s+[MF])?$", span.text):
                _flag(candidate, "possible_form_fields_in_name")
            if _PERSON_FIELD_LABEL.search(span.text):
                _flag(candidate, "possible_form_fields_in_name")
        if label == "LOCATION":
            prefix = _ADMINISTRATIVE_PLACE.match(span.text)
            if prefix and prefix.end() < len(span.text):
                candidate["start"] += prefix.end()
                candidate["text"] = text[candidate["start"]:candidate["end"]]
                _flag(candidate, "administrative_place_boundary")
            prefix = _RESIDENCE_PREFIX.match(candidate["text"])
            if prefix and prefix.end() < len(candidate["text"]):
                candidate["start"] += prefix.end()
                candidate["text"] = text[candidate["start"]:candidate["end"]]
                _flag(candidate, "residence_place_boundary")
            if _GENERIC_PLACE.fullmatch(candidate["text"].strip()):
                _flag(candidate, "generic_place_term")
        result.append(candidate)
    return result


def _reconcile_form_names(text: str, entities: list) -> list:
    """Prefer a form-based person span over a conflicting NLP prediction.

    Reconciliation allows only field labels, separators and an M/F selection
    suffix around the rule spans. A span covering paired fields is replaced by
    their separate values. The NLP prediction is retained as alternative evidence.
    """
    names = [e for e in entities if e["source"] == "rule:person_form_context"]
    kept = []
    for entity in entities:
        matched = []
        if entity["source"] == "spacy:ner" and entity["label"] in {"PERSON", "LOCATION"}:
            contained = sorted(
                (name for name in names
                 if entity["start"] <= name["start"] and name["end"] <= entity["end"]),
                key=lambda name: name["start"],
            )
            if contained:
                cursor = entity["start"]
                gaps = []
                for name in contained:
                    gaps.append(text[cursor:name["start"]])
                    cursor = name["end"]
                gaps.append(text[cursor:entity["end"]])
                # Do not remove predictions that also contain unexplained words.
                clean_gaps = all(
                    not _PERSON_FIELD_LABEL.sub("", gap).strip(" \t\r\n:=")
                    for gap in gaps[:-1]
                )
                suffix = gaps[-1]
                clean_suffix = (not _PERSON_FIELD_LABEL.sub("", suffix).strip(" \t\r\n:=")
                                or bool(_SEX_FIELDS.fullmatch(suffix)))
                equal_person = (len(contained) == 1 and entity["label"] == "PERSON"
                                and entity["start"] == contained[0]["start"]
                                and entity["end"] == contained[0]["end"])
                if clean_gaps and clean_suffix and not equal_person:
                    matched = contained
        if not matched:
            kept.append(entity)
        else:
            for name in matched:
                name.setdefault("alternative_predictions", []).append(dict(entity))
                _flag(name, "form_name_prediction_adjusted")
    return kept


def _merge_addresses(text: str, result: list) -> list:
    """Extend address candidates with adjacent city names and remove inner places.

    Only short gaps containing spaces or commas are accepted. Generic or suspect
    place candidates cannot extend an address. Contained NLP names are recorded
    as absorbed evidence, following the agreed flat address convention. Partial
    overlaps remain separate candidates for review.
    """
    addresses = [e for e in result if e["source"] == "rule:street_address"]
    blocked_reasons = {
        "generic_place_term", "name_without_proper_noun",
        "legal_reference_as_name", "possible_form_fields_in_name",
        "common_form_term_as_name",
    }
    for address in addresses:
        for place in result:
            if place["source"] == "spacy:ner" and place["label"] == "LOCATION":
                if blocked_reasons.intersection(place["review_reasons"]):
                    continue
                gap = text[place["end"]:address["start"]]
                if place["end"] <= address["start"] and len(gap) <= 4 and not gap.strip(" ,\t"):
                    address["start"] = place["start"]
                    address["text"] = text[address["start"]:address["end"]]
    kept = []
    for entity in result:
        container = None
        if entity["source"] == "spacy:ner" and entity["label"] in {"LOCATION", "PERSON", "COMPANY"}:
            container = next((a for a in addresses
                              if a["start"] <= entity["start"] and entity["end"] <= a["end"]), None)
        if container is not None:
            container.setdefault("absorbed_entities", []).append(dict(entity))
        else:
            kept.append(entity)
    return kept


def _finish_entities(result: list) -> list:
    """Deduplicate equal spans and labels, merge their sources, and sort results.

    Repeated mentions at different positions remain separate. Overlapping spans
    with different boundaries or labels are retained and flagged for later review.
    """
    unique = {}
    for candidate in result:
        key = (candidate["start"], candidate["end"], candidate["label"])
        if key not in unique:
            unique[key] = candidate
        else:
            existing = unique[key]
            existing["source"] += ";" + candidate["source"]
            if "nlp_label" in candidate:
                existing["nlp_label"] = candidate["nlp_label"]
            for reason in candidate["review_reasons"]:
                _flag(existing, reason)
    ordered = sorted(unique.values(), key=lambda e: (e["start"], e["end"], e["label"]))
    for index, first in enumerate(ordered):
        for second in ordered[index+1:]:
            if second["start"] >= first["end"]:
                break
            if _overlaps(first, second):
                _flag(first, "overlapping_candidates")
                _flag(second, "overlapping_candidates")
    return ordered


def extract_entities(text: str, nlp=None, *, doc=None) -> list:
    """Return first-pass candidates for one complete document.

    Args:
        text: Original document text, without normalization.
        nlp: Loaded Italian spaCy pipeline, or None to run only explicit rules.
        doc: Already processed spaCy Doc, for use with nlp.pipe. Supply either
            nlp or doc; passing a Doc avoids processing the same text twice.

    Returns:
        Candidate dictionaries sorted by character position. Each contains its
        text, label, offsets, source and review information. Review flags are
        heuristic signals, not confidence scores; unflagged mentions may be wrong.

    Raises:
        TypeError: If text is not a string.
        ValueError: If both nlp and doc are supplied, or the Doc text differs.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if nlp is not None and doc is not None:
        raise ValueError("Supply either nlp or doc, not both.")
    if nlp is not None:
        doc = nlp(text)

    entities = _extract_rules(text)
    entities.extend(_extract_times(text, entities))
    if doc is not None:
        entities.extend(_extract_nlp(text, doc))
    entities = _reconcile_form_names(text, entities)
    entities = _merge_addresses(text, entities)
    return _finish_entities(entities)
