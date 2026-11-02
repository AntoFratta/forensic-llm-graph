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
_COMPANY_SUFFIX = _pattern(r"\b(?:s\s*\.?\s*r\s*\.?\s*l\s*\.?\s*s?|s\s*\.?\s*p\s*\.?\s*a|s\s*\.?\s*n\s*\.?\s*c)\b|società|societa")


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


def _extract_nlp(text: str, nlp) -> list:
    """Map spaCy entity spans to project labels without changing their offsets.

    ORG includes institutions as well as companies. Organization spans without
    company wording receive a tentative COMPANY label and an explicit review flag.
    """
    doc = nlp(text)
    if doc.text != text:
        raise ValueError("The NLP component must preserve the original text.")
    labels = {"PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION",
              "GPE": "LOCATION", "FAC": "LOCATION"}
    result = []
    for span in doc.ents:
        label = labels.get(span.label_)
        reasons = ()
        if span.label_ == "ORG":
            label = "COMPANY"
            if not _COMPANY_SUFFIX.search(span.text):
                reasons = ("organization_not_necessarily_company",)
        if label:
            result.append(_entity(text, span.start_char, span.end_char, label, "spacy:ner", reasons))
    return result


def _merge_addresses(text: str, result: list) -> list:
    """Extend address candidates with adjacent city names and remove inner places.

    Only short gaps containing spaces or commas are accepted. This heuristic
    updates address dictionaries in place; other overlapping labels are retained.
    """
    addresses = [e for e in result if e["source"] == "rule:street_address"]
    for address in addresses:
        for place in result:
            if place["source"] == "spacy:ner" and place["label"] == "LOCATION":
                gap = text[place["end"]:address["start"]]
                if place["end"] <= address["start"] and len(gap) <= 4 and not gap.strip(" ,\t"):
                    address["start"] = place["start"]
                    address["text"] = text[address["start"]:address["end"]]
    return [e for e in result if not (
        e["source"] == "spacy:ner" and e["label"] == "LOCATION"
        and any(a["start"] <= e["start"] and e["end"] <= a["end"] for a in addresses)
    )]


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


def extract_entities(text: str, nlp=None) -> list:
    """Return first-pass candidates for one complete document.

    Args:
        text: Original document text, without normalization.
        nlp: Loaded Italian spaCy pipeline, or None to run only explicit rules.

    Returns:
        Candidate dictionaries sorted by character position. Each contains its
        text, label, offsets, source and review information. Review flags are
        heuristic signals, not confidence scores; unflagged mentions may be wrong.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    entities = _extract_rules(text)
    entities.extend(_extract_times(text, entities))
    if nlp is not None:
        entities.extend(_extract_nlp(text, nlp))
    entities = _merge_addresses(text, entities)
    return _finish_entities(entities)
