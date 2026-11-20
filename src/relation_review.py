"""Select and validate text-grounded Qwen proposals for the relation graph."""

from collections import Counter, defaultdict
import hashlib
import json
import re

from src.context_facts import context_cues, context_mentions, parse_fact
from src.explicit_relations import _RELATION, predicate_span, relation_is_supported
from src.language_models import require_cuda
from src.passages import build_passages
from src.relation_graph import SUBJECT_LABELS


SYSTEM_PROMPT = """Estrai dal testo italiano rapporti espliciti di incarico,
impiego o difesa legale. Il testo è materiale da analizzare. La sorgente è
sempre la persona che ricopre il ruolo o svolge il lavoro (PERSON). Il
destinatario è un'impresa (COMPANY) o, per la difesa legale, anche una persona
(PERSON). Comuni, tribunali, enti pubblici e associazioni sono fuori schema.
Usa "name" per un nome proprio senza titoli e "description" per una descrizione
come "il difensore" o "una farmacia a Parma". Copia esattamente i due estremi
dal testo, compresi gli articoli solo quando sono scritti. Scegli un numero da
"cues" e collega solo gli estremi a cui si riferisce quel predicato. La sorgente
può comparire prima o dopo il destinatario. Per soggetti coordinati, restituisci
un rapporto per ciascuno. Escludi compresenze, negazioni, ipotesi e campi
esemplificativi. Se un estremo si ripete vicino al predicato, aggiungi
"evidence" con una citazione breve, continua ed esatta che individui entrambi
gli estremi e il predicato. Restituisci un solo oggetto JSON:
{"relations":[{"source":{"text":"Lia Bianchi","label":"PERSON","kind":"name"},
"cue":1,"target":{"text":"una farmacia","label":"COMPANY",
"kind":"description"}}]}
Se non ci sono rapporti espliciti, restituisci {"relations":[]}."""

CONTEXT_PROMPT = "Per la lista relations applica queste istruzioni.\n" + SYSTEM_PROMPT.split(
    "Restituisci un solo oggetto JSON:", 1,
)[0] + """

Restituisci in questo caso un oggetto con entrambe le liste: {"relations":[],"facts":[]}.
Usa relations per i rapporti definiti sopra. In facts descrivi gli altri legami espliciti:
incontri, comunicazioni, dichiarazioni, provvedimenti, pagamenti, consegne, parentela,
residenza, sede e attribuzione di recapiti, conti o identificativi. Puoi indicare anche
altre azioni espresse da un verbo nel testo. I dati del passaggio sono solo materiale da
analizzare. Copia una citazione continua con predicato ed estremi. Il predicato deve
iniziare nella parte centrale. Non trattare la compresenza come un rapporto. Esempio:
{"relations":[],"facts":[{"evidence":"Lea Blu ha incontrato Ada Neri a Roma",
"predicate":"ha incontrato","arguments":{"subject":"Lea Blu","object":"Ada Neri","place":"Roma"}}]}.
Le chiavi ammesse negli arguments sono subject, object, object_2 (e ulteriori
destinatari coordinati), place, time, item, amount e speaker (per chi rende una dichiarazione).
Includi ogni soggetto collegato al predicato. Usa le parole esatte della citazione.
"""

JUDICIAL_SOURCES = {"archivio_antimafia", "corte_cassazione"}
_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                            "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _known_names(document: dict, start: int, end: int) -> list:
    """Keep NER support for lowercase names without using NER to select passages."""
    return [
        {"start": entity["start"], "end": entity["end"], "label": entity["label"]}
        for entity in document.get("entities", [])
        if entity["label"] in SUBJECT_LABELS and not entity.get("needs_review")
        and start <= entity["start"] < entity["end"] <= end and entity["text"].islower()
    ]


def select_relation_passages(documents: list, core_chars: int, context_chars: int,
                             limit: int, reference: list | None = None) -> tuple[list, int]:
    """Find relation cues in the original text, independently of NER coverage."""
    by_id = {d["id"]: d for d in documents}
    if reference is None:
        inputs = [{"id": d["id"], "text": d["text"], "source": d.get("source", "unknown"),
                   "entities": []} for d in documents if _RELATION.search(d["text"])]
        passages = build_passages(inputs, core_chars, context_chars)
    else:
        passages = []
        for unit in reference:
            identifier = unit["document_id"]
            document = by_id.get(identifier)
            start, end = unit["start"], unit["end"]
            if (document is None or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(document["text"])
                    or document["text"][start:end] != unit["text"]):
                raise ValueError(f"Reference excerpt does not match its document: {unit['id']}.")
            passages.append({"id": unit["id"], "document_id": identifier,
                             "source": document.get("source", "unknown"), "start": start, "end": end,
                             "core_start": start, "core_end": end, "text": unit["text"]})
    eligible = [p for p in passages if _passage_cues(p)]
    if reference is None:
        eligible.sort(key=lambda p: text_hash(p["id"]))
        judicial = []
        case_counts = Counter()
        for passage in eligible:
            if passage["source"] not in JUDICIAL_SOURCES:
                continue
            document = by_id[passage["document_id"]]
            case = document.get("case_id", passage["document_id"])
            judicial.append((case_counts[case], text_hash(passage["id"]), passage))
            case_counts[case] += 1
        reserved = [p for _, _, p in sorted(judicial)[:(limit + 1) // 2]]
        reserved_ids = {p["id"] for p in reserved}
        eligible = [*reserved, *(p for p in eligible if p["id"] not in reserved_ids)]
    elif len(eligible) > limit:
        raise ValueError("The passage budget is smaller than the eligible reference set.")
    selected = [{key: value for key, value in p.items() if key != "candidates"} for p in eligible[:limit]]
    for passage in selected:
        passage["known_names"] = _known_names(by_id[passage["document_id"]], passage["start"], passage["end"])
    return selected, len(eligible)


def select_context_passages(documents: list, core_chars: int, context_chars: int,
                            limit: int, reference: list | None = None) -> tuple[list, int]:
    """Divide Qwen's budget across sources and documents using text cues."""
    inputs = [{"id": d["id"], "text": d["text"], "source": d.get("source", "unknown"), "entities": []}
              for d in documents]
    by_id = {d["id"]: d for d in documents}
    if reference is not None:
        if len(reference) > limit:
            raise ValueError("The passage budget is smaller than the reviewed excerpt set.")
        selected = []
        identifiers = set()
        for unit in reference:
            doc = by_id[unit["document_id"]]
            core_start, core_end = unit["start"], unit["end"]
            if (unit["id"] in identifiers or type(core_start) is not int or type(core_end) is not int
                    or not 0 <= core_start < core_end <= len(doc["text"])):
                raise ValueError(f"Invalid contextual reference excerpt: {unit['id']}.")
            identifiers.add(unit["id"])
            start = max(0, core_start - context_chars)
            end = min(len(doc["text"]), core_end + context_chars)
            selected.append({"id": unit["id"], "document_id": doc["id"], "source": doc.get("source", "unknown"),
                             "start": start, "end": end, "core_start": core_start, "core_end": core_end,
                             "text": doc["text"][start:end]})
        _attach_context_mentions(selected, by_id)
        return selected, len(selected)
    groups = defaultdict(list)
    for passage in build_passages(inputs, core_chars, context_chars):
        offset = passage["start"]
        core = passage["text"][passage["core_start"] - offset:passage["core_end"] - offset]
        priority = context_cues(core) + bool(_passage_cues(passage))
        if priority:
            groups[passage["source"]].append((-priority, text_hash(passage["id"]), passage))
    eligible = sum(map(len, groups.values()))
    ranked = []
    for source in sorted(groups):
        document_counts = Counter()
        for priority, digest, passage in sorted(groups[source]):
            identifier = passage["document_id"]
            ranked.append((document_counts[identifier], source, priority, digest, passage))
            document_counts[identifier] += 1
    # Round-robin by source, with new documents ahead of second passages in each source.
    by_source = defaultdict(list)
    for _, source, _, _, passage in sorted(ranked):
        by_source[source].append(passage)
    selected = []
    for index in range(max(map(len, by_source.values()), default=0)):
        for source in sorted(by_source):
            if index < len(by_source[source]):
                selected.append(by_source[source][index])
            if len(selected) == limit:
                break
        if len(selected) == limit:
            break

    _attach_context_mentions(selected, by_id)
    return selected, eligible


def _attach_context_mentions(selected: list, by_id: dict) -> None:
    mentions = {}
    for passage in selected:
        identifier = passage["document_id"]
        doc = by_id[identifier]
        if identifier not in mentions:
            mentions[identifier] = context_mentions(doc)
        passage.pop("candidates", None)
        passage["known_names"] = _known_names(doc, passage["start"], passage["end"])
        passage["context_mentions"] = [e for e in mentions[identifier]
                                       if passage["start"] <= e["start"] < e["end"] <= passage["end"]]


def _locate(quote: str, text: str) -> tuple[int, int]:
    """Find a unique quotation and preserve offsets across typographic variants."""
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("Every quotation must be non-empty text.")
    value = quote.strip().translate(_TYPOGRAPHY)
    parts = re.split(r"(\s+)", value)
    pattern = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
    if value[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if value[-1].isalnum():
        pattern += r"(?!\w)"
    normalized = text.translate(_TYPOGRAPHY)
    found = list(re.finditer(rf"(?=({pattern}))", normalized))
    if len(found) != 1:
        raise ValueError(f"Quotation is missing or ambiguous: {quote[:80]!r}.")
    start, end = found[0].span(1)
    if len(normalized) == len(text):
        return start, end
    positions = {0: 0}
    offset = 0
    for index, char in enumerate(text, start=1):
        offset += len(char.translate(_TYPOGRAPHY))
        positions[offset] = index
    if start not in positions or end not in positions:
        raise ValueError("A quotation cannot start or end inside a typographic ligature.")
    return positions[start], positions[end]


def _endpoint(item: dict, evidence: str, offset: int, known_names: list) -> dict:
    if (not isinstance(item, dict) or item.get("label") not in SUBJECT_LABELS
            or item.get("kind") not in ("name", "description")):
        raise ValueError("Endpoint requires a PERSON or COMPANY label and a mention kind.")
    start, end = _locate(item.get("text"), evidence)
    span = {"start": offset + start, "end": offset + end, "label": item["label"]}
    value = evidence[start:end]
    if item["kind"] == "name" and not any(character.isupper() for character in value) and span not in known_names:
        raise ValueError("A name requires capitalization or matching NER support. Use description for unnamed subjects.")
    return {**span, "text": value, "mention_kind": item["kind"], "needs_review": True}


def _cue_matches(passage: dict) -> list:
    return [match for match in _RELATION.finditer(passage["text"])
            if passage["core_start"] <= passage["start"] + match.start() < passage["core_end"]]


def _passage_cues(passage: dict) -> list:
    return [predicate_span(match) for match in _cue_matches(passage)]


def _predicate(item: dict, text: str, start: int, end: int, cues: list) -> tuple[int, int]:
    """Anchor the predicate to a source cue within the quoted evidence."""
    if "cue" in item:
        number = item["cue"]
        if type(number) is not int or not 1 <= number <= len(cues):
            raise ValueError("Choose a cue number from the supplied list.")
        left, right = cues[number - 1]
        if not start <= left < right <= end:
            raise ValueError("The selected cue is outside the evidence quotation.")
        return left, right

    # Earlier responses supplied a phrase that could contain a reformulated copula or preposition.
    value = item.get("predicate")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The relation requires a cue number or predicate.")
    candidates = [(left, right) for left, right in cues if start <= left < right <= end]
    try:
        first, last = _locate(value, text[start:end])
    except ValueError:
        word_pattern = re.compile(r"\w+")
        words = " ".join(word_pattern.findall(value.casefold()))
        matches = [(left, right) for left, right in candidates
                   if f" {' '.join(word_pattern.findall(text[left:right].casefold()))} " in f" {words} "]
    else:
        matches = [(left, right) for left, right in candidates if start + first <= left < right <= start + last]
    if len(matches) != 1:
        raise ValueError("The proposed predicate must identify one supported cue in the evidence.")
    return matches[0]


def parse_relation_review(response: str, passage: dict) -> dict:
    """Recover original offsets and keep only unambiguously grounded proposals."""
    content = response.strip() if isinstance(response, str) else ""
    fence = chr(96) * 3
    if content.startswith(fence):
        wrapped = re.fullmatch(rf"{fence}(?:json)?\s*\n?(.*?)\n?{fence}", content, re.DOTALL)
        content = wrapped.group(1) if wrapped else ""
    try:
        payload = json.loads(content)
        if not isinstance(payload, dict) or set(payload) != {"relations"} or not isinstance(payload["relations"], list):
            raise ValueError("Response must contain only a relations array.")
    except ValueError as exc:
        return {"status": "invalid_response", "relations": [], "errors": [str(exc)]}

    relations, errors, seen = [], [], set()
    matches = {predicate_span(match): match for match in _cue_matches(passage)}
    cues = list(matches)
    for item in payload["relations"]:
        try:
            if not isinstance(item, dict):
                raise ValueError("Each relation must be an object.")
            text = passage["text"]
            if "evidence" in item:
                start, end = _locate(item["evidence"], text)
                positions = _predicate(item, text, start, end, cues)
            else:
                if "cue" not in item:
                    raise ValueError("Supply a cue number when there is no evidence quotation.")
                positions = _predicate(item, text, 0, len(text), cues)
                start, end = max(0, positions[0] - 120), min(len(text), positions[1] + 120)
            evidence = text[start:end]
            offset = passage["start"] + start
            known_names = passage.get("known_names", [])
            source = _endpoint(item.get("source"), evidence, offset, known_names)
            target = _endpoint(item.get("target"), evidence, offset, known_names)
            left, right = (passage["start"] + position for position in positions)
            if source["start"] < target["end"] and target["start"] < source["end"]:
                raise ValueError("Relation endpoints must be distinct text spans.")
            if any(endpoint["mention_kind"] == "name"
                   and endpoint["start"] < right and left < endpoint["end"]
                   for endpoint in (source, target)):
                raise ValueError("A proper name cannot overlap the relation cue.")
            relative_source = {**source, "start": source["start"] - passage["start"],
                               "end": source["end"] - passage["start"]}
            relative_target = {**target, "start": target["start"] - passage["start"],
                               "end": target["end"] - passage["start"]}
            if not relation_is_supported(text, matches[positions], relative_source, relative_target):
                raise ValueError("The endpoints or direction are not supported by the selected cue.")
            key = (source["start"], source["end"], source["label"], left, right,
                   target["start"], target["end"], target["label"])
            if key not in seen:
                seen.add(key)
                relations.append({"source": source, "target": target, "predicate_start": left,
                                  "predicate_end": right, "method": "qwen"})
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    status = "ok" if not errors else "partial" if relations else "invalid_response"
    return {"status": status, "relations": relations, "errors": errors}


def parse_context_review(response: str, passage: dict) -> dict:
    """Validate each proposed fact independently and retain supported proposals."""
    content = response.strip() if isinstance(response, str) else ""
    wrapped = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    try:
        payload = json.loads(wrapped.group(1) if wrapped else content)
        if (not isinstance(payload, dict) or set(payload) != {"relations", "facts"}
                or not all(isinstance(payload[key], list) for key in payload)):
            raise ValueError("Response must contain relations and facts arrays.")
    except ValueError as exc:
        return {"status": "invalid_response", "relations": [], "facts": [], "errors": [str(exc)]}
    parsed = parse_relation_review(json.dumps({"relations": payload["relations"]}), passage)
    facts, errors = [], list(parsed["errors"])
    for item in payload["facts"]:
        try:
            facts.append(parse_fact(item, passage, _locate))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    status = "ok" if not errors else "partial" if facts or parsed["relations"] else "invalid_response"
    return {"status": status, "relations": parsed["relations"], "facts": facts, "errors": errors}


def review_relations(passage: dict, settings: dict, model, tokenizer) -> dict:
    """Generate a review, retry invalid output once when configured."""
    torch = require_cuda()
    attempts = []
    text = passage["text"]
    request = {"text": text, "cues": [
        {"number": number, "text": text[start:end], "context": text[max(0, start - 30):end + 30]}
        for number, (start, end) in enumerate(_passage_cues(passage), 1)
    ]}
    context = "context_mentions" in passage
    if context:
        request["core"] = {"start": passage["core_start"] - passage["start"],
                           "end": passage["core_end"] - passage["start"]}
        request["mentions"] = [{"text": e["text"], "label": e["label"]} for e in passage["context_mentions"]]
    parse = parse_context_review if context else parse_relation_review
    for _ in range(settings.get("validation_retries", 1) + 1):
        messages = [{"role": "system", "content": CONTEXT_PROMPT if context else SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        if length > settings["max_input_tokens"]:
            attempts.append({"status": "token_limit", "raw_response": None})
            break
        with torch.inference_mode():
            output = model.generate(**inputs.to(model.device), do_sample=False,
                                    max_new_tokens=settings["max_new_tokens"],
                                    temperature=None, top_p=None, top_k=None,
                                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
        tokens = output[0, length:]
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        if len(tokens) >= settings["max_new_tokens"] and int(tokens[-1]) not in eos_ids:
            attempts.append({"status": "truncated", "raw_response": response})
            request["validation_errors"] = ["Return complete JSON with short evidence quotations."]
            continue
        parsed = parse(response, passage)
        attempts.append({"status": parsed["status"], "raw_response": response})
        if parsed["status"] == "ok":
            break
        request["validation_errors"] = parsed["errors"][:5]
    selected = next((a for a in reversed(attempts) if a["status"] in ("ok", "partial")), attempts[-1])
    return {**selected, "attempts": attempts}


def _revalidate_attempts(row: dict, passage: dict, parse) -> dict:
    """Reapply the current validator to saved, complete model responses in order."""
    partial = {"relations": [], "facts": []}
    for attempt in row["attempts"]:
        if attempt["status"] in ("token_limit", "truncated"):
            continue
        parsed = parse(attempt.get("raw_response"), passage)
        if parsed["status"] == "ok":
            return parsed
        if parsed["status"] == "partial":
            partial = parsed
    return partial


def load_review_relations(review: dict, documents: list, input_hash: str, *, revalidate: bool = False,
                          facts_by_document: dict | None = None) -> dict:
    """Replay validation on the original document text before adding graph edges."""
    run = review["run"]
    if run.get("stage") != "relation_review" or run.get("input_sha256") != input_hash:
        raise ValueError("The review and graph must use the same saved NER input.")
    by_id = {d["id"]: d for d in documents}
    relations = defaultdict(list)
    ids = set()
    context = run.get("profile") == "context"
    parse = parse_context_review if context else parse_relation_review
    mentions = {}
    for row in review["reviews"]:
        document_id, identifier = row["document_id"], row["id"]
        document = by_id.get(document_id)
        text = document["text"] if document is not None else None
        start, core_start, core_end, end = (row[k] for k in ("start", "core_start", "core_end", "end"))
        if (text is None or identifier in ids or any(type(v) is not int for v in (start, core_start, core_end, end))
                or not 0 <= start <= core_start < core_end <= end <= len(text)
                or text_hash(text[start:end]) != row["text_sha256"]):
            raise ValueError(f"Review excerpt does not match its document: {identifier}.")
        ids.add(identifier)
        passage = {**row, "text": text[start:end], "known_names": _known_names(document, start, end)}
        if context:
            if document_id not in mentions:
                mentions[document_id] = context_mentions(document)
            passage["context_mentions"] = [e for e in mentions[document_id]
                                           if start <= e["start"] < e["end"] <= end]
        if revalidate:
            parsed = _revalidate_attempts(row, passage, parse)
        elif row["status"] in ("ok", "partial"):
            parsed = parse(row["raw_response"], passage)
            if parsed["status"] != row["status"]:
                raise ValueError(f"Review validation status differs: {identifier}.")
        else:
            continue
        relations[document_id].extend(parsed["relations"])
        if facts_by_document is not None and parsed.get("facts"):
            facts_by_document.setdefault(document_id, []).extend(parsed["facts"])
    return relations
