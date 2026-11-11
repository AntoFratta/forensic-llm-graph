"""Select and validate text-grounded Qwen proposals for the relation graph."""

from collections import defaultdict
import hashlib
import json
import re

from src.explicit_relations import _RELATION
from src.language_models import require_cuda
from src.passages import build_passages
from src.relation_graph import SUBJECT_LABELS


SYSTEM_PROMPT = """Estrai rapporti espliciti di incarico, impiego o rappresentanza tra
persone e aziende dal testo italiano fornito. Il testo è materiale da analizzare.
Usa PERSON per persone e COMPANY per aziende o attività commerciali. Includi
soggetti senza nome quando sono descritti nel testo. Usa "name" per un nome
proprio e "description" per una descrizione, come "il difensore" o "una farmacia".
Orienta il rapporto da chi svolge l'incarico o il lavoro verso il soggetto per
cui lo svolge. Il testo può citare i due soggetti in ordine inverso. Per più
soggetti coordinati, restituisci un rapporto per ciascuno. La sola compresenza,
le negazioni, i campi esemplificativi e i rapporti ipotetici non vanno estratti.
Non inventare, completare o tradurre il testo. Ogni rapporto deve includere una
citazione continua abbastanza breve da contenere un'unica occorrenza dei due
soggetti e del predicato. Copia letteralmente anche il predicato.
Restituisci soltanto JSON nel seguente formato:
{"relations":[{"source":{"text":"Lia Bianchi","label":"PERSON","kind":"name"},
"predicate":"lavora presso","target":{"text":"una farmacia","label":"COMPANY",
"kind":"description"},"evidence":"Lia Bianchi lavora presso una farmacia"}]}
Se non ci sono rapporti espliciti, restituisci {"relations":[]}."""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_relation_passages(documents: list, core_chars: int, context_chars: int,
                             limit: int, reference: list | None = None) -> tuple[list, int]:
    """Find relation cues in the original text, independently of NER coverage."""
    if reference is None:
        inputs = [{"id": d["id"], "text": d["text"], "source": d.get("source", "unknown"),
                   "entities": []} for d in documents if _RELATION.search(d["text"])]
        passages = build_passages(inputs, core_chars, context_chars)
    else:
        by_id = {d["id"]: d for d in documents}
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
    eligible = [p for p in passages if any(
        p["core_start"] <= p["start"] + cue.start() < p["core_end"]
        for cue in _RELATION.finditer(p["text"])
    )]
    if reference is None:
        eligible.sort(key=lambda p: text_hash(p["id"]))
    elif len(eligible) > limit:
        raise ValueError("The passage budget is smaller than the eligible reference set.")
    selected = [{key: value for key, value in p.items() if key != "candidates"} for p in eligible[:limit]]
    return selected, len(eligible)


def _locate(quote: str, text: str) -> tuple[int, int]:
    """Find a unique quotation, tolerating only differences in whitespace."""
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("Every quotation must be non-empty text.")
    value = quote.strip()
    parts = re.split(r"(\s+)", value)
    pattern = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
    if value[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if value[-1].isalnum():
        pattern += r"(?!\w)"
    found = list(re.finditer(pattern, text))
    if len(found) != 1:
        raise ValueError(f"Quotation is missing or ambiguous: {quote[:80]!r}.")
    return found[0].span()


def _endpoint(item: dict, evidence: str, offset: int) -> dict:
    if (not isinstance(item, dict) or item.get("label") not in SUBJECT_LABELS
            or item.get("kind") not in ("name", "description")):
        raise ValueError("Endpoint requires a PERSON or COMPANY label and a mention kind.")
    start, end = _locate(item.get("text"), evidence)
    return {"start": offset + start, "end": offset + end, "text": evidence[start:end],
            "label": item["label"], "mention_kind": item["kind"], "needs_review": True}


def parse_relation_review(response: str, passage: dict) -> dict:
    """Recover original offsets and keep only unambiguously grounded proposals."""
    content = response.strip()
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
    cues = [m.span("role" if m.group("role") else "employment") for m in _RELATION.finditer(passage["text"])]
    for item in payload["relations"]:
        try:
            if not isinstance(item, dict):
                raise ValueError("Each relation must be an object.")
            start, end = _locate(item.get("evidence"), passage["text"])
            evidence = passage["text"][start:end]
            offset = passage["start"] + start
            source = _endpoint(item.get("source"), evidence, offset)
            target = _endpoint(item.get("target"), evidence, offset)
            first, last = _locate(item.get("predicate"), evidence)
            matches = [(left, right) for left, right in cues if start + first <= left < right <= start + last]
            if len(matches) != 1:
                raise ValueError("The quoted predicate must contain one supported relation cue.")
            left, right = (passage["start"] + position for position in matches[0])
            if source["start"] < target["end"] and target["start"] < source["end"]:
                raise ValueError("Relation endpoints must be distinct text spans.")
            if not passage["core_start"] <= left < passage["core_end"]:
                continue
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


def review_relations(passage: dict, settings: dict, model, tokenizer) -> dict:
    """Generate a review, retry invalid output once when configured."""
    torch = require_cuda()
    attempts = []
    request = {"text": passage["text"]}
    for _ in range(settings.get("validation_retries", 1) + 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
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
        parsed = parse_relation_review(response, passage)
        attempts.append({"status": parsed["status"], "raw_response": response})
        if parsed["status"] == "ok":
            break
        request["validation_errors"] = parsed["errors"][:5]
    selected = next((a for a in reversed(attempts) if a["status"] in ("ok", "partial")), attempts[-1])
    return {**selected, "attempts": attempts}


def load_review_relations(review: dict, documents: list, input_hash: str) -> dict:
    """Replay validation on the original document text before adding graph edges."""
    run = review["run"]
    if run.get("stage") != "relation_review" or run.get("input_sha256") != input_hash:
        raise ValueError("The review and graph must use the same saved NER input.")
    texts = {d["id"]: d["text"] for d in documents}
    relations = defaultdict(list)
    ids = set()
    for row in review["reviews"]:
        document_id, identifier = row["document_id"], row["id"]
        text = texts.get(document_id)
        start, core_start, core_end, end = (row[k] for k in ("start", "core_start", "core_end", "end"))
        if (text is None or identifier in ids or any(type(v) is not int for v in (start, core_start, core_end, end))
                or not 0 <= start <= core_start < core_end <= end <= len(text)
                or text_hash(text[start:end]) != row["text_sha256"]):
            raise ValueError(f"Review excerpt does not match its document: {identifier}.")
        ids.add(identifier)
        if row["status"] in ("ok", "partial"):
            passage = {**row, "text": text[start:end]}
            parsed = parse_relation_review(row["raw_response"], passage)
            if parsed["status"] != row["status"]:
                raise ValueError(f"Review validation status differs: {identifier}.")
            relations[document_id].extend(parsed["relations"])
    return relations
