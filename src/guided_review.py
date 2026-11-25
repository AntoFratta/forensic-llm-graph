"""Prepare clause-local review questions and ground Qwen's corrections and additions."""

import hashlib
import json

from src import compact_review
from src.context_facts import mention_key, reviewable_facts


DECISION_PROMPT = """Esamina la frase italiana. I candidati sono ipotesi, anche errate. Controlla il tipo,
chi fa cosa, i dettagli espliciti (anche l'importo di un pagamento) e la qualificazione.
Rispondi con un oggetto JSON. In "decisions" dai una risposta per candidato nello stesso ordine:
1 se completo e corretto, 0 se assente, oppure un oggetto con i campi da correggere: kind,
predicate, assertion, arguments. Se modifichi arguments, elenca tutti i ruoli corretti.
Esempio di correzione per un pagamento: {"decisions":[{"arguments":{"subject":2,"amount":4}}]}.
I numeri in arguments sono ID di menzioni di questa frase. Usa ["testo esatto","CATEGORIA"]
per un partecipante mancante o con categoria errata. Non dedurre un partecipante solo dalla vicinanza.
Il predicato deve iniziare nel core. Chiudi il JSON dopo decisions. Non aggiungere altri campi."""

TYPE_INSTRUCTIONS = """
Tipi subject→object: family (parentela), residence (persona→luogo), registered_office (ente→luogo),
account_holder (intestatario→conto), contact (titolare→recapito), identifier (titolare→codice).
Eventi con subject e object: meeting, communication, agreement, action. Per procedure usa subject;
per payment usa subject ed eventualmente amount; per delivery anche item o object; per birth anche place;
per statement usa speaker. Altri ruoli possibili: place, time, speaker, object_2.
Assertion: asserted, reported, negated, hypothetical. Categorie: PERSON, COMPANY, ORGANIZATION, PARTY,
LOCATION, BANK_ACCOUNT, EMAIL, PHONE, TAX_CODE, VAT_NUMBER, OBJECT, AMOUNT, TIME."""

ADDITION_PROMPT = """Cerca altri rapporti o eventi espliciti nella frase italiana. I candidati elencati
sono già stati esaminati. Evita di ripetere gli stessi fatti con gli stessi partecipanti.
Rispondi con {"facts":[],"relations":[]}. Se non trovi aggiunte, restituisci entrambe le liste vuote.
In facts ogni nuovo fatto ha kind, predicate (parole esatte che iniziano nel core), assertion e arguments.
Per gli argomenti usa gli ID delle menzioni o ["parole esatte","CATEGORIA"] se la menzione manca.
Per incarico, impiego o difesa usa "relations": [[numero_cue,persona_che_ricopre_il_ruolo,impresa_o_assistito]].
Usa solo i cues forniti e due partecipanti espliciti PERSON o COMPANY. Ometti i rapporti privi di riscontro."""


def questions(passage, passage_cues):
    text, base = passage["text"], passage["start"]
    candidates = reviewable_facts(passage)
    groups = []
    for lo, hi in compact_review.clauses(text):
        start, end = base + lo, base + hi
        left, right = max(start, passage["core_start"]), min(end, passage["core_end"])
        if left >= right:
            continue
        local = {**passage, "text": text[lo:hi], "start": start, "end": end, "core_start": left,
                 "core_end": right, "context_mentions": [e for e in passage["context_mentions"]
                                                       if start <= e["start"] < e["end"] <= end]}
        mentions = {mention_key(e): n for n, e in enumerate(local["context_mentions"], 1)}
        hints, spans = [], []
        for fact in candidates:
            a, b = fact["predicate_start"], fact["predicate_end"]
            if not start <= a < b <= end:
                continue
            args = {role: (mentions.get(mention_key(e), [e["text"], e["label"]])
                           if start <= e["start"] < e["end"] <= end else None)
                    for role, e in fact["arguments"].items()}
            hints.append({"kind": fact["kind"], "predicate": text[a-base:b-base],
                          "assertion": fact["assertion"], "arguments": args})
            spans.append((a, b))
        cues = passage_cues(local)
        core = "tutta la frase" if (left, right) == (start, end) else local["text"][left-start:right-start]
        request = {"frase": local["text"], "core": core,
                   "menzioni": [[i, e["text"], e["label"]] for i, e in enumerate(local["context_mentions"], 1)],
                   "candidati": hints, "cues": [[i, local["text"][a:b]] for i, (a, b) in enumerate(cues, 1)]}
        # Bind positional decisions to the exact occurrence, request and mention spans on replay.
        signature = [start, end, left, right, request, spans, cues, [mention_key(e) for e in local["context_mentions"]]]
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        groups.append((local, request, spans, key))
    focused = [group for group in groups if group[1]["candidati"] or group[1]["cues"]]
    return focused or groups


def parse_answer(payload, question, locate, parse_roles):
    local, request, spans, _ = question
    if (not isinstance(payload, dict) or not {"decisions"} <= payload.keys() <= {"decisions", "facts", "relations"}
            or not all(isinstance(rows, list) for rows in payload.values())):
        raise ValueError("Return decisions, plus optional facts and relations arrays.")
    parsed = parse_roles(payload.get("relations", []), local, compact=True)
    parsed["facts"] = []
    candidates, decisions = request["candidati"], payload["decisions"]
    proposals = []
    if len(decisions) != len(candidates):
        # A shifted decision could silently confirm a different candidate.
        parsed["errors"].append(f"Expected exactly {len(candidates)} ordered decisions.")
    else:
        for hint, decision, span in zip(candidates, decisions, spans, strict=True):
            if type(decision) is int and decision in (0, 1):
                if decision:
                    proposals.append((hint, span))
            elif isinstance(decision, dict) and decision and decision.keys() <= hint.keys():
                same_predicate = decision.get("predicate", hint["predicate"]) == hint["predicate"]
                proposals.append((hint | decision, span if same_predicate else None))
            else:
                parsed["errors"].append("A candidate decision must be 0, 1, or an explicit correction.")
    proposals.extend((item, None) for item in payload.get("facts", []))
    for item, span in proposals:
        try:
            if not isinstance(item, dict) or set(item) != {"kind", "predicate", "assertion", "arguments"}:
                raise ValueError("A new fact needs kind, predicate, assertion and arguments.")
            row = [1, item["kind"], item["predicate"], item["assertion"], item["arguments"]]
            fact = compact_review.ground_fact(row, local, [(0, len(local["text"]))], locate, predicate_span=span)
            parsed["facts"].append(fact)
        except (TypeError, ValueError) as exc:
            parsed["errors"].append(str(exc))
    return parsed
