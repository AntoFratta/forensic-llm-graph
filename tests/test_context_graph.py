import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from scripts.graph import build_saved_graph, prepare, run_review
from src import guided_review
from src.context_facts import context_mentions, reviewable_facts
from src.graph_queries import inspect_graph, score_context
from src.relation_graph import build_graph
from src.relation_review import (_fact_reference, _guided_answer, _passage_cues, _parse_proposals, parse_context_review,
                                 parse_relation_review, review_relations, select_context_passages)


def document(identifier, text, mentions=(), **metadata):
    entities = []
    for mention, label in mentions:
        start = text.index(mention)
        entities.append({"start": start, "end": start + len(mention), "text": mention, "label": label,
                         "needs_review": False, "source": "test", "review_reasons": []})
    return {"id": identifier, "text": text, "entities": entities, **metadata}


def edges(graph, kind):
    return [edge for edge in graph["edges"] if edge["type"] == kind]


class ContextGraphTests(unittest.TestCase):
    def test_split_checkpoint_survives_interruption_and_truncated_additions(self):
        doc = document("lease", "Lea Blu abita a Roma.", [("Lea Blu", "PERSON"), ("Roma", "LOCATION")])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ner, plan, checkpoint = [root / name for name in ("ner.json", "plan.json", "partial.json")]
            ner.write_text(json.dumps({"run": {"stage": "hybrid"}, "documents": [doc]}), encoding="utf-8")
            plan.write_text(json.dumps(prepare(ner, None, 1, context=True)), encoding="utf-8")
            config = Path(__file__).resolve().parents[1] / "configs/hybrid.json"
            def vote(facts, *args):
                return [{"reference": _fact_reference(f), "answer": "1"} for f in facts]
            def interrupt(prompt, *args):
                if prompt.startswith(guided_review.DECISION_PROMPT):
                    return {"status": "truncated", "raw_response": '{"decisions":[1],"facts":['}
                self.assertTrue(checkpoint.exists())
                raise RuntimeError("interrupted during additions")
            def truncated(prompt, *args):
                self.assertTrue(prompt.startswith(guided_review.ADDITION_PROMPT))
                return {"status": "truncated", "raw_response": '{"facts":['}
            with patch("scripts.graph.load_model", return_value=(None, None, {})), patch(
                    "src.relation_review.require_cuda"), patch("src.relation_review._fact_responses", side_effect=vote):
                with patch("src.relation_review._generate_reply", side_effect=interrupt), self.assertRaisesRegex(
                        RuntimeError, "during additions"):
                    run_review(plan, config, checkpoint_path=checkpoint)
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                self.assertEqual(saved["pending_decisions"]["state"]["checks"][0]["answer"], "1")
                with patch("src.relation_review._generate_reply", side_effect=truncated):
                    result = run_review(plan, config, checkpoint_path=checkpoint)
            row, = result["reviews"]
            self.assertEqual((row["status"], row["accepted_count"]), ("partial", 1))
            self.assertNotIn("pending_decisions", json.loads(checkpoint.read_text(encoding="utf-8")))
            fact, = parse_context_review(row["raw_response"], json.loads(plan.read_text())["passages"][0])["facts"]
            self.assertEqual(fact["arguments"]["object"]["text"], "Roma")

    def test_repeated_addition_cannot_override_a_rejected_candidate(self):
        doc = document("lease", "Lea Blu abita a Roma.", [("Lea Blu", "PERSON"), ("Roma", "LOCATION")])
        passage = {"text": doc["text"], "start": 0, "core_start": 0, "core_end": len(doc["text"]),
                   "context_mentions": context_mentions(doc)}
        q, = guided_review.questions(passage, _passage_cues)
        def row(raw):
            return {"question_sha256": q[3], "raw_response": json.dumps(raw), "status": "ok"}
        fact, = _guided_answer("[1]", q, phase="decisions")["facts"]
        payload = {"decision_responses": [row({"decisions": [1]})],
                   "addition_responses": [row({"facts": q[1]["candidati"]})],
                   "checks": [{"reference": _fact_reference(fact), "answer": "0"}]}
        parsed = parse_context_review(json.dumps(payload), passage)
        self.assertEqual((parsed["status"], parsed["facts"]), ("ok", []))
        for raw in ('{"decisions":[1', '{"decisions":[]}', '{"decisions":[1,1]}'):
            self.assertEqual(_guided_answer(raw, q, phase="decisions")["status"], "invalid_response")

    def test_guided_decisions_correct_add_and_reject_using_only_the_source_clause(self):
        text = "Ada Neri ha pagato 500 euro. Lia Belli abita a Roma."
        doc = document("example", text, [("Ada Neri", "PERSON"), ("Lia Belli", "PERSON"), ("Roma", "LOCATION")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        pay, home = guided_review.questions(passage, _passage_cues)
        self.assertEqual([c["kind"] for c in pay[1]["candidati"]], ["payment"])
        self.assertEqual([c["kind"] for c in home[1]["candidati"]], ["residence"])
        self.assertNotIn("Roma", pay[1]["frase"])
        def parse(q, decisions, **fields):
            return _guided_answer(json.dumps({"decisions": decisions, **fields}), q)
        self.assertFalse(parse(pay, [0])["facts"])
        self.assertFalse(parse(pay, [])["facts"])
        self.assertEqual(parse(pay, [])["status"], "invalid_response")
        args = {"subject": 1, "amount": ["500 euro", "AMOUNT"]}
        corrected, = parse(pay, [{"arguments": args}])["facts"]
        self.assertEqual(corrected["arguments"]["amount"]["start"], text.index("500 euro"))
        self.assertFalse(parse(pay, [{"arguments": {**args, "subject": ["Lia Belli", "PERSON"]}}])["facts"])
        self.assertEqual(parse(home, [1])["facts"][0]["predicate_start"], text.index("abita"))
        extra = {"kind": "residence", "predicate": "abita", "assertion": "asserted",
                 "arguments": {"subject": 1, "object": 2}}
        self.assertEqual(parse(home, [0], facts=[extra])["facts"][0]["arguments"]["object"]["text"], "Roma")
        old_signature = home[3]
        home_raw = json.dumps({"decisions": [1]})
        rows = [{"question_sha256": pay[3], "raw_response": json.dumps({"decisions": [0]})},
                {"question_sha256": old_signature, "raw_response": home_raw}]
        fact, = _guided_answer(home_raw, home)["facts"]
        envelope = {"guided_responses": rows, "checks": [{"reference": _fact_reference(fact), "answer": "0"}]}
        self.assertFalse(parse_context_review(json.dumps(envelope), passage)["facts"])
        envelope["checks"][0]["answer"] = "1"
        self.assertEqual(parse_context_review(json.dumps(envelope), passage)["status"], "ok")
        rows[1]["question_sha256"] = pay[3]
        self.assertEqual(parse_context_review(json.dumps(envelope), passage)["status"], "invalid_response")

    def test_guided_failure_in_one_clause_retains_separate_verified_fact(self):
        text = "Ada Neri abita a Bari. Lea Blu abita a Roma."
        doc = document("example", text, [("Ada Neri", "PERSON"), ("Bari", "LOCATION"),
                                         ("Lea Blu", "PERSON"), ("Roma", "LOCATION")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        def reply(raw, status="complete"):
            return {"raw_response": raw, "status": status}
        def confirm(proposals, *args):
            return [{"reference": _fact_reference(f), "answer": "1"} for f in proposals]
        with patch("src.relation_review.require_cuda"), patch("src.relation_review._fact_responses", side_effect=confirm), \
                patch("src.relation_review._generate_reply", side_effect=[
                    reply("{}", "truncated"), reply('{"decisions":[]}'), reply('{"decisions":[1]}'),
                    reply('{"facts":[]}'), reply('{"facts":[]}')]) as generate:
            result = review_relations(passage, {"max_new_tokens": 400, "validation_retries": 1}, None, None)
        self.assertEqual(generate.call_count, 5)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["accepted_count"], 1)
        fact, = parse_context_review(result["raw_response"], passage)["facts"]
        self.assertEqual(fact["arguments"]["object"]["text"], "Roma")

    def test_context_generation_proposes_a_fact_without_role_cues_or_listed_rule_candidates(self):
        text = "Ada dimora a Bari."
        passage = {"text": text, "start": 50, "core_start": 50, "core_end": 50 + len(text),
                   "context_mentions": []}
        item = {"kind": "residence", "predicate": "dimora a", "assertion": "asserted",
                "arguments": {"subject": ["Ada", "PERSON"], "object": ["Bari", "LOCATION"]}}
        raw = json.dumps({"facts": [item]})
        tokenizer, model, torch = MagicMock(), MagicMock(), MagicMock()
        torch.inference_mode.side_effect = nullcontext
        tokenizer.return_value["input_ids"].shape = (1, 30)
        tokenizer.encode.side_effect = [[10], [11]]
        tokenizer.decode.side_effect = [raw, "1"]
        model.generation_config.eos_token_id = 0
        with patch("src.relation_review.require_cuda", return_value=torch):
            result = review_relations(passage, {"max_input_tokens": 3072, "max_new_tokens": 1536}, model, tokenizer)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["accepted_count"], 1)
        fact, = parse_context_review(result["raw_response"], passage)["facts"]
        self.assertEqual(fact["predicate_start"], 54)
        self.assertEqual(fact["arguments"]["object"]["text"], "Bari")
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(model.generate.call_args_list[0].kwargs["max_new_tokens"], 384)
        self.assertEqual(model.generate.call_args_list[1].kwargs["max_new_tokens"], 1)

    def test_compact_fact_uses_exact_indexed_occurrence_and_original_clause_for_evidence(self):
        text = "Ada parte. Ada incontra Ada a Bari."
        base = 50
        mentions = [{"text": "Ada", "label": "PERSON", "start": base+i, "end": base+i+3} for i in (0, 11, 24)]
        passage = {"text": text, "start": base, "core_start": base+11, "core_end": base+len(text),
                   "context_mentions": mentions}
        args = {"subject": 2, "object": 3, "place": ["Bari", "LOCATION"]}
        row = [2, "meeting", "incontra", "asserted", args]
        def parse(rows):
            raw = json.dumps({"relations": [], "facts": rows})
            return _parse_proposals(raw, passage, compact=True)
        accepted, = parse([row, row])["facts"]
        self.assertEqual(accepted["arguments"]["object"]["start"], base+24)
        self.assertEqual(accepted["arguments"]["place"]["source"], "qwen")
        self.assertEqual(text[accepted["start"]-base:accepted["end"]-base], "Ada incontra Ada a Bari")
        for wrong in (1, 0, 42, False, ["Ada", "PERSON"], ["Parma", "LOCATION"]):
            with self.subTest(ref=wrong):
                self.assertTrue(parse([[*row[:4], {**args, "object": wrong}]])["errors"])
        passage["core_end"] = base+12
        self.assertTrue(parse([row])["errors"])

    def test_compact_role_keeps_cue_direction_and_literal_missing_participant(self):
        text = "Lea Blu, difensore di fiducia di A.A., conosce Ada Neri."
        passage = {"text": text, "start": 20, "core_start": 20, "core_end": 20+len(text),
                   "context_mentions": [{"text": "Lea Blu", "start": 20, "end": 27, "label": "PERSON"}]}
        missing = ["A.A.", "PERSON"]
        raw = json.dumps({"relations": [[1, 1, missing], [1, missing, 1], [0, 1, missing]], "facts": []})
        parsed = parse_context_review(json.dumps({"compact_response": raw, "checks": []}), passage)
        relation, = parsed["relations"]
        self.assertEqual(parsed["status"], "partial")
        self.assertEqual(relation["target"]["start"], 20 + text.index("A.A."))
        self.assertEqual(relation["predicate_start"], 20 + text.index("difensore"))

    def test_new_proposals_add_unlisted_relations_and_events_with_quoted_participants(self):
        cases = [
            ("Lea Blu dimora stabilmente a Roma.", "residence", "dimora stabilmente a",
             {"subject": ("Lea Blu", "PERSON"), "object": ("Roma", "LOCATION")}),
            ("Ada Neri sborsò cinquanta euro.", "payment", "sborsò",
             {"subject": ("Ada Neri", "PERSON"), "amount": ("cinquanta euro", "AMOUNT")}),
        ]
        for text, kind, predicate, arguments in cases:
            with self.subTest(kind=kind):
                doc = document("example", text)
                passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                           "context_mentions": context_mentions(doc)}
                self.assertFalse(reviewable_facts(passage))
                item = {"kind": kind, "evidence": text, "predicate": predicate, "assertion": "asserted",
                        "arguments": {r: {"text": t, "label": l} for r, (t, l) in arguments.items()}}
                raw = json.dumps({"relations": [], "facts": [item, item]})
                proposed, = _parse_proposals(raw, passage)["facts"]
                vote = {"reference": _fact_reference(proposed), "answer": "1"}
                saved = {"proposal_response": raw, "checks": [vote]}
                parsed = parse_context_review(json.dumps(saved), passage)
                self.assertEqual(parsed["status"], "ok")
                graph = build_graph([doc], context=True, reviewed_facts={"example": parsed["facts"]})
                evidence_type = "CONTEXT_RELATION" if kind == "residence" else "STATED_IN"
                edge, = edges(graph, evidence_type)
                ev, = edge["evidence"]
                self.assertEqual(text[ev["predicate_start"]:ev["predicate_end"]], predicate)
                self.assertEqual(ev["method"], "qwen")
                vote["answer"] = "0"
                self.assertFalse(parse_context_review(json.dumps(saved), passage)["facts"])
                vote["reference"][1] += 1
                self.assertEqual(parse_context_review(json.dumps(saved), passage)["status"], "invalid_response")
                item["arguments"]["subject"]["text"] = "Un nome assente"
                self.assertTrue(_parse_proposals(json.dumps({"relations": [], "facts": [item]}), passage)["errors"])

    def test_meeting_keeps_participants_place_and_date_on_one_fact(self):
        doc = document("a", "Maria Rossi ha incontrato Luca Bianchi e Ada Neri a Roma il 12 maggio.",
                       [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON"), ("Ada Neri", "PERSON"),
                        ("Roma", "LOCATION"), ("12 maggio", "TIME")])
        graph = build_graph([doc], context=True)
        fact, = [n for n in graph["nodes"] if n["type"] == "FACT"]
        nodes = {n["id"]: n for n in graph["nodes"]}
        args = {e["role"]: nodes[e["target"]]["name"] for e in edges(graph, "FACT_ARGUMENT")
                if e["source"] == fact["id"]}
        self.assertEqual(args, {"subject": "Maria Rossi", "object": "Luca Bianchi", "object_2": "Ada Neri",
                                "place": "Roma", "time": "12 maggio"})
        expected = {"kind": "meeting", "predicate_start": 12, "predicate_end": 25, "assertion": "asserted",
                    "arguments": dict(zip(args, doc["entities"], strict=True))}
        report = score_context(graph, {"units": [{"id": "event", "document_id": "a", "start": 0,
                                                  "end": len(doc["text"]), "facts": [expected]}]})
        self.assertEqual(report["facts"]["overall"]["tp"], 1)
        shown = inspect_graph(graph, [doc], "a", ["FACT_ARGUMENT"])
        self.assertTrue(all("ha incontrato" in edge["evidence"][0]["quote"] for edge in shown["edges"]))

    def test_synthetic_account_has_a_directed_relation_and_evidence(self):
        iban = "IT60X0542811101000000123456"
        text = f"Il conto con IBAN {iban} è intestato ad Alfa S.r.l."
        doc = document("synthetic:a", text, [(iban, "BANK_ACCOUNT"), ("Alfa S.r.l", "COMPANY")], synthetic=True)
        graph = build_graph([doc], context=True)
        edge, = edges(graph, "CONTEXT_RELATION")
        nodes = {n["id"]: n for n in graph["nodes"]}
        self.assertEqual((edge["relation"], nodes[edge["target"]]["name"]), ("account_holder", iban))
        span = edge["evidence"][0]["source_mention"]
        self.assertEqual(text[span["start"]:span["end"]], "Alfa S.r.l.")
        self.assertTrue(nodes["document:synthetic:a"]["synthetic"])

    def test_account_title_and_unnamed_party_keep_the_correct_holder(self):
        iban = "IT60X0542811101000000123456"
        for wording, value, label, other_mentions in (
            ("al Sig. Susanna Catalano", "Susanna Catalano", "PERSON", [("Sig", "COMPANY")]),
            ("alla Parte Venditrice", "Parte Venditrice", "PARTY", []),
            ("alla società Logitalia Italia", "Logitalia Italia", "COMPANY", [("Logitalia Italia", "PERSON")]),
        ):
            with self.subTest(wording=wording):
                doc = document("a", f"L'IBAN {iban} è intestato {wording}.",
                               [(iban, "BANK_ACCOUNT"), *other_mentions], synthetic=True)
                graph = build_graph([doc], context=True)
                edge, = edges(graph, "CONTEXT_RELATION")
                nodes = {n["id"]: n for n in graph["nodes"]}
                holder = nodes[edge["source"]]
                self.assertEqual((holder["name"], holder["label"]), (value, label))

    def test_payment_requires_monetary_wording(self):
        mentions = [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON")]
        docs = [document("transfer", "Maria Rossi ha ricevuto 500 euro da Luca Bianchi.", mentions),
                document("call", "Maria Rossi ha ricevuto una telefonata da Luca Bianchi.", mentions),
                document("invoice", "Maria Rossi ha ricevuto una fattura da Luca Bianchi.", mentions),
                document("metaphor", "Maria Rossi ha pagato con la vita.", mentions[:1])]
        graph = build_graph(docs, context=True)
        payments = [n["document_id"] for n in graph["nodes"] if n.get("label") == "payment"]
        self.assertEqual(payments, ["transfer"])

    def test_review_checks_residence_separately_from_the_procedure_location(self):
        text = "I carabinieri hanno denunciato un 60enne residente a Carpi per furto."
        doc = document("a", text, [("Carpi", "LOCATION")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        candidates = {f["kind"]: f for f in reviewable_facts(passage)}
        self.assertIn("residence", candidates)
        procedure = candidates["procedure"]
        self.assertEqual(procedure["arguments"]["object"]["text"], "un 60enne")
        self.assertNotIn("place", procedure["arguments"])

    def test_qwen_can_accept_a_residence_formulation_outside_the_baseline(self):
        text = "Maria Rossi vive a Roma."
        doc = document("a", text, [("Maria Rossi", "PERSON"), ("Roma", "LOCATION")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        self.assertEqual(edges(build_graph([doc], context=True), "CONTEXT_RELATION"), [])
        candidate, = reviewable_facts(passage)
        self.assertTrue(candidate["additional_candidate"])
        vote = {"reference": _fact_reference(candidate), "answer": "1"}
        response = {"role_response": None, "fact_responses": [vote]}
        parsed = parse_context_review(json.dumps(response), passage)
        graph = build_graph([doc], context=True, reviewed_facts={"a": parsed["facts"]})
        self.assertEqual(edges(graph, "CONTEXT_RELATION")[0]["relation"], "residence")
        vote["reference"][1] += 1
        self.assertEqual(parse_context_review(json.dumps(response), passage)["status"], "invalid_response")

    def test_qwen_coordination_assigns_one_target_to_each_initial_pair(self):
        text = "Antonio Rossi, difensore di H.H., I.I. e J.J., ha depositato l'istanza."
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
        relation = {"cue": 1,
                    "source": {"text": "Antonio Rossi", "label": "PERSON", "kind": "name"},
                    "target": {"text": "H.H., I.I. e J.J.", "label": "COMPANY", "kind": "description"}}
        parsed = parse_relation_review(json.dumps({"relations": [relation]}), passage)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual([r["target"]["text"] for r in parsed["relations"]], ["H.H.", "I.I.", "J.J."])
        self.assertTrue(all(r["target"]["label"] == "PERSON" for r in parsed["relations"]))

    def test_review_does_not_skip_a_listed_person_to_attach_a_family_relation(self):
        text = "Bruno Verdi, Luca Bianchi, fratello di Ada Neri."
        doc = document("a", text, [("Bruno Verdi", "PERSON"), ("Ada Neri", "PERSON")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        self.assertEqual(reviewable_facts(passage), [])

        text = "Luca Bianchi, 45 anni, fratello di Ada Neri."
        doc = document("b", text, [("Luca Bianchi", "PERSON"), ("Ada Neri", "PERSON")])
        passage = {**passage, "text": text, "core_end": len(text), "context_mentions": context_mentions(doc)}
        fact, = reviewable_facts(passage)
        self.assertEqual(fact["arguments"]["subject"]["text"], "Luca Bianchi")

    def test_article_variants_confirm_the_same_legal_edge(self):
        text = "L 'Avvocato Antonio Rossi, difensore di H.H., I.I. e J.J."
        doc = document("a", text, [("Avvocato Antonio Rossi", "PERSON")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
        payload = {"relations": [{"cue": 1,
                                  "source": {"text": "L 'Avvocato Antonio Rossi", "label": "PERSON",
                                             "kind": "description"},
                                  "target": {"text": "H.H., I.I. e J.J.", "label": "PERSON", "kind": "name"}}]}
        parsed = parse_relation_review(json.dumps(payload), passage)
        graph = build_graph([doc], context=True, reviewed_relations={"a": parsed["relations"]})
        relations = edges(graph, "EXPLICIT_RELATION")
        self.assertEqual(len(relations), 3)
        self.assertEqual(len({r["source"] for r in relations}), 1)
        self.assertTrue(all(r["evidence"][0]["qwen_reviewed"] for r in relations))

    def test_agreement_recovers_two_parties_from_the_same_contract_phrase(self):
        text = "Il contratto intercorso tra Alfa e Beta è acquisito agli atti."
        doc = document("a", text, [("Alfa", "COMPANY"), ("Beta", "COMPANY")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        agreement, = reviewable_facts(passage)
        self.assertEqual(agreement["kind"], "agreement")
        self.assertEqual({role: mention["text"] for role, mention in agreement["arguments"].items()},
                         {"subject": "Alfa", "object": "Beta"})

    def test_agreement_can_recover_parties_absent_from_ner(self):
        text = ("Con la presente scrittura privata tra il Sig. Ada Rossi, in qualità di locatore, "
                "e la società Centro Verde, in qualità di conduttore, si conviene quanto segue.")
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text), "context_mentions": []}
        agreement, = reviewable_facts(passage)
        self.assertEqual(agreement["kind"], "agreement")
        self.assertEqual(agreement["arguments"]["subject"]["text"], "Ada Rossi")
        self.assertEqual(agreement["arguments"]["object"]["text"], "Centro Verde")

    def test_attribution_and_negation_are_visible(self):
        text = "Ada Neri ha dichiarato che Lea Blu non ha incontrato Ida Mori."
        doc = document("a", text, [("Ada Neri", "PERSON"), ("Lea Blu", "PERSON"), ("Ida Mori", "PERSON")])
        graph = build_graph([doc], context=True)
        meeting, = [n for n in graph["nodes"] if n.get("label") == "meeting"]
        self.assertEqual(meeting["assertion"], "negated")
        attributed, = [e for e in edges(graph, "FACT_ARGUMENT")
                       if e["source"] == meeting["id"] and e["role"] == "speaker"]
        self.assertEqual(attributed["evidence"][0]["mention"], {"start": 0, "end": 8})

    def test_delivery_distinguishes_recipient_from_the_object_delivered(self):
        text = "Maria Rossi ha consegnato i documenti a Luca Bianchi."
        doc = document("a", text, [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON")])
        graph = build_graph([doc], context=True)
        nodes = {node["id"]: node for node in graph["nodes"]}
        args = {e["role"]: nodes[e["target"]]["name"] for e in edges(graph, "FACT_ARGUMENT")}
        self.assertEqual(args, {"subject": "Maria Rossi", "item": "i documenti", "object": "Luca Bianchi"})

    def test_model_can_add_an_action_and_reversed_proposal_is_rejected(self):
        text = "Maria Rossi ha accompagnato Luca Bianchi a Roma."
        doc = document("a", text, [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON"), ("Roma", "LOCATION")])
        passage = {"text": text, "start": 0, "end": len(text), "core_start": 0,
                   "core_end": len(text), "context_mentions": context_mentions(doc)}
        proposed = {"evidence": text, "predicate": "ha accompagnato",
                    "arguments": {"subject": "Maria Rossi", "object": "Luca Bianchi", "place": "Roma"}}
        reversed_proposal = {**proposed, "arguments": {"subject": "Luca Bianchi", "object": "Maria Rossi"}}
        parsed = parse_context_review(json.dumps({"relations": [], "facts": [proposed, reversed_proposal]}), passage)
        self.assertEqual(parsed["status"], "partial")
        fact, = parsed["facts"]
        self.assertEqual(fact["kind"], "action")

    def test_numbered_actions_keep_supported_spans_and_reject_unknown_choices(self):
        text = "Maria Rossi ha accompagnato Luca Bianchi a Roma."
        doc = document("a", text, [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON"), ("Roma", "LOCATION")])
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        response = '{"relations":[],"facts":[1,1,2]}]'
        parsed = parse_context_review(response, passage)
        self.assertEqual(parsed["status"], "partial")
        fact, = parsed["facts"]
        self.assertEqual(fact["arguments"]["subject"]["text"], "Maria Rossi")
        self.assertEqual(fact["arguments"]["object"]["text"], "Luca Bianchi")
        self.assertEqual(text[fact["predicate_start"]:fact["predicate_end"]], "ha accompagnato")
        self.assertEqual(parse_context_review(response + " testo", passage)["status"], "invalid_response")

    def test_binary_decision_distinguishes_a_rejected_action_from_a_supported_one(self):
        text = "La polizia ha denunciato un 30enne per furto."
        doc = document("a", text)
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text),
                   "context_mentions": context_mentions(doc)}
        denied = parse_context_review('{"role_response":null,"action_responses":["0"]}', passage)
        accepted = parse_context_review('{"role_response":null,"action_responses":["1"]}', passage)
        stale = parse_context_review('{"role_response":null,"action_responses":["1"]}', passage,
                                     allow_positional=False)
        missing = parse_context_review('{"role_response":null,"action_responses":[]}', passage)
        self.assertEqual((denied["status"], denied["facts"]), ("ok", []))
        fact, = accepted["facts"]
        self.assertEqual(fact["arguments"]["subject"]["text"], "La polizia")
        self.assertEqual(fact["arguments"]["object"]["text"], "un 30enne")
        self.assertEqual(missing["status"], "invalid_response")
        self.assertEqual(stale.get("facts", []), [])

    def test_age_description_ends_before_the_reason_for_the_report(self):
        doc = document("a", "La polizia ha denunciato un 30enne per furto.")
        mentions = context_mentions(doc)
        self.assertIn("un 30enne", [mention["text"] for mention in mentions])

    def test_case_membership_and_similar_titles_have_separate_outputs(self):
        documents = [document("a", "Verbale del caso Delta."), document("b", "Resoconto del caso Delta."),
                     document("c", "Indagine a Roma nel 2023.", title="Arresto a Roma"),
                     document("d", "Indagine a Roma nel 2024.", title="Arresto a Roma")]
        config = {"references": [{"type": "CASE", "id": "delta", "name": "Caso Delta",
                                  "anchors": ["caso Delta"], "documents": ["a", "b"]}]}
        graph = build_graph(documents, context=True, link_config=config)
        self.assertEqual({e["source"] for e in edges(graph, "REFERS_TO_CASE")}, {"document:a", "document:b"})
        candidate, = graph["link_candidates"]
        self.assertEqual(candidate["documents"], ["c", "d"])
        report = score_context(graph, {"document_pairs": [{"kind": "case", "documents": ["a", "b"], "linked": True},
                                                         {"kind": "case", "documents": ["c", "d"], "linked": False}]})
        self.assertEqual(report["document_links"]["by_kind"]["case"]["tp"], 1)
        self.assertEqual(report["document_links"]["by_kind"]["case"]["tn"], 1)

    def test_shared_identifier_respects_fictitious_and_source_scopes(self):
        value = "RSSMRA00A00H501X"
        text = f"Maria Rossi, codice fiscale {value}."
        names = [("Maria Rossi", "PERSON"), (value, "TAX_CODE")]
        docs = [document("a", text, names), document("b", text, names),
                document("c", text, names, synthetic=True)]
        graph = build_graph(docs, context=True)
        identity, = [n for n in graph["nodes"] if n["type"] == "IDENTITY"]
        self.assertEqual(identity["data_scope"], "source")
        self.assertEqual({e["evidence"][0]["document_id"] for e in edges(graph, "REFERS_TO_IDENTITY")}, {"a", "b"})

    def test_docket_number_requires_matching_jurisdiction(self):
        docs = [document("a", "Tribunale di Reggio Emilia, procedimento n. 123/2023 R.G.N.R."),
                document("b", "Procura della Repubblica di Reggio Emilia n. 123/2023 R.G.N.R."),
                document("c", "Tribunale di Reggio Calabria, procedimento n. 123/2023 R.G.N.R.")]
        graph = build_graph(docs, context=True)
        self.assertEqual({e["source"] for e in edges(graph, "REFERS_TO_CASE")}, {"document:a", "document:b"})

    def test_selection_uses_broader_cues_and_retains_original_spans(self):
        text = "Maria Rossi ha accompagnato Luca Bianchi a Roma."
        doc = document("a", text, [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON")])
        irrelevant = document("b", "Maria Rossi ha accompagnato.", [("Maria Rossi", "PERSON")])
        planned, eligible = select_context_passages([doc, irrelevant], 800, 160, 100)
        self.assertEqual(eligible, 1)
        self.assertEqual(planned[0]["text"], text)
        self.assertEqual([e["text"] for e in planned[0]["context_mentions"]], ["Maria Rossi", "Luca Bianchi"])

    def test_saved_context_review_replays_as_a_new_fact(self):
        text = "Maria Rossi ha accompagnato Luca Bianchi a Roma."
        doc = document("a", text, [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON"), ("Roma", "LOCATION")])
        def reviewed(passage, *_):
            candidate, = reviewable_facts(passage)
            response = json.dumps({"role_response": None, "fact_responses": [
                {"reference": _fact_reference(candidate), "answer": "1"},
            ]})
            parsed = parse_context_review(response, passage)
            attempt = {"status": parsed["status"], "raw_response": response,
                       "accepted_count": len(parsed["facts"])}
            return {**attempt, "attempts": [attempt]}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            ner, plan, reviews = (root / name for name in ("ner.json", "plan.json", "reviews.json"))
            ner.write_text(json.dumps({"run": {"stage": "hybrid"}, "documents": [doc]}), encoding="utf-8")
            prepared = prepare(ner, None, 100, context=True)
            plan.write_text(json.dumps(prepared), encoding="utf-8")
            config = Path(__file__).resolve().parents[1] / "configs/hybrid.json"
            with patch("scripts.graph.load_model", return_value=(None, None, {})), patch(
                "scripts.graph.review_relations", side_effect=reviewed,
            ):
                result = run_review(plan, config)
            reviews.write_text(json.dumps(result), encoding="utf-8")
            graph = build_saved_graph(ner, reviews, revalidate_reviews=False)
            self.assertEqual(graph["run"]["profile"], "context")
            fact, = [node for node in graph["nodes"] if node["type"] == "FACT"]
            self.assertEqual(fact["name"], "ha accompagnato")
            self.assertEqual(edges(graph, "STATED_IN")[0]["evidence"][0]["method"], "qwen")

    def test_review_checkpoint_stops_initial_failure_and_keeps_completed_rows(self):
        text = "Maria Rossi ha accompagnato Luca Bianchi."
        mentions = [("Maria Rossi", "PERSON"), ("Luca Bianchi", "PERSON")]
        docs = [document("a", text, mentions), document("b", text, mentions)]
        empty_review = {"status": "ok", "raw_response": '{"role_response":null,"action_responses":["0"]}',
                        "attempts": [], "accepted_count": 0}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ner, plan, checkpoint = (root / name for name in ("ner.json", "plan.json", "partial.json"))
            ner.write_text(json.dumps({"run": {"stage": "hybrid"}, "documents": docs}), encoding="utf-8")
            plan.write_text(json.dumps(prepare(ner, None, 2, context=True)), encoding="utf-8")
            config = Path(__file__).resolve().parents[1] / "configs/hybrid.json"
            with patch("scripts.graph.load_model", return_value=(None, None, {})), patch(
                "scripts.graph.review_relations", return_value=empty_review,
            ) as review:
                with self.assertRaisesRegex(RuntimeError, "first 1 passages"):
                    run_review(plan, config, checkpoint_path=checkpoint, check_first=1)
                self.assertEqual(review.call_count, 1)
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                self.assertEqual(saved["run"]["statuses"], {"ok": 1})
                result = run_review(plan, config, checkpoint_path=checkpoint)
                self.assertEqual(review.call_count, 2)
                self.assertEqual(result["run"]["statuses"], {"ok": 2})


if __name__ == "__main__":
    unittest.main()
