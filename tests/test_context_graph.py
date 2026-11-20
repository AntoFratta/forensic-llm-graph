import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.graph import build_saved_graph, prepare, run_review
from src.context_facts import context_mentions
from src.graph_queries import inspect_graph, score_context
from src.relation_graph import build_graph
from src.relation_review import parse_context_review, select_context_passages


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
        response = json.dumps({"relations": [], "facts": [1]})

        def reviewed(passage, *_):
            status = parse_context_review(response, passage)["status"]
            attempt = {"status": status, "raw_response": response}
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
        failure = {"status": "invalid_response", "raw_response": "invalid", "attempts": []}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ner, plan, checkpoint = (root / name for name in ("ner.json", "plan.json", "partial.json"))
            ner.write_text(json.dumps({"run": {"stage": "hybrid"}, "documents": docs}), encoding="utf-8")
            plan.write_text(json.dumps(prepare(ner, None, 2, context=True)), encoding="utf-8")
            config = Path(__file__).resolve().parents[1] / "configs/hybrid.json"
            with patch("scripts.graph.load_model", return_value=(None, None, {})), patch(
                "scripts.graph.review_relations", return_value=failure,
            ) as review:
                with self.assertRaisesRegex(RuntimeError, "first 1 passages"):
                    run_review(plan, config, checkpoint_path=checkpoint, check_first=1)
                self.assertEqual(review.call_count, 1)
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                self.assertEqual(saved["run"]["statuses"], {"invalid_response": 1})
                result = run_review(plan, config, checkpoint_path=checkpoint)
                self.assertEqual(review.call_count, 2)
                self.assertEqual(result["run"]["statuses"], {"invalid_response": 2})


if __name__ == "__main__":
    unittest.main()
