import json
import unittest

from src.relation_graph import build_graph
from src.relation_review import (load_review_relations, parse_relation_review, select_relation_passages,
                                 text_hash)


class RelationReviewTests(unittest.TestCase):
    def test_cue_only_proposal_keeps_original_offsets_for_pdf_ligatures(self):
        text = "In capo a K.K., semplice dipendente della Raﬃneria."
        passage = {"text": text, "start": 50, "core_start": 50, "core_end": 50 + len(text)}
        proposal = {
            "source": {"text": "K.K.", "label": "PERSON", "kind": "name"},
            "target": {"text": "Raffineria", "label": "COMPANY", "kind": "description"},
            "cue": 1,
        }
        result = parse_relation_review(json.dumps({"relations": [proposal]}), passage)
        relation, = result["relations"]
        target = relation["target"]
        self.assertEqual(text[target["start"] - 50:target["end"] - 50], "Raﬃneria")

    def test_cue_must_connect_a_person_to_an_in_scope_target(self):
        cases = [
            ("Lea Blu, direttrice della clinica, ha incontrato Ada Neri.", "Ada Neri", "clinica", "COMPANY"),
            ("Lea Blu, dipendente del Comune di Roma.", "Lea Blu", "Comune di Roma", "COMPANY"),
            ("Lea Blu, difensore di uno dei due imputati.", "Lea Blu", "uno dei due imputati", "COMPANY"),
            ("Lea Blu non lavora presso una farmacia.", "Lea Blu", "una farmacia", "COMPANY"),
            ("Lea Blu era titolare di una speciﬁca posizione di garanzia.",
             "Lea Blu", "una specifica posizione di garanzia", "COMPANY"),
            ("Lea Blu, titolare del servizio sharing Alfa.", "Lea Blu", "servizio sharing Alfa", "COMPANY"),
        ]
        for text, person, other, label in cases:
            with self.subTest(text=text):
                passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
                proposal = {
                    "source": {"text": person, "label": "PERSON", "kind": "name"},
                    "target": {"text": other, "label": label, "kind": "description"},
                    "cue": 1,
                }
                result = parse_relation_review(json.dumps({"relations": [proposal]}), passage)
                self.assertEqual(result["status"], "invalid_response")

    def test_qualified_legal_cue_keeps_the_client_span_and_its_number(self):
        for qualifier in ("fiducia", "ﬁducia"):
            with self.subTest(qualifier=qualifier):
                text = f"Lea Blu, difensore di {qualifier} di A.A., conosce Ada Neri, dipendente della ditta Alfa."
                passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
                proposal = {"source": {"text": "Lea Blu", "label": "PERSON", "kind": "name"}, "cue": 1,
                            "target": {"text": "A.A.", "label": "PERSON", "kind": "name"}}
                result = parse_relation_review(json.dumps({"relations": [proposal]}), passage)
                relation, = result["relations"]
                self.assertEqual(relation["target"]["start"], text.index("A.A."))
                self.assertEqual(relation["predicate_start"], text.index("difensore"))
                proposal["cue"] = 2
                self.assertEqual(parse_relation_review(json.dumps({"relations": [proposal]}), passage)["status"],
                                 "invalid_response")

    def test_two_roles_can_share_an_explicit_company(self):
        text = 'Claudio (amministratore \nunico e poi liquidatore della società "Sigimet s.r.l.")'
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
        endpoints = {
            "source": {"text": "Claudio", "label": "PERSON", "kind": "name"},
            "target": {"text": "Sigimet s.r.l.", "label": "COMPANY", "kind": "name"},
        }
        proposals = [{**endpoints, "cue": cue} for cue in (1, 2)]
        result = parse_relation_review(json.dumps({"relations": proposals}), passage)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [text[r["predicate_start"]:r["predicate_end"]] for r in result["relations"]],
            ["amministratore \nunico", "liquidatore"],
        )

    def test_corpus_review_reserves_slots_for_distinct_judicial_cases(self):
        text = "Una donna lavora presso una farmacia."
        documents = [
            {"id": name, "case_id": case, "source": source, "text": text, "entities": []}
            for name, case, source in [
                ("a1", "case-a", "archivio_antimafia"), ("a2", "case-a", "archivio_antimafia"),
                ("b", "case-b", "corte_cassazione"), ("news", "news", "dice"),
            ]
        ]
        selected, count = select_relation_passages(documents, 800, 160, 3)
        self.assertEqual(count, 4)
        self.assertEqual({p["source"] for p in selected[:2]}, {"archivio_antimafia", "corte_cassazione"})

    def test_numbered_cue_and_evidence_must_identify_the_same_occurrence(self):
        text = "Lea Blu, direttrice della clinica. Ada Neri, presidente della cooperativa."
        passage = {"text": text, "start": 100, "core_start": 100, "core_end": 100 + len(text)}
        proposal = {"source": {"text": "Ada Neri", "label": "PERSON", "kind": "name"},
                    "target": {"text": "cooperativa", "label": "COMPANY", "kind": "description"},
                    "evidence": "Ada Neri, presidente della cooperativa", "cue": 2}
        parsed = parse_relation_review(json.dumps({"relations": [proposal]}), passage)
        relation, = parsed["relations"]
        self.assertEqual(relation["predicate_start"], 100 + text.index("presidente"))
        proposal["cue"] = 1
        invalid = parse_relation_review(json.dumps({"relations": [proposal]}), passage)
        self.assertEqual(invalid["status"], "invalid_response")

    def test_revalidation_recovers_a_literal_role_from_a_reformulated_predicate(self):
        text = "Lea Blu, amministratrice della cooperativa."
        raw = json.dumps({"relations": [{
            "source": {"text": "Lea Blu", "label": "PERSON", "kind": "name"},
            "predicate": "è amministratrice di",
            "target": {"text": "cooperativa", "label": "COMPANY", "kind": "description"},
            "evidence": text,
        }]})
        document = {"id": "example", "text": text, "entities": []}
        saved_empty = '{"relations":[]}'
        review = {"run": {"stage": "relation_review", "input_sha256": "fixture"},
                  "reviews": [{"id": "review", "document_id": "example", "start": 0, "end": len(text),
                               "core_start": 0, "core_end": len(text), "text_sha256": text_hash(text),
                               "status": "ok", "raw_response": saved_empty,
                               "attempts": [{"status": "invalid_response", "raw_response": raw},
                                            {"status": "ok", "raw_response": saved_empty}]}]}
        self.assertEqual(load_review_relations(review, [document], "fixture")["example"], [])
        relation, = load_review_relations(review, [document], "fixture", revalidate=True)["example"]
        self.assertEqual(text[relation["predicate_start"]:relation["predicate_end"]], "amministratrice")

    def test_lowercase_name_requires_support_from_ner(self):
        text = "neri lavora presso una farmacia."
        passage = {"text": text, "start": 20, "core_start": 20, "core_end": 20 + len(text)}
        raw = json.dumps({"relations": [{
            "source": {"text": "neri", "label": "PERSON", "kind": "name"},
            "cue": 1,
            "target": {"text": "una farmacia", "label": "COMPANY", "kind": "description"},
            "evidence": text,
        }]})
        self.assertEqual(parse_relation_review(raw, passage)["status"], "invalid_response")
        passage["known_names"] = [{"start": 20, "end": 24, "label": "PERSON"}]
        self.assertEqual(parse_relation_review(raw, passage)["status"], "ok")

    def test_revalidation_rejects_unrelated_predicates(self):
        text = "Lea Blu, direttrice della clinica, ha incontrato Ada Neri."
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
        raw = json.dumps({"relations": [{
            "source": {"text": "Lea Blu", "label": "PERSON", "kind": "name"},
            "predicate": "ha incontrato",
            "target": {"text": "Ada Neri", "label": "PERSON", "kind": "name"},
            "evidence": text,
        }]})
        self.assertEqual(parse_relation_review(raw, passage)["status"], "invalid_response")

    def test_relation_cue_selects_text_without_ner_subjects(self):
        documents = [{"id": "example", "text": "Una donna lavora presso una farmacia.", "entities": []},
                     {"id": "other", "text": "Lea Blu e Ada Neri erano presenti.", "entities": []}]
        selected, count = select_relation_passages(documents, 800, 160, 10)
        self.assertEqual(count, 1)
        self.assertEqual([p["document_id"] for p in selected], ["example"])

    def test_grounded_model_review_adds_subjects_and_preserves_direction(self):
        text = "Il difensore di Ada Neri,\navv. Lea Blu, ha risposto."
        raw = json.dumps({"relations": [{
            "source": {"text": "Lea Blu", "label": "PERSON", "kind": "name"},
            "predicate": "difensore",
            "target": {"text": "Ada Neri", "label": "PERSON", "kind": "name"},
            "evidence": "Il difensore di Ada Neri, avv. Lea Blu",
        }]})
        document = {"id": "example", "text": text, "entities": []}
        review = {
            "run": {"stage": "relation_review", "input_sha256": "fixture"},
            "reviews": [{"id": "review-1", "document_id": "example", "start": 0, "end": len(text),
                         "core_start": 0, "core_end": len(text), "text_sha256": text_hash(text),
                         "status": "ok", "raw_response": raw}],
        }
        graph = build_graph([document], reviewed_relations=load_review_relations(review, [document], "fixture"))
        nodes = {node["id"]: node for node in graph["nodes"]}
        edge, = [e for e in graph["edges"] if e["type"] == "EXPLICIT_RELATION"]
        self.assertEqual(nodes[edge["source"]]["name"], "Lea Blu")
        self.assertEqual(nodes[edge["target"]]["name"], "Ada Neri")
        self.assertEqual(edge["evidence"][0]["method"], "qwen")
        self.assertEqual(text[edge["evidence"][0]["start"]:edge["evidence"][0]["end"]],
                         "difensore di Ada Neri,\navv. Lea Blu")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_review_relations(review, [{**document, "text": text.replace("Ada", "Eva")}], "fixture")

    def test_invalid_endpoint_or_ambiguous_evidence_is_rejected(self):
        text = "Lea Blu lavora presso una farmacia."
        proposal = {"source": {"text": "Lea Blu", "label": "PERSON", "kind": "name"},
                    "predicate": "lavora presso",
                    "target": {"text": "una farmacia", "label": "COMPANY", "kind": "description"},
                    "evidence": text}
        wrong = {**proposal, "target": {**proposal["target"], "text": "una banca"}}
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
        partial = parse_relation_review(json.dumps({"relations": [proposal, wrong]}), passage)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(len(partial["relations"]), 1)
        repeated = {**passage, "text": f"{text} {text}", "core_end": len(text) * 2 + 1}
        invalid = parse_relation_review(json.dumps({"relations": [proposal]}), repeated)
        self.assertEqual(invalid["status"], "invalid_response")

    def test_rule_and_model_evidence_for_the_same_relation_is_counted_once(self):
        text = "Una donna, dipendente di una cooperativa."
        raw = json.dumps({"relations": [{
            "source": {"text": "Una donna", "label": "PERSON", "kind": "description"},
            "predicate": "dipendente",
            "target": {"text": "una cooperativa", "label": "COMPANY", "kind": "description"},
            "evidence": text,
        }]})
        passage = {"text": text, "start": 0, "core_start": 0, "core_end": len(text)}
        review = parse_relation_review(raw, passage)
        document = {"id": "example", "text": text, "entities": []}
        graph = build_graph([document], reviewed_relations={"example": review["relations"]})
        edge, = [e for e in graph["edges"] if e["type"] == "EXPLICIT_RELATION"]
        evidence, = edge["evidence"]
        self.assertTrue(evidence["qwen_reviewed"])


if __name__ == "__main__":
    unittest.main()
