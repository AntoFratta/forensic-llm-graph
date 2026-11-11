import json
import unittest

from src.relation_graph import build_graph
from src.relation_review import (load_review_relations, parse_relation_review, select_relation_passages,
                                 text_hash)


class RelationReviewTests(unittest.TestCase):
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
