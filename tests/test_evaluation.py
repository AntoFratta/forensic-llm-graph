"""Check exact-span metrics, reference integrity and development-only calibration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate import calibrate
from src.evaluation import load_reference, reference_passages, score_reference
from src.llm_filter import review_messages
from src.passages import build_passages, select_passages


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.text = "Ada visita Roma."
        self.unit = {"id": "one@0:16", "reference_id": "D001", "document_id": "one",
                     "source": "example", "split": "development", "start": 0, "end": len(self.text),
                     "core_start": 0, "core_end": len(self.text), "text": self.text,
                     "text_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
                     "annotation_status": "reviewed", "human_review": {"reviewer": "test human"},
                     "issues": [], "entities": [
                         {"start": 0, "end": 3, "text": "Ada", "label": "PERSON"},
                         {"start": 11, "end": 15, "text": "Roma", "label": "LOCATION"}]}
        self.unit["id"] = f"one@0:{len(self.text)}"
        self.reference = {"schema_version": 1, "split": "development", "core_chars": 100,
                          "context_chars": 0, "units": [self.unit]}
        self.document = {"id": "one", "source": "example", "text": self.text, "entities": []}

    def read(self, reference):
        root = Path(__file__).resolve().parents[1] / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            return load_reference(path)[0]

    def test_wrong_label_and_boundary_are_errors_and_duplicate_predictions_count_once(self):
        self.document["entities"] = [self.unit["entities"][0], self.unit["entities"][0],
                                     {"start": 11, "end": 15, "text": "Roma", "label": "PERSON"}]
        result = score_reference(self.reference, [self.document])
        micro = result["groups"]["source_documents"]["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (1, 1, 1))
        self.assertEqual(micro["f1"], 0.5)
        self.document["entities"][2] = {"start": 11, "end": 14, "text": "Rom", "label": "LOCATION"}
        self.assertEqual(score_reference(self.reference, [self.document])["groups"]["source_documents"]["micro"], micro)

    def test_context_mentions_are_not_scored_in_the_wrong_core_and_synthetic_is_separate(self):
        self.unit["core_end"] = 10
        self.unit["entities"] = self.unit["entities"][:1]
        self.document["entities"] = [{"start": 11, "end": 15, "text": "Roma", "label": "LOCATION"}]
        synthetic = deepcopy(self.unit)
        synthetic.update(id="two@0:10", document_id="two", reference_id="D002", source="rizzo_synthetic")
        self.reference["units"].append(synthetic)
        second = {**self.document, "id": "two", "source": "rizzo_synthetic", "entities": synthetic["entities"]}
        result = score_reference(self.reference, [self.document, second])["groups"]
        self.assertEqual(result["source_documents"]["micro"]["fp"], 0)
        self.assertEqual(result["source_documents"]["micro"]["fn"], 1)
        self.assertEqual(result["synthetic"]["micro"]["tp"], 1)
        self.assertNotIn("combined", result)
        self.assertIsNone(result["source_documents"]["by_category"]["EMAIL"]["recall"])

    def test_unreviewed_changed_or_overlapping_reference_cannot_be_used(self):
        self.read(self.reference)
        for change in ("unreviewed", "changed_text", "overlapping", "unfrozen_test"):
            with self.subTest(change=change):
                reference = deepcopy(self.reference)
                if change == "unreviewed":
                    reference["units"][0]["human_review"] = None
                elif change == "changed_text":
                    reference["units"][0]["text"] = self.text.upper()
                elif change == "overlapping":
                    reference["units"].append({**self.unit, "id": "overlap"})
                else:
                    reference["split"] = reference["units"][0]["split"] = "test"
                with self.assertRaises(ValueError):
                    self.read(reference)

    def test_gold_does_not_become_candidates_or_enter_the_prompt(self):
        passages = build_passages([self.document], core_chars=100, context_chars=0)
        selected = reference_passages(self.reference, [self.document], passages,
                                      {"core_chars": 100, "context_chars": 0})
        self.assertEqual(selected[0]["candidates"], [])
        message = json.loads(review_messages(selected[0])[1]["content"])
        self.assertEqual(message, {"text": self.text})
        with self.assertRaises(ValueError):
            reference_passages(self.reference, [self.document], passages,
                               {"core_chars": 200, "context_chars": 0})

    def test_without_perplexity_neither_threshold_nor_loss_can_change_selection(self):
        passages = [{"id": name, "text": "Breve testo", "candidates": [
            {"review_reasons": ["role_context_unverified"]}]} for name in ("a", "b")]
        settings = {"perplexity_threshold": None, "max_review_fraction": 0.5, "max_review_passages": 1}
        scores = {"a": {"status": "ok", "perplexity": 2, "mean_nll": 0.7},
                  "b": {"status": "ok", "perplexity": 1000, "mean_nll": 6.9}}
        result = select_passages(passages, scores, settings, use_perplexity=False)
        self.assertTrue(result["decisions"]["a"]["selected"])
        self.assertFalse(result["decisions"]["b"]["selected"])
        self.assertEqual(result["threshold_mode"], "disabled")

    def test_calibration_refuses_test_references_before_reading_predictions(self):
        self.reference["split"] = "test"
        with self.assertRaises(ValueError):
            calibrate(self.reference, [], [], {})


if __name__ == "__main__":
    unittest.main()
