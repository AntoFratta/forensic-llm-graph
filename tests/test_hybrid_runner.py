"""Offline checks for stage ordering, resume protection and result provenance."""

from contextlib import redirect_stderr, redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_hybrid import main, read_checkpoint
from scripts.evaluate import main as evaluate_main


class HybridRunnerTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        (root / "outputs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="test_hybrid_", dir=root / "outputs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.input_path = self.directory / "input.json"
        self.output_path = self.directory / "output.json"
        self.config_path = self.directory / "config.json"
        self.input_path.write_text(json.dumps({"run": {"stage": "first_pass"}, "documents": [
            {"id": "one", "source": "example", "text": "Un testo italiano per la prova.", "entities": []}
        ]}), encoding="utf-8")
        self.config = json.loads((root / "configs/hybrid.json").read_text(encoding="utf-8"))
        self.config.update(input_path=str(self.input_path), output_path=str(self.output_path))
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def execute(self, *arguments):
        with patch("sys.argv", ["run_hybrid.py", "--config", str(self.config_path), *arguments]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main()

    def test_prepare_only_needs_no_gpu_or_model(self):
        with patch("scripts.run_hybrid.require_cuda") as cuda, patch("scripts.run_hybrid.load_model") as loader:
            self.execute("--prepare-only", "--smoke-test")
            cuda.assert_not_called()
            loader.assert_not_called()
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["run"]["scope"], "smoke_test")
        self.assertEqual(result["run"]["passage_count"], 1)

    def test_models_run_in_separate_stages_and_checkpoint_can_rebuild_result(self):
        events = []
        cuda = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda index: SimpleNamespace(total_memory=16_000_000_000),
            get_device_name=lambda index: "test GPU", max_memory_allocated=lambda: 1234))

        def load(settings, quantized):
            events.append("qwen" if quantized else "minerva")
            return object(), object(), {"model": settings["model"]}

        def score(passages, settings, model, tokenizer):
            yield {p["id"]: {"status": "ok", "perplexity": 20.0, "mean_nll": 3.0} for p in passages}

        with patch("scripts.run_hybrid.require_cuda", return_value=cuda), \
                patch("scripts.run_hybrid.load_model", side_effect=load) as loader, \
                patch("scripts.run_hybrid.release_cuda_memory", side_effect=lambda: events.append("release")), \
                patch("scripts.run_hybrid.perplexity_batches", side_effect=score), \
                patch("scripts.run_hybrid.review_passage", return_value={"status": "ok", "entities": []}):
            self.execute("--smoke-test")
            self.assertEqual(events, ["minerva", "release", "qwen", "release"])
            result = json.loads(self.output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["run"]["review_status_counts"], {"ok": 1})
            self.output_path.unlink()
            loader.reset_mock()
            self.execute("--smoke-test", "--resume")
            loader.assert_not_called()
            resumed = json.loads(self.output_path.read_text(encoding="utf-8"))
            self.assertTrue(resumed["run"]["resumed"])

    def test_extended_review_requires_a_threshold_before_loading_models(self):
        self.config["routing"]["perplexity_threshold"] = None
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with patch("scripts.run_hybrid.load_model") as loader, self.assertRaises(SystemExit):
            self.execute()
        loader.assert_not_called()

    def test_development_reference_selects_cores_without_sending_gold_to_review(self):
        text = "Un testo italiano per la prova."
        unit = {"id": f"one@0:{len(text)}", "reference_id": "D001", "document_id": "one",
                "source": "example", "split": "development", "start": 0, "end": len(text),
                "core_start": 0, "core_end": len(text), "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "issues": [],
                "annotation_status": "reviewed", "human_review": {"reviewer": "test human"},
                "entities": [{"start": 3, "end": 8, "text": "testo", "label": "PERSON"}]}
        reference = {"schema_version": 1, "split": "development", "units": [unit],
                     "core_chars": self.config["passages"]["core_chars"],
                     "context_chars": self.config["passages"]["context_chars"]}
        path = self.directory / "reference.json"
        path.write_text(json.dumps(reference), encoding="utf-8")
        cuda = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda index: SimpleNamespace(total_memory=16_000_000_000),
            get_device_name=lambda index: "test GPU", max_memory_allocated=lambda: 1234))

        def review(passage, settings, model, tokenizer):
            self.assertEqual(passage["candidates"], [])
            self.assertNotIn("entities", passage)
            self.assertNotIn("human_review", passage)
            return {"status": "ok", "entities": []}

        with patch("scripts.run_hybrid.require_cuda", return_value=cuda), \
                patch("scripts.run_hybrid.load_model", return_value=(object(), object(), {})), \
                patch("scripts.run_hybrid.release_cuda_memory"), \
                patch("scripts.run_hybrid.perplexity_batches", return_value=[{
                    unit["id"]: {"status": "ok", "perplexity": 20.0, "mean_nll": 3.0}}]), \
                patch("scripts.run_hybrid.review_passage", side_effect=review):
            self.execute("--reference", str(path), "--development-all")
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["run"]["scope"], "development_all_review")
        self.assertEqual(result["run"]["reviewed_passage_count"], 1)
        self.assertIsNone(result["routing"])
        self.assertEqual(result["documents"][0]["entities"], [])
        metrics_path = self.directory / "metrics.json"
        calibrated_path = self.directory / "calibrated.json"
        with patch("sys.argv", ["evaluate.py", "--reference", str(path),
                                 "--baseline", str(self.input_path), "--hybrid", str(self.output_path),
                                 "--calibrate-routing", "--calibrated-config", str(calibrated_path),
                                 "--output", str(metrics_path)]), redirect_stdout(io.StringIO()):
            evaluate_main()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        calibrated = json.loads(calibrated_path.read_text(encoding="utf-8"))
        self.assertFalse(metrics["calibration"]["threshold_quality_varies_on_development"])
        self.assertEqual(calibrated["routing"]["perplexity_threshold"], 20.0)
        self.assertIn("fallback", calibrated["calibration"]["threshold_selection"])
        self.assertEqual(calibrated["calibration"]["split"], "development")

    def test_checkpoint_rejects_a_different_experiment_and_repairs_only_its_last_line(self):
        path = self.directory / "checkpoint.jsonl"
        header = json.dumps({"kind": "header", "fingerprint": "expected"}).encode() + b"\n"
        valid = b'{"kind":"review","passage_id":"one"}\n'
        path.write_bytes(header + valid + b'{"unfinished":')
        with self.assertRaises(ValueError):
            read_checkpoint(path, "other")
        self.assertTrue(path.read_bytes().endswith(b'{"unfinished":'))
        records = read_checkpoint(path, "expected")
        self.assertEqual(len(records), 1)
        self.assertEqual(path.read_bytes(), header + valid)

    def test_fixed_routing_comparison_reviews_the_union_but_merges_only_the_primary_selection(self):
        text = "Anna presenta una breve dichiarazione."
        entity = {"start": 0, "end": 4, "text": "Anna", "label": "PERSON", "source": "spacy:ner",
                  "needs_review": True, "review_reasons": ["role_context_unverified"]}
        documents = [{"id": name, "source": "example", "text": text, "entities": [entity]}
                     for name in ("a", "b", "c")]
        self.input_path.write_text(json.dumps({"run": {"stage": "first_pass"}, "documents": documents}), encoding="utf-8")
        units = [{"id": f"{d['id']}@0:{len(text)}", "reference_id": d["id"], "document_id": d["id"],
                  "source": "example", "split": "development", "start": 0, "end": len(text),
                  "core_start": 0, "core_end": len(text), "text": text,
                  "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "issues": [],
                  "annotation_status": "reviewed", "human_review": {"reviewer": "test human"},
                  "entities": [entity]} for d in documents]
        reference = {"schema_version": 1, "split": "development", "units": units,
                     "core_chars": self.config["passages"]["core_chars"],
                     "context_chars": self.config["passages"]["context_chars"]}
        path = self.directory / "reference.json"
        path.write_text(json.dumps(reference), encoding="utf-8")
        self.config["routing"]["perplexity_threshold"] = 25
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        cuda = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda index: SimpleNamespace(total_memory=16_000_000_000),
            get_device_name=lambda index: "test GPU", max_memory_allocated=lambda: 1234))
        scores = {u["id"]: {"status": "ok", "perplexity": value, "mean_nll": value / 10}
                  for u, value in zip(units, (10, 30, 20))}
        with patch("scripts.run_hybrid.require_cuda", return_value=cuda), \
                patch("scripts.run_hybrid.load_model", return_value=(object(), object(), {})), \
                patch("scripts.run_hybrid.release_cuda_memory"), \
                patch("scripts.run_hybrid.perplexity_batches", return_value=[scores]), \
                patch("scripts.run_hybrid.review_passage", return_value={"status": "ok", "entities": []}) as reviewer:
            self.execute("--reference", str(path), "--compare-routing")
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(reviewer.call_count, 2)
        self.assertEqual(result["run"]["primary_selected_passage_count"], 1)
        self.assertEqual(result["run"]["reviewed_passage_count"], 2)
        after = {d["id"]: len(d["entities"]) for d in result["documents"]}
        self.assertEqual(after, {"a": 1, "b": 0, "c": 1})
        metrics_path = self.directory / "comparison.json"
        with patch("sys.argv", ["evaluate.py", "--reference", str(path), "--baseline", str(self.input_path),
                                 "--hybrid", str(self.output_path), "--output", str(metrics_path)]), redirect_stdout(io.StringIO()):
            evaluate_main()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("without_perplexity", metrics["metrics"])
        self.assertEqual(metrics["fixed_routing_comparison"]["selected_passage_ids"], [units[0]["id"]])


if __name__ == "__main__":
    unittest.main()
