"""Offline checks for stage ordering, resume protection and result provenance."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_hybrid import main, read_checkpoint


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
        with patch("scripts.run_hybrid.load_model") as loader, self.assertRaises(SystemExit):
            self.execute()
        loader.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
