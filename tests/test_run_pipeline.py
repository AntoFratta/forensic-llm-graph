"""Offline integration checks for batching and experiment output integrity."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.run_pipeline import main


class PipelineTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1] / "outputs"
        workspace.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_pipeline_", dir=workspace)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.documents = [
            {"id": "one", "text": "Email: anna@example.org", "source": "example",
             "annotations": [{"label": "SHOULD_NOT_BE_USED"}]},
            {"id": "two", "text": "Data 5/15/21", "source": "example"},
        ]
        self.input_path = self.directory / "documents.json"
        self.output_path = self.directory / "result.json"
        self.config_path = self.directory / "config.json"
        self.input_path.write_text(json.dumps(self.documents), encoding="utf-8")
        self.config = {
            "input_path": str(self.input_path), "output_path": str(self.output_path),
            "use_spacy": True, "spacy_model": "it_core_news_lg",
            "spacy_model_version": "3.8.0", "spacy_batch_size": 8,
        }
        self.nlp = Mock(meta={"name": "core_news_lg", "version": "3.8.0"},
                        pipe_names=["morphologizer", "ner"])
        self.nlp.pipe.side_effect = lambda texts, batch_size: (
            SimpleNamespace(text=text, ents=[], has_annotation=lambda name: True)
            for text in texts
        )

    def run_script(self, *args):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with patch("sys.argv", ["run_pipeline.py", "--config", str(self.config_path), *args]), \
                patch("scripts.run_pipeline.load_nlp", return_value=self.nlp), \
                patch("scripts.run_pipeline.version", return_value="3.8.11"), \
                patch("scripts.run_pipeline.git_info", return_value={}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main()

    def test_batch_processing_preserves_order_offsets_and_provenance(self):
        self.run_script()
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.nlp.assert_not_called()
        self.assertEqual(self.nlp.pipe.call_args.kwargs, {"batch_size": 8})
        self.assertEqual([row["id"] for row in result["documents"]], ["one", "two"])
        self.assertEqual(result["run"]["spacy_pipeline"], ["morphologizer", "ner"])
        self.assertEqual(result["run"]["spacy_batch_size"], 8)
        self.assertEqual(result["run"]["document_count"], 2)
        self.assertEqual(result["run"]["by_source"]["example"]["candidates"], 2)
        for row in result["documents"]:
            self.assertNotIn("annotations", row)
            for entity in row["entities"]:
                self.assertEqual(row["text"][entity["start"]:entity["end"]], entity["text"])

    def test_rules_only_does_not_invoke_nlp(self):
        self.run_script("--rules-only")
        self.nlp.pipe.assert_not_called()
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["run"]["mode"], "rules_only")
        self.assertIsNone(result["run"]["spacy_batch_size"])

    def test_incomplete_nlp_stream_does_not_write_results(self):
        self.nlp.pipe.side_effect = lambda texts, batch_size: iter([])
        with self.assertRaises(SystemExit):
            self.run_script()
        self.assertFalse(self.output_path.exists())

    def test_batch_size_rejects_invalid_values(self):
        for value in (0, -1, True, 1.5, "8"):
            with self.subTest(value=value):
                self.config["spacy_batch_size"] = value
                with self.assertRaises(SystemExit):
                    self.run_script()
                self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
