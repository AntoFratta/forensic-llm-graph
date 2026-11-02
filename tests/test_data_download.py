"""Offline checks for downloads, file protection and subset reproduction."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts.download_data import download_file, prepare_redit, prepare_synthetic


class DownloadTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1] / "outputs"
        workspace.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_download_", dir=workspace)
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name).resolve()

    def test_matching_file_is_reused_without_network(self):
        content = b"existing content"
        path = self.raw / "sample.csv"
        path.write_bytes(content)
        source = {"filename": path.name, "url": "https://example.invalid/file",
                  "sha256": hashlib.sha256(content).hexdigest()}
        with patch("scripts.download_data.urlopen") as network:
            self.assertEqual(download_file(source, self.raw, False), path)
            network.assert_not_called()

    def test_changed_local_file_is_protected(self):
        path = self.raw / "sample.csv"
        path.write_bytes(b"local changes")
        source = {"filename": path.name, "url": "https://example.invalid/file", "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            download_file(source, self.raw, False)
        self.assertEqual(path.read_bytes(), b"local changes")

    def test_bad_download_does_not_replace_local_file(self):
        path = self.raw / "sample.csv"
        path.write_bytes(b"previous version")
        source = {"filename": path.name, "url": "https://example.invalid/file", "sha256": "0" * 64}
        with patch("scripts.download_data.urlopen", return_value=io.BytesIO(b"wrong content")):
            with self.assertRaises(ValueError):
                download_file(source, self.raw, True)
        self.assertEqual(path.read_bytes(), b"previous version")
        self.assertEqual(list(self.raw.glob("*.part")), [])

    def test_archive_extracts_only_expected_members(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("data-tsv/one.tsv", b"original TSV")
            archive.writestr("__MACOSX/extra", b"metadata")
            archive.writestr("../outside.txt", b"unexpected")
        content = buffer.getvalue()
        source = {"filename": "data-tsv.zip", "url": "https://example.invalid/archive",
                  "sha256": hashlib.sha256(content).hexdigest(),
                  "tsv_files": {"one.tsv": hashlib.sha256(b"original TSV").hexdigest()}}
        with patch("scripts.download_data.urlopen", return_value=io.BytesIO(content)):
            prepare_redit(source, self.raw, False)
        self.assertEqual((self.raw / "data-tsv/one.tsv").read_bytes(), b"original TSV")
        self.assertEqual({p.name for p in self.raw.iterdir()}, {"data-tsv.zip", "data-tsv"})

    def test_synthetic_selection_is_reproducible_and_balanced(self):
        rows = [
            {"language": language, "template_id": template, "source_text": value,
             "entities": [{"label": label, "start": 0, "end": len(value), "value": value}]}
            for language, template, value, label in [
                ("it", 1, "bank-one", "IBAN"), ("it", 1, "bank-two", "IBAN"),
                ("it", 2, "bank-three", "IBAN"), ("en", 3, "bank-four", "IBAN"),
                ("it", 4, "tax-code", "CF"),
            ]
        ]
        lines = [(json.dumps(row) + "\n").encode() for row in rows]
        provenance = {"url": "https://example.invalid/source", "records_scanned": 5,
                      "scanned_prefix_sha256": hashlib.sha256(b"".join(lines)).hexdigest(),
                      "selected_records": 2, "templates": 2, "bank_mentions": {"IBAN": 2}}
        expected = {"provenance": provenance, "records": [
            {"source_line": i + 1, "source_line_sha256": hashlib.sha256(lines[i]).hexdigest(), **rows[i]}
            for i in (0, 2)
        ]}
        content = json.dumps(expected, ensure_ascii=False, indent=2).encode()
        source = {"filename": "subset.json", "source": provenance, "max_per_template": 1,
                  "sha256": hashlib.sha256(content).hexdigest()}
        with patch("scripts.download_data.urlopen", return_value=io.BytesIO(b"".join(lines))):
            prepare_synthetic(source, self.raw, False)
        self.assertEqual((self.raw / "subset.json").read_bytes(), content)
        self.assertEqual(Counter(r["template_id"] for r in expected["records"]), {1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()
