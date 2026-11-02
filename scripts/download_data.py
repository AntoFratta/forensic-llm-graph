"""Download the fixed Italian corpus and rebuild its synthetic banking subset.

Existing files are reused only when their hashes match the source configuration.
This script prepares data/raw; text extraction is handled by prepare_data.py.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_hash(path: Path) -> str:
    """Hash a file in blocks, avoiding a second in-memory copy of the CSV."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reusable(path: Path, expected_hash: str, overwrite: bool) -> bool:
    """Reuse matching files and protect unexpected local versions."""
    if not path.exists():
        return False
    if file_hash(path) == expected_hash:
        return True
    if not overwrite:
        raise ValueError(f"Different local content: {path}. Use --overwrite to replace it.")
    return False


def target_path(raw_dir: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Source filenames must not contain directory components.")
    path = (raw_dir / filename).resolve()
    if path.parent != raw_dir:
        raise ValueError("Source destination must stay inside the raw directory.")
    return path


def download_file(source: dict, raw_dir: Path, overwrite: bool) -> Path:
    """Download through a temporary file and verify content before replacing it."""
    path = target_path(raw_dir, source["filename"])
    if reusable(path, source["sha256"], overwrite):
        print(f"Verified existing file: {path.name}")
        return path

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=raw_dir, prefix="download_", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            # Default HTTPS certificate validation remains enabled.
            with urlopen(source["url"], timeout=60) as response:
                while block := response.read(1024 * 1024):
                    stream.write(block)
        if file_hash(temporary) != source["sha256"]:
            raise ValueError(f"Source content differs from the configured version: {path.name}")
        temporary.replace(path)
        print(f"Downloaded and verified: {path.name}")
        return path
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_redit(source: dict, raw_dir: Path, overwrite: bool) -> None:
    """Restore only the expected TSV files; ignore archive metadata directories."""
    directory = (raw_dir / "data-tsv").resolve()
    if directory.parent != raw_dir:
        raise ValueError("REDIT directory must stay inside the raw directory.")
    expected = source["tsv_files"]
    extra = {p.name for p in directory.glob("*.tsv")} - set(expected)
    if extra:
        raise ValueError("Unexpected TSV files found in data-tsv; review the selected corpus.")
    missing = [name for name, digest in expected.items()
               if not reusable(target_path(directory, name), digest, overwrite)]
    if not missing:
        print(f"Verified {len(expected)} existing REDIT documents.")
        return

    archive_path = download_file(source, raw_dir, overwrite)
    directory.mkdir(exist_ok=True)
    with ZipFile(archive_path) as archive:
        for name in missing:
            content = archive.read(f"data-tsv/{name}")
            if hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError(f"Unexpected REDIT document: {name}")
            # Explicit member names avoid extracting arbitrary archive paths.
            target_path(directory, name).write_bytes(content)
    print(f"Prepared {len(expected)} REDIT documents.")


def prepare_synthetic(source: dict, raw_dir: Path, overwrite: bool) -> None:
    """Reproduce the bounded, template-balanced selection from a fixed revision.

    Only the configured source prefix is streamed. Original rows and their line
    hashes are retained; selection never uses predictions from our extractor.
    """
    path = target_path(raw_dir, source["filename"])
    if reusable(path, source["sha256"], overwrite):
        print(f"Verified existing synthetic subset: {path.name}")
        return
    provenance = source["source"]
    prefix_hash = hashlib.sha256()
    per_template = Counter()
    bank_mentions = Counter()
    records = []
    with urlopen(provenance["url"], timeout=60) as response:
        for number in range(1, provenance["records_scanned"] + 1):
            line = response.readline()
            if not line:
                raise ValueError("Synthetic source ended before the configured prefix.")
            prefix_hash.update(line)
            row = json.loads(line)
            entities = row.get("entities", [])
            bank = [e for e in entities if e["label"] in {"IBAN", "CONTO"}]
            if row.get("language") != "it" or not bank:
                continue
            template = str(row["template_id"])
            if per_template[template] >= source["max_per_template"]:
                continue
            for entity in entities:
                if row["source_text"][entity["start"]:entity["end"]] != entity["value"]:
                    raise ValueError(f"Synthetic offsets mismatch at source line {number}.")
            per_template[template] += 1
            bank_mentions.update(e["label"] for e in bank)
            records.append({"source_line": number,
                            "source_line_sha256": hashlib.sha256(line).hexdigest(), **row})
    if prefix_hash.hexdigest() != provenance["scanned_prefix_sha256"]:
        raise ValueError("Synthetic source prefix differs from the selected version.")
    if (len(records) != provenance["selected_records"]
            or len(per_template) != provenance["templates"]
            or dict(bank_mentions) != provenance["bank_mentions"]):
        raise ValueError("Synthetic selection differs from its recorded counts.")
    content = json.dumps({"provenance": provenance, "records": records},
                         ensure_ascii=False, indent=2).encode("utf-8")
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("Reconstructed synthetic subset differs from the recorded artifact.")
    path.write_bytes(content)
    print(f"Rebuilt {len(records)} synthetic documents from a fixed source revision.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_sources.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(project_path(args.config).read_text(encoding="utf-8-sig"))
        raw_dir = project_path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        prepare_redit(config["redit"], raw_dir, args.overwrite)
        for source in config["files"]:
            download_file(source, raw_dir, args.overwrite)
        prepare_synthetic(config["synthetic_banking"], raw_dir, args.overwrite)
        print(f"Corpus download complete: {raw_dir}")
    except (OSError, ValueError, KeyError, URLError, BadZipFile) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
