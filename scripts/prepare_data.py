"""Prepare document text from REDIT TSV files and ArchivioAntimafia PDFs.

Text for extraction and existing REDIT annotations are saved separately.
Raw files are read only; PDF text is not corrected or processed with OCR.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_SEPARATOR = "\n\n"
MIN_PAGE_CHARACTERS = 80


def project_path(value: str) -> Path:
    """Resolve a path from the repository root unless it is already absolute."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def source_info(path: Path, raw_dir: Path, source: str) -> dict:
    """Identify a source file by collection, relative path and content hash.

    source_file is relative to raw_dir, so it remains usable after moving the
    dataset between the local checkout and Colab.
    """
    return {
        "id": f"{source}:{path.name}",
        "document": path.name,
        "source": source,
        "source_file": path.relative_to(raw_dir).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read_redit(path: Path, raw_dir: Path) -> tuple[dict, dict]:
    """Read one WebAnno TSV 3.3 document from the REDIT release.

    Returns a text record and a separate token-level annotation record. Gaps
    between tokens are filled with spaces to retain source character offsets.
    Annotation fields keep the original WebAnno encoding and are not a new gold
    standard. Files without a relation layer are supported.

    Raises ValueError for an unsupported schema, missing tokens or inconsistent
    offsets, rather than silently producing an incomplete document.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if "#FORMAT=WebAnno TSV 3.3" not in lines:
        raise ValueError(f"Unsupported TSV format: {path.name}")
    headers = [line for line in lines if line.startswith("#T_")]
    relation_header = "#T_RL=custom.Relation|label|BT_custom.Span"
    if "#T_SP=custom.Span|label" not in headers or any(
        header not in {"#T_SP=custom.Span|label", relation_header} for header in headers
    ):
        raise ValueError(f"Unexpected annotation columns: {path.name}")
    has_relations = relation_header in headers

    tokens = []
    identifiers = set()
    previous_end = 0
    for number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < (6 if has_relations else 4):
            raise ValueError(f"Incomplete TSV row in {path.name}, line {number}")
        offsets = re.fullmatch(r"(\d+)-(\d+)", columns[1])
        if not offsets:
            raise ValueError(f"Invalid token offsets in {path.name}, line {number}")
        start, end = map(int, offsets.groups())
        # WebAnno escapes reserved characters; annotation fields remain untouched.
        text = re.sub(r"\\([\\|_\[\]*])", r"\1", columns[2])
        if start < previous_end or end <= start or len(text) != end - start:
            raise ValueError(f"Inconsistent token offsets in {path.name}, line {number}")
        if columns[0] in identifiers:
            raise ValueError(f"Duplicate token id in {path.name}, line {number}")
        identifiers.add(columns[0])
        previous_end = end
        tokens.append({
            "id": columns[0], "start": start, "end": end, "text": text,
            "entity_annotation": columns[3],
            "relation_labels": columns[4] if has_relations else None,
            "relation_targets": columns[5] if has_relations else None,
        })
    if not tokens:
        raise ValueError(f"No tokens found in {path.name}")

    # Joining tokens with a single space could change the annotated offsets.
    characters = [" "] * tokens[-1]["end"]
    for token in tokens:
        characters[token["start"]:token["end"]] = token["text"]
    document = source_info(path, raw_dir, "redit")
    document["text"] = "".join(characters)
    document["token_count"] = len(tokens)
    annotations = {
        "id": document["id"],
        "source_sha256": document["source_sha256"],
        "status": "source_annotations_not_reviewed",
        "headers": headers,
        "tokens": tokens,
    }
    return document, annotations


def read_pdf(path: Path, raw_dir: Path) -> dict:
    """Return one complete PDF text with source metadata and page boundaries.

    Page offsets refer to the joined document, with start inclusive and end
    exclusive. Separators fall outside page spans. low_text flags pages with
    little extracted text; it does not detect all OCR or layout errors.
    """
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDF is not supported: {path.name}")
    if not reader.pages:
        raise ValueError(f"No pages found in {path.name}")

    parts = []
    pages = []
    position = 0
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "page": number, "start": position, "end": position + len(text),
            "low_text": len(text.strip()) < MIN_PAGE_CHARACTERS,
        })
        parts.append(text)
        position += len(text) + len(PAGE_SEPARATOR)

    document = source_info(path, raw_dir, "archivio_antimafia")
    document["text"] = PAGE_SEPARATOR.join(parts)
    document["pages"] = pages
    return document


def main() -> None:
    """Prepare both collections and write text, annotations and a summary report.

    All sources are parsed before writing. Existing outputs require --overwrite;
    the raw directory is never used as the output destination.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        raw_dir = project_path(args.raw_dir)
        output_dir = project_path(args.output_dir)
        tsv_files = sorted((raw_dir / "data-tsv").glob("*.tsv"))
        pdf_files = sorted(raw_dir.glob("*.pdf"))
        if not tsv_files or not pdf_files:
            raise ValueError("Expected data-tsv/*.tsv and PDF documents in the raw directory.")
        output_names = ["documents.json", "redit_annotations.json", "preparation_report.json"]
        output_paths = [output_dir / name for name in output_names]
        for path in output_paths:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {path.name}. Use --overwrite to replace it.")
            if path == raw_dir or raw_dir in path.parents:
                raise ValueError("Prepared outputs must be stored outside the raw directory.")

        documents = []
        annotations = []
        for path in tsv_files:
            document, annotation = read_redit(path, raw_dir)
            documents.append(document)
            annotations.append(annotation)
        print(f"Read {len(tsv_files)} REDIT documents.")
        for path in pdf_files:
            document = read_pdf(path, raw_dir)
            documents.append(document)
            print(f"Read {path.name}: {len(document['pages'])} pages.")

        # Counts describe preparation coverage, not NER performance or text quality.
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_directory": str(raw_dir),
            "pypdf_version": version("pypdf"),
            "redit_documents": len(tsv_files),
            "redit_tokens": sum(d.get("token_count", 0) for d in documents),
            "pdf_documents": len(pdf_files),
            "pdf_pages": sum(len(d.get("pages", [])) for d in documents),
            "total_documents": len(documents),
            "empty_documents": [d["id"] for d in documents if not d["text"].strip()],
            "low_text_pages": [
                {"document": d["id"], "page": page["page"]}
                for d in documents for page in d.get("pages", []) if page["low_text"]
            ],
            "notes": [
                "REDIT text is reconstructed from token offsets; gaps are filled with spaces.",
                "Source annotations are retained separately and are not a validated NER reference.",
                "PDF pages are joined with two newlines; no OCR or text correction is applied.",
                "Low-text detection does not identify every OCR or extraction error.",
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, content in zip(output_paths, [documents, annotations, report]):
            path.write_text(
                json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n",
            )
        print(f"Saved {len(documents)} complete documents to {output_dir}")
    except (OSError, ValueError, PdfReadError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
