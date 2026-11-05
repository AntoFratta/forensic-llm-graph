"""Prepare REDIT forms, ArchivioAntimafia PDFs, DICE articles and synthetic banking texts.

Save the text for extraction separately from the available source annotations.
Raw files are read only; PDF text is not corrected or processed with OCR.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
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


def read_dice(path: Path, raw_dir: Path) -> tuple[list, dict]:
    """Read crime-news articles, excluding empty text and exact text duplicates.

    UTF-8 is attempted first, followed by Windows-1252 for the original CSV.
    Whitespace is normalized only for duplicate comparison; retained text stays
    unchanged. Similar articles about the same event are not removed by this rule.
    The report records exclusions and the retained counterpart of each duplicate.
    """
    content = path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = content.decode("cp1252")
        encoding = "cp1252"
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not {"id", "text"}.issubset(reader.fieldnames or []):
        raise ValueError("DICE CSV must contain id and text columns.")

    source_hash = hashlib.sha256(content).hexdigest()
    documents = []
    exclusions = []
    identifiers = set()
    seen_text = {}
    rows_read = 0
    for row in reader:
        rows_read += 1
        identifier = (row.get("id") or "").strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"Missing or duplicate DICE id at CSV record {rows_read}.")
        identifiers.add(identifier)
        original_text = row.get("text") or ""
        if not original_text.strip():
            exclusions.append({"id": identifier, "reason": "empty_text"})
            continue
        comparison_text = re.sub(r"\s+", " ", original_text).strip()
        text_hash = hashlib.sha256(comparison_text.encode("utf-8")).hexdigest()
        if text_hash in seen_text:
            exclusions.append({"id": identifier, "reason": "exact_text_duplicate",
                               "retained_id": seen_text[text_hash]})
            continue
        seen_text[text_hash] = identifier
        documents.append({
            "id": f"dice:{identifier}", "document": f"article_{identifier}",
            "source": "dice", "source_file": path.relative_to(raw_dir).as_posix(),
            "source_sha256": source_hash, "text": original_text,
            "article_id": identifier, "url": row.get("url"),
            "title": row.get("title"), "crime_category": row.get("newspaper_tag"),
            "publication_date": row.get("publication_date"), "event_date": row.get("event_date"),
        })
    return documents, {
        "source_file": path.relative_to(raw_dir).as_posix(),
        "source_sha256": source_hash, "encoding": encoding,
        "records_read": rows_read, "records_retained": len(documents),
        "empty_text": sum(e["reason"] == "empty_text" for e in exclusions),
        "exact_text_duplicates": sum(e["reason"] == "exact_text_duplicate" for e in exclusions),
        "excluded_records": exclusions,
        "near_duplicate_detection": "not_performed",
    }


def read_synthetic_banking(path: Path, raw_dir: Path) -> tuple[list, dict]:
    """Read the documented Italian synthetic subset and separate its annotations.

    Text is preserved. Banking annotations are mapped to BANK_ACCOUNT, while
    original labels are retained for inspection. This is a separate diagnostic
    collection, not a validated reference for all NER categories.
    """
    artifact = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(artifact, dict):
        raise ValueError("Synthetic subset must contain records and provenance metadata.")
    provenance = artifact.get("provenance")
    records = artifact.get("records")
    if not isinstance(provenance, dict) or not isinstance(records, list):
        raise ValueError("Synthetic subset requires a provenance object and a record array.")
    if provenance.get("synthetic") is not True or not all(
        isinstance(provenance.get(key), str) and provenance[key] for key in ("revision", "url")
    ):
        raise ValueError("Synthetic data require explicit origin and dataset revision.")
    if not records or provenance.get("selected_records") != len(records):
        raise ValueError("Synthetic record count does not match its selection metadata.")

    documents = []
    annotations = []
    identifiers = set()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in records:
        if not isinstance(row, dict) or not {"source_text", "source_line", "template_id", "entities"}.issubset(row):
            raise ValueError("Incomplete synthetic source record.")
        if not isinstance(row["source_line"], int) or row["source_line"] < 1:
            raise ValueError("Synthetic source line must be a positive integer.")
        if not isinstance(row["entities"], list):
            raise ValueError("Synthetic entities must be an array.")
        text = row["source_text"]
        if row.get("language") != "it" or not isinstance(text, str) or not text.strip():
            raise ValueError("Expected non-empty Italian text in the synthetic subset.")
        identifier = f"rizzo_synthetic:{provenance['revision']}:{row['source_line']}"
        if identifier in identifiers:
            raise ValueError("Duplicate synthetic source record.")
        identifiers.add(identifier)
        bank_entities = []
        for entity in row["entities"]:
            if not isinstance(entity, dict) or not {"start", "end", "label", "value"}.issubset(entity):
                raise ValueError(f"Incomplete synthetic annotation: {identifier}")
            start, end = entity["start"], entity["end"]
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text) or text[start:end] != entity["value"]:
                raise ValueError(f"Synthetic annotation does not match text: {identifier}")
            if entity["label"] in {"IBAN", "CONTO"}:
                bank_entities.append({
                    "text": entity["value"], "start": start, "end": end,
                    "label": "BANK_ACCOUNT",
                    "subtype": "iban" if entity["label"] == "IBAN" else "account_number",
                })
        if not bank_entities:
            raise ValueError("Synthetic banking subset contains a record without account identifiers.")
        documents.append({
            "id": identifier, "document": f"synthetic_record_{row['source_line']}",
            "source": "rizzo_synthetic", "synthetic": True,
            "language": "it", "template_id": str(row["template_id"]),
            "source_file": path.relative_to(raw_dir).as_posix(),
            "source_sha256": source_hash, "source_revision": provenance["revision"],
            "source_url": provenance["url"], "source_line": row["source_line"], "text": text,
        })
        annotations.append({
            "id": identifier, "status": "source_annotations_not_reviewed",
            "bank_entities": bank_entities, "source_entities": row["entities"],
        })
    return documents, {
        "scope": "BANK_ACCOUNT only; separate synthetic diagnostic collection",
        "selection_provenance": provenance, "documents": annotations,
    }


def main() -> None:
    """Prepare available collections and save text, annotations and a report.

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
        output_names = ["documents.json", "redit_annotations.json",
                        "synthetic_annotations.json", "preparation_report.json"]
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

        dice_report = None
        dice_path = raw_dir / "italian_crime_news.csv"
        if dice_path.exists():
            dice_documents, dice_report = read_dice(dice_path, raw_dir)
            documents.extend(dice_documents)
            print(f"Retained {len(dice_documents)} DICE articles.")
        else:
            print("DICE CSV not present; this collection was not included.")

        synthetic_annotations = {"scope": "not_included", "documents": []}
        synthetic_path = raw_dir / "rizzo_banking_subset.json"
        if synthetic_path.exists():
            synthetic_documents, synthetic_annotations = read_synthetic_banking(synthetic_path, raw_dir)
            documents.extend(synthetic_documents)
            print(f"Read {len(synthetic_documents)} synthetic diagnostic documents.")
        else:
            print("Synthetic banking subset not present; this collection was not included.")

        # Counts describe preparation coverage, not NER performance or text quality.
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_directory": str(raw_dir),
            "pypdf_version": version("pypdf"),
            "redit_documents": len(tsv_files),
            "redit_tokens": sum(d.get("token_count", 0) for d in documents),
            "pdf_documents": len(pdf_files),
            "pdf_pages": sum(len(d.get("pages", [])) for d in documents),
            "dice": dice_report,
            "synthetic_banking": synthetic_annotations.get("selection_provenance"),
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
                "DICE annotations are not used; only empty and exact duplicate texts are excluded.",
                "Similar news and related cases should stay in the same split when building a reference.",
                "The synthetic banking subset retains its source and template identifiers.",
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, content in zip(output_paths, [documents, annotations, synthetic_annotations, report]):
            path.write_text(
                json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n",
            )
        print(f"Saved {len(documents)} complete documents to {output_dir}")
    except (OSError, ValueError, csv.Error, PdfReadError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
