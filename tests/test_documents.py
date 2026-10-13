import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from ownllm.data import documents


def archive_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


def test_text_encoding_and_output_exclusion(tmp_path):
    (tmp_path / "a.TXT").write_bytes("Hello world".encode("utf-16"))
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "ignore.txt").write_text("Generated output")
    assert list(documents.iter_documents(tmp_path, exclude=output)) == [(str(tmp_path / "a.TXT"), "Hello world")]


def test_zip_nested_files_and_traversal_names_stay_in_memory(tmp_path):
    nested = archive_bytes([("chapter.txt", "Inner chapter")])
    source = tmp_path / "books.zip"
    source.write_bytes(archive_bytes([("../escape.txt", "Safe text"), ("nested.zip", nested), ("image.png", b"unsupported")]))
    results = list(documents.iter_documents(source))
    assert [text for _, text in results] == ["Safe text", "Inner chapter"]
    assert not (tmp_path.parent / "escape.txt").exists()
    assert list(tmp_path.iterdir()) == [source]


def test_bad_document_does_not_discard_good_documents(tmp_path, caplog):
    (tmp_path / "a.zip").write_bytes(b"not a ZIP")
    (tmp_path / "b.txt").write_text("Valid document")
    assert [text for _, text in documents.iter_documents(tmp_path)] == ["Valid document"]
    assert "Skipping" in caplog.text


def test_archive_budget_and_nesting_are_enforced(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(documents, "MAX_ARCHIVE_BYTES", 5)
    source = tmp_path / "books.zip"
    source.write_bytes(archive_bytes([("one.txt", "1234"), ("two.txt", "5678")]))
    assert [text for _, text in documents.iter_documents(source)] == ["1234"]
    assert "size limit" in caplog.text
    monkeypatch.setattr(documents, "MAX_ARCHIVE_BYTES", 10000)
    monkeypatch.setattr(documents, "MAX_DEPTH", 1)
    source.write_bytes(archive_bytes([("nested.zip", archive_bytes([("text.txt", "nested")]))]))
    assert list(documents.iter_documents(source)) == []
    assert "nesting limit" in caplog.text


def test_epub_spine_order_and_html_text(tmp_path):
    source = tmp_path / "book.epub"
    source.write_bytes(archive_bytes([
        ("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>'),
        ("OPS/book.opf", '<package><manifest><item id="b" href="b.xhtml" media-type="application/xhtml+xml"/><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="a"/><itemref idref="b"/></spine></package>'),
        ("OPS/b.xhtml", "<p>Second chapter</p>"),
        ("OPS/a.xhtml", "<style>hidden style</style><p>First &amp; best chapter</p><script>hidden script</script>"),
    ]))
    text = list(documents.iter_documents(source))[0][1]
    assert text.index("First & best") < text.index("Second")
    assert "hidden" not in text


def test_pdf_text_and_scanned_pdf_warning(tmp_path, caplog):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Readable PDF text) Tj ET")
    page[NameObject("/Contents")] = stream
    source = tmp_path / "text.pdf"
    with source.open("wb") as handle:
        writer.write(handle)
    assert "Readable PDF text" in list(documents.iter_documents(source))[0][1]
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with source.open("wb") as handle:
        writer.write(handle)
    assert list(documents.iter_documents(source)) == []
    assert "OCR" in caplog.text


def test_missing_source_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input does not exist"):
        list(documents.iter_documents(tmp_path / "missing"))


def test_docx_extracts_paragraphs_and_table_text(tmp_path):
    source = tmp_path/'paper.docx'
    source.write_bytes(archive_bytes([('word/document.xml','<w:document xmlns:w="urn:word"><w:body><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table value</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')]))
    assert list(documents.iter_documents(source))[0][1] == 'First paragraph\nTable value'


def test_prepare_cli_builds_tokenizer_and_shards_from_archive(tmp_path):
    source = tmp_path / "documents.zip"
    source.write_bytes(archive_bytes([("first.txt", "A useful training document. " * 100), ("second.text", "Another training document. " * 100)]))
    output = tmp_path / "prepared"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/prepare_tokens.py"), str(source), str(output), "--vocab", "256"], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (output / "tokenizer/tokenizer.json").exists()
    shard = output / "shards/tokens_00000.bin"
    assert shard.exists()
    assert json.loads(shard.with_suffix(".bin.json").read_text())["tokens"] > 0
    assert "'files': 2" in result.stdout
    assert source.exists()
