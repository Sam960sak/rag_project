import re
from pathlib import Path
from typing import List, Dict, Any, Iterator
import PyPDF2
from docx import Document
from docx.document import Document as _DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


class UnsupportedFileTypeError(ValueError):
    pass


class UnreadableDocumentError(Exception):
    pass


def _iter_docx_blocks(parent) -> Iterator[Any]:
    """Yield paragraphs and tables of a DOCX in document order."""
    if isinstance(parent, _DocxDocument):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


class DocumentParser:
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx'}

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. Supported types: "
                f"{', '.join(sorted(self.SUPPORTED_EXTENSIONS))}.")
        if ext == '.txt':
            return self._parse_txt(path)
        if ext == '.pdf':
            return self._parse_pdf(path)
        return self._parse_docx(path)

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ''
        # Drop any page markers that may have leaked into the extracted text;
        # page numbers are kept as metadata, never embedded.
        text = re.sub(r'[-=]{0,3}\s*PAGE\s+\d+\s*[-=]{0,3}', ' ', text, flags=re.IGNORECASE)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Rejoin words hyphenated across a line break ("chunk-\ning").
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _build(pages: List[Dict[str, Any]], meta: Dict[str, Any], path: Path) -> Dict[str, Any]:
        if not pages:
            raise UnreadableDocumentError(
                f"No readable text found in '{path.name}'. If this is a scanned or "
                f"image-only PDF, run OCR (e.g. ocrmypdf) first, or upload a text-based "
                f"PDF/TXT/DOCX.")
        meta = dict(meta)
        meta['pages'] = len(pages)
        body = '\n\n'.join(p['text'] for p in pages if p['text'])
        return {'text': body, 'metadata': meta, 'pages': pages}

    def _parse_txt(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        text = self._clean_text(raw)
        pages = [{'page': 1, 'text': text}] if text else []
        return self._build(pages, {'source': str(path), 'file_name': path.name,
                                   'file_type': 'txt'}, path)

    def _parse_pdf(self, path: Path) -> Dict[str, Any]:
        pages: List[Dict[str, Any]] = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = self._clean_text(page.extract_text() or '')
                except Exception:
                    page_text = ''
                if page_text:
                    pages.append({'page': page_num, 'text': page_text})
        meta = {'source': str(path), 'file_name': path.name,
                'file_type': 'pdf', 'total_pages': total}
        return self._build(pages, meta, path)

    def _parse_docx(self, path: Path) -> Dict[str, Any]:
        doc = Document(path)
        parts: List[str] = []
        for block in _iter_docx_blocks(doc):
            if isinstance(block, Paragraph):
                if block.text and block.text.strip():
                    parts.append(block.text.strip())
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(' | '.join(cells))
        text = self._clean_text('\n\n'.join(parts))
        pages = [{'page': 1, 'text': text}] if text else []
        return self._build(pages, {'source': str(path), 'file_name': path.name,
                                   'file_type': 'docx'}, path)
