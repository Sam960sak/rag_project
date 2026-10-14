import os
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from docx import Document


class DocumentParser:
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx'}

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f'Unsupported file type: {ext}')
        if ext == '.txt':
            return self._parse_txt(path)
        elif ext == '.pdf':
            return self._parse_pdf(path)
        elif ext == '.docx':
            return self._parse_docx(path)
        return {'text': '', 'metadata': {}}

    def _parse_txt(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        metadata = {'source': str(path), 'file_name': path.name, 'file_type': 'txt', 'pages': 1}
        return {'text': text, 'metadata': metadata}

    def _parse_pdf(self, path: Path) -> Dict[str, Any]:
        text = ''
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += '\n\n--- PAGE ' + str(page_num + 1) + ' ---\n\n' + page_text
                except Exception:
                    continue
        metadata = {'source': str(path), 'file_name': path.name, 'file_type': 'pdf', 'pages': num_pages}
        return {'text': text, 'metadata': metadata}

    def _parse_docx(self, path: Path) -> Dict[str, Any]:
        doc = Document(path)
        text_parts = []
        for para in doc.paragraphs:
            if para.text:
                text_parts.append(para.text)
        text = '\n\n'.join(text_parts)
        metadata = {'source': str(path), 'file_name': path.name, 'file_type': 'docx', 'pages': 1}
        return {'text': text, 'metadata': metadata}
