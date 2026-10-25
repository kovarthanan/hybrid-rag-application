import re
from pypdf import PdfReader
from .config import get_settings


def load_pdf_text(path: str) -> str:
    """Extract and clean text from the PDF."""
    reader = PdfReader(path)
    text = "\n".join((p.extract_text() or "") for p in reader.pages)

    # PDF extraction sometimes glues lines together, e.g. "files.5) Search".
    # Put numbered sections and bullets back on their own lines.
    text = re.sub(r"(?<=[^\d\s])(\d{1,2}\) [A-Z])", r"\n\1", text)
    text = re.sub(r"\.- ", ".\n- ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[dict]:
    """
    Line-aware chunking: never cuts a line in half, and carries the
    last few lines into the next chunk so context isn't lost at edges.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks, current, length = [], [], 0

    for line in lines:
        if length + len(line) > size and current:
            chunks.append("\n".join(current))
            # Build the overlap tail from the end of the current chunk
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            current, length = tail, tail_len
        current.append(line)
        length += len(line)

    if current:
        chunks.append("\n".join(current))

    return [{"id": i, "text": c} for i, c in enumerate(chunks)]


def build_chunks() -> list[dict]:
    s = get_settings()
    return chunk_text(load_pdf_text(s.kb_file), s.chunk_size, s.chunk_overlap)