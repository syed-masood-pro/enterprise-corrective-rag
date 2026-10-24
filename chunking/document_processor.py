"""
Universal document processor built on LangChain loaders and splitters.

Supports any file type the user may upload:
  Text / Docs  : .txt  .md  .pdf  .docx
  Code         : .py  .js  .ts  .jsx  .tsx  .java  .cpp  .c  .go  .rs
                 .rb  .cs  .scala  .swift  .kt  .php  .lua  .pl
  Shell        : .sh  .bash  .zsh  .ps1  .bat  .cmd
  Data/Config  : .sql  .json  .yaml  .yml  .xml  .csv  .toml  .ini  .conf
  Web          : .html  .htm
  Markup       : .rst  .tex  .latex
  Other        : .r  .proto  .sol  .ex  .exs  .hs

Strategy:
  - Code files    → language-aware RecursiveCharacterTextSplitter
  - SQL / Shell   → separator-tuned splitter
  - Markdown/RST  → header-aware splitter
  - Everything else → generic recursive splitter
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from chunking.base import Chunk
from config.settings import Settings

logger = logging.getLogger(__name__)

# ── Extension → LangChain Language enum ──────────────────────────────────────
_CODE_LANGUAGE_MAP: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".ts":    Language.JS,
    ".jsx":   Language.JS,
    ".tsx":   Language.JS,
    ".java":  Language.JAVA,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".cxx":   Language.CPP,
    ".c":     Language.C,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".rb":    Language.RUBY,
    ".cs":    Language.CSHARP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".kt":    Language.KOTLIN,
    ".php":   Language.PHP,
    ".sol":   Language.SOL,
    ".lua":   Language.LUA,
    ".pl":    Language.PERL,
    ".ex":    Language.ELIXIR,
    ".exs":   Language.ELIXIR,
    ".hs":    Language.HASKELL,
    ".proto": Language.PROTO,
}

# Shell-like extensions (no LangChain Language entry)
_SHELL_EXTENSIONS: set[str] = {".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"}

# Data/config extensions
_DATA_EXTENSIONS: set[str] = {".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".env"}

# HTML
_HTML_EXTENSIONS: set[str] = {".html", ".htm"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """16-char SHA-256 prefix — used to build deterministic chunk IDs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _classify_file(ext: str) -> str:
    """Return a human-readable file-type label stored in chunk metadata."""
    if ext in _CODE_LANGUAGE_MAP:
        return f"code/{_CODE_LANGUAGE_MAP[ext].value}"
    if ext == ".sql":
        return "code/sql"
    if ext in _SHELL_EXTENSIONS:
        return "code/shell"
    if ext == ".pdf":
        return "document/pdf"
    if ext in (".docx", ".doc"):
        return "document/word"
    if ext == ".csv":
        return "data/csv"
    if ext in _DATA_EXTENSIONS:
        return f"data/{ext.lstrip('.')}"
    if ext == ".xml":
        return "data/xml"
    if ext in (".md", ".markdown"):
        return "text/markdown"
    if ext == ".rst":
        return "text/rst"
    if ext in (".html", ".htm"):
        return "text/html"
    return "text/plain"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_documents(file_path: Path) -> list[Document]:
    """Pick the right LangChain loader for *file_path* and return raw Documents."""
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            # PyMuPDFLoader automatically splits by pages
            loader = PyMuPDFLoader(str(file_path))

        elif ext in (".docx", ".doc"):
            loader = Docx2txtLoader(str(file_path))

        elif ext in _HTML_EXTENSIONS:
            loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")

        elif ext == ".csv":
            loader = CSVLoader(
                file_path=str(file_path),
                encoding="utf-8",
            )

        else:
            # TextLoader handles: .txt .md .py .js .ts .sql .yaml .json
            # .sh .xml .toml .conf .ini .rst .tex .r — anything text-based
            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
                autodetect_encoding=True,
            )

        docs = loader.load()
        logger.info("Loaded %d raw documents from '%s'", len(docs), file_path.name)
        return docs

    except Exception as exc:
        logger.error("Failed to load '%s': %s", file_path.name, exc)
        raise RuntimeError(f"Cannot read '{file_path.name}': {exc}") from exc


# ── Splitters ─────────────────────────────────────────────────────────────────

def _get_splitter(ext: str, settings: Settings) -> RecursiveCharacterTextSplitter:
    """Return the most appropriate splitter for the given file extension."""

    # ── Language-aware code splitter ──────────────────────────────────────────
    if ext in _CODE_LANGUAGE_MAP:
        return RecursiveCharacterTextSplitter.from_language(
            language=_CODE_LANGUAGE_MAP[ext],
            chunk_size=settings.code_chunk_size,
            chunk_overlap=settings.code_chunk_overlap,
        )

    # ── SQL ───────────────────────────────────────────────────────────────────
    if ext == ".sql":
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", ";\n", ";", "\n-- ", "\n", " "],
            chunk_size=settings.code_chunk_size,
            chunk_overlap=settings.code_chunk_overlap,
        )

    # ── Shell scripts ─────────────────────────────────────────────────────────
    if ext in _SHELL_EXTENSIONS:
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\nfunction ", "\nalias ", "\n", " "],
            chunk_size=settings.code_chunk_size,
            chunk_overlap=settings.code_chunk_overlap,
        )

    # ── Markdown / RST ────────────────────────────────────────────────────────
    if ext in (".md", ".markdown"):
        return RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    if ext == ".rst":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.RST,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    if ext in (".tex", ".latex"):
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.LATEX,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── HTML ──────────────────────────────────────────────────────────────────
    if ext in _HTML_EXTENSIONS:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── YAML / TOML / config ──────────────────────────────────────────────────
    if ext in _DATA_EXTENSIONS:
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── Default generic splitter ──────────────────────────────────────────────
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def process_document(file_path: str | Path, settings: Settings) -> list[Chunk]:
    """
    Load, split, and convert any supported file into a list of Chunk objects.

    Args:
        file_path: Absolute path to the uploaded file.
        settings:  Application settings (chunk sizes, overlaps, etc.).

    Returns:
        Non-empty list of Chunk objects with enriched metadata.

    Raises:
        RuntimeError: If the file cannot be loaded or yields no content.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    doc_name = path.name
    file_type = _classify_file(ext)

    logger.info("Processing '%s' (type=%s)", doc_name, file_type)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    lc_docs: list[Document] = _load_documents(path)
    if not lc_docs:
        raise RuntimeError(f"No content extracted from '{doc_name}'.")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    splitter = _get_splitter(ext, settings)
    split_docs: list[Document] = splitter.split_documents(lc_docs)
    logger.info("Splitter produced %d chunks from %d documents", len(split_docs), len(lc_docs))

    if not split_docs:
        raise RuntimeError(f"Splitting produced no chunks from '{doc_name}'.")

    # ── 3. Build Chunk objects ────────────────────────────────────────────────
    chunks: list[Chunk] = []
    empty_count = 0
    error_count = 0
    for idx, doc in enumerate(split_docs):
        try:
            content = doc.page_content.strip() if doc.page_content else ""
            if not content:
                empty_count += 1
                continue

            # Filter metadata to only Pinecone-safe types (str, int, float, bool)
            raw_meta: dict = dict(doc.metadata) if doc.metadata else {}
            meta: dict = {}
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is not None:
                    meta[k] = str(v)

            chunk_id = f"{doc_name}_{idx}_{_content_hash(content)}"
            meta.update(
                {
                    "doc_name": doc_name,
                    "file_type": file_type,
                    "extension": ext,
                    "chunk_index": idx,
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "chunk_id": chunk_id,
                }
            )

            chunks.append(Chunk(content=content, metadata=meta, chunk_id=chunk_id))
        except Exception as exc:
            error_count += 1
            if error_count <= 3:
                logger.error(
                    "Chunk %d/%d failed: %s (content_len=%d)",
                    idx, len(split_docs), exc,
                    len(doc.page_content) if doc.page_content else 0,
                )

    if empty_count > 0:
        logger.info("Skipped %d empty chunks from '%s'", empty_count, doc_name)
    if error_count > 0:
        logger.warning("Failed to build %d/%d chunks from '%s'", error_count, len(split_docs), doc_name)

    if not chunks:
        raise RuntimeError(f"All chunks from '{doc_name}' were empty after stripping.")

    logger.info("Produced %d chunks from '%s'", len(chunks), doc_name)
    return chunks
