"""
Ingest route — POST /api/v1/ingest

Accepts a multipart file upload, validates it, saves it temporarily,
runs the full ingest pipeline, then cleans up the temp file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from api.schemas.response import IngestResponse
from config.settings import get_settings
from pipeline.ingest_pipeline import run_ingest
from utils.bm25_sync import publish_bm25_reload

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingest"])

# All file extensions the processor can handle
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    # Text / documents
    ".txt", ".md", ".markdown", ".pdf", ".docx", ".doc", ".rst", ".log",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".cc", ".cxx", ".c",
    ".go", ".rs", ".rb", ".cs", ".scala",
    ".swift", ".kt", ".php", ".lua", ".pl",
    ".sol", ".ex", ".exs", ".hs", ".proto",
    ".r", ".tex", ".latex",
    # Shell / scripts
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    # Data / config
    ".sql", ".json", ".yaml", ".yml", ".xml",
    ".csv", ".toml", ".ini", ".conf", ".env",
    # Web
    ".html", ".htm",
})

# Magic-byte signatures for binary formats we accept.
# Text/code formats are skipped (no reliable signature) — they're plain UTF-8.
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf":  (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),          # zip-based OOXML
    ".doc":  (b"\xd0\xcf\x11\xe0",),    # OLE2 compound file
}


def _content_matches_extension(suffix: str, content: bytes) -> bool:
    """Return True if the file bytes match the claimed extension.

    Only binary formats with a known signature are checked. Text/code
    formats always pass here (validated as UTF-8 later in the pipeline).
    """
    sigs = _MAGIC_SIGNATURES.get(suffix)
    if not sigs:
        return True  # text/code — no magic-byte check
    head = content[:8]
    return any(head.startswith(sig) for sig in sigs)

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into the RAG pipeline",
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="File to ingest (any supported type)"),
) -> IngestResponse:
    """
    Upload any document — code, text, SQL, scripts, PDFs, spreadsheets, etc.
    The system will:
    1. Detect the file type automatically.
    2. Split it using the most appropriate strategy (code-aware, markdown-aware, etc.).
    3. Embed and store chunks in Pinecone.
    4. Update the BM25 keyword index.

    Subsequent `/query` calls will search across all ingested documents.
    """
    settings = get_settings()

    # ── Validate extension ────────────────────────────────────────────────────
    filename = os.path.basename(file.filename or "upload")
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # Read & validate content (streamed, size-capped)
    # Read in chunks and abort as soon as we exceed the limit — this prevents a
    # "zip/​txt bomb" or huge upload from being fully buffered into memory.
    max_bytes = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB at a time
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum allowed size: {settings.max_upload_size_mb} MB.",
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    if not _content_matches_extension(suffix,content):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File content does not match its '{suffix}' extension.",
        )

    # ── Write temp file ───────────────────────────────────────────────────────
    os.makedirs(settings.docs_temp_dir, exist_ok=True)
    temp_path = Path(settings.docs_temp_dir) / filename

    try:
        temp_path.write_bytes(content)

        result = await run_ingest(
            file_path=temp_path,
            store=request.app.state.store,
            bm25_index=request.app.state.bm25_index,
            settings=settings,
        )

    except RuntimeError as exc:
        logger.error("Ingestion failed for '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    finally:
        # Always delete the temp file — even on error
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    # Notify peer replicas to rebuild their in-memory BM25 index
    # This replica already rebuilt in-process during run_ingest; the event tells
    # every OTHER replica to reload from Pinecone so keyword search stays
    # consistent across the fleet. Guarded internally — never fails the request.
    valkey_client = getattr(request.app.state, "valkey_client", None)
    instance_id = getattr(request.app.state, "instance_id", "")
    if valkey_client is not None:
        await publish_bm25_reload(valkey_client, instance_id)
    return IngestResponse(
        **result,
        message=f"Successfully ingested '{result['doc_name']}'.",
    )
