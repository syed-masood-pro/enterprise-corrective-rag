"""
Application settings — driven entirely by environment variables / .env file.
No secrets are ever hardcoded here.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    api_keys: str = Field(..., description="Comma-separated valid API keys")
    cors_allow_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated allowed CORS origins. Use '*' only for dev.",
    )

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field(default="llama-3.1-8b-instant")
    groq_max_tokens: int = Field(default=1024)
    groq_temperature: float = Field(default=0.1)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = Field(default="all-MiniLM-L6-v2")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index: str = Field(default="rag-docs")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    code_chunk_size: int = Field(default=768)
    code_chunk_overlap: int = Field(default=128)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=10, description="Candidates per retriever before RRF")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    rrf_top_k: int = Field(default=20, description="Candidates kept after RRF (fed to reranker)")
    final_top_k: int = Field(default=5, description="Final results after cross-encoder reranking")

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_enabled: bool = Field(default=True)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=10)

    # ── Valkey (semantic cache backend) ───────────────────────────────────────
    valkey_host: str = Field(default="localhost")
    valkey_port: int = Field(default=6379)
    valkey_db: int = Field(default=0)
    valkey_password: str = Field(default="")

    # ── Semantic cache ────────────────────────────────────────────────────────
    semantic_cache_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for a semantic cache hit (0–1).",
    )
    cache_ttl_seconds: int = Field(default=300)

    # ── CRAG (Corrective RAG) ──────────────────────────────────────
    crag_enabled: bool = Field(default=True, description="Enable CRAG evaluator + refinement + web fallback")
    crag_grade_ttl_seconds: int = Field(default=3600, description="TTL for cached CRAG grades")
    crag_web_ttl_seconds: int = Field(default=1800, description="TTL for cached web-search results")
    crag_max_web_attempts: int = Field(default=2, description="Max web-search attempts (with query rewrite) before giving up")
    crag_grade_heuristic_enabled: bool = Field(
        default=True,
        description="Skip the CRAG grader LLM call when retrieval is confidently strong (both retrievers agree).",
    )
    crag_grade_min_docs: int = Field(
        default=3,
        description="Minimum docs required for the CRAG grade heuristic to short-circuit to 'Correct'.",
    )
    web_search_provider: str = Field(default="stub", description="Web search provider name (stub|tavily|serpapi|...)")
    web_search_top_k: int = Field(default=5, description="Top-K docs returned by the web search fallback")
    tavily_api_key: str = Field(default="", description="Tavily API key (required if web_search_provider=tavily)")

    # ── File upload ───────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=50)
    docs_temp_dir: str = Field(default="./temp_uploads")

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def valkey_storage_uri(self) -> str:
        """Redis-compatible URI for slowapi shared rate-limit storage.

        Valkey speaks the Redis protocol, so the redis:// scheme works.
        Includes the password when configured.
        """
        auth = f":{self.valkey_password}@" if self.valkey_password else ""
        return f"redis://{auth}{self.valkey_host}:{self.valkey_port}/{self.valkey_db}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator("groq_temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("groq_temperature must be between 0.0 and 2.0")
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — reads .env once."""
    return Settings()

