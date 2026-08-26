from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_profile: Literal["mvp", "prod"] = "mvp"
    log_level: str = "INFO"

    llm_backend: Literal["stub", "openai_compatible"] = "stub"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 120.0

    embedding_backend: Literal["stub", "sentence_transformers"] = "stub"
    embedding_model: str = "darklethelong/vnlegal-lal"
    embedding_fallback_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    reranker_backend: Literal["stub", "flag_embedding"] = "stub"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    qdrant_mode: Literal["memory", "server"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "vn_legal_chunks"

    graph_backend: Literal["memory", "neo4j"] = "memory"
    graph_snapshot_path: Path = Path("data/processed/legal_graph.json")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    retrieval_top_k_dense: int = 20
    retrieval_top_k_sparse: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 8
    max_retrieval_attempts: int = Field(2, ge=1, le=5)
    grounding_threshold: float = Field(0.6, ge=0.0, le=1.0)
    claim_support_threshold: float = Field(0.6, ge=0.0, le=1.0)

    max_chunk_chars: int = 2000
    min_chunk_chars: int = 40

    enable_run_log: bool = True
    run_log_path: Path = Path("data/processed/run_log.jsonl")

    eval_dataset_path: Path = Path("data/eval/golden_set.jsonl")
    eval_report_dir: Path = Path("data/processed/eval")

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    bm25_index_path: Path = Path("data/processed/bm25_index.pkl")

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def abs_raw_data_dir(self) -> Path:
        return self.resolve(self.raw_data_dir)

    @property
    def abs_processed_data_dir(self) -> Path:
        return self.resolve(self.processed_data_dir)

    @property
    def abs_bm25_index_path(self) -> Path:
        return self.resolve(self.bm25_index_path)

    @property
    def abs_graph_snapshot_path(self) -> Path:
        return self.resolve(self.graph_snapshot_path)

    @property
    def abs_run_log_path(self) -> Path:
        return self.resolve(self.run_log_path)

    @property
    def abs_eval_dataset_path(self) -> Path:
        return self.resolve(self.eval_dataset_path)

    @property
    def abs_eval_report_dir(self) -> Path:
        return self.resolve(self.eval_report_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
