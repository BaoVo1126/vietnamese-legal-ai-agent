from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import get_settings
from ..llm.prompts import DISCLAIMER
from ..logging_config import get_logger, setup_logging
from .deps import get_agent_service
from .routers import admin, ask, health, metrics

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Khởi động API (profile=%s)...", settings.app_profile)
    service = get_agent_service()
    logger.info("Sẵn sàng: %d chunks trong vector store.", service.vector_store.count())
    yield
    service.close()
    logger.info("Đã đóng kết nối.")


app = FastAPI(
    title="Trợ lý Hỏi-Đáp Pháp Luật Việt Nam",
    version="0.1.0",
    description=(
        "AI Agent pipeline hỏi-đáp pháp luật Việt Nam theo nguyên tắc "
        "grounded-or-refuse và version-aware.\n\n" + DISCLAIMER
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(metrics.router)
app.include_router(admin.router)


@app.get("/", tags=["health"], summary="Thông tin dịch vụ")
def root() -> dict:
    return {
        "service": "legal-agent-vn",
        "version": "0.1.0",
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }
