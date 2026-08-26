FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

ARG INSTALL_PROFILE=core

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    if [ "$INSTALL_PROFILE" = "ml" ]; then \
        pip install -r requirements.txt; \
    else \
        pip install fastapi "uvicorn[standard]" pydantic pydantic-settings langgraph \
            langchain-core qdrant-client rank-bm25 openai python-dotenv underthesea neo4j streamlit pandas; \
    fi

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY data/raw/ ./data/raw/
COPY data/eval/ ./data/eval/

RUN useradd --create-home --uid 1000 appuser && \
    mkdir -p /app/data/processed && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8080/live || exit 1

CMD ["uvicorn", "legal_agent.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
