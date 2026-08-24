# 🤖 AI Agents with Hybrid RAG

**A framework-agnostic AI agent that answers questions from any document you give it — in English or Vietnamese, with citations, refusing rather than guessing — benchmarking four reasoning strategies side by side over multilingual BM25 + FAISS hybrid retrieval, with Postgres/Redis behind the chat history and an n8n orchestration layer for automation.**

![status](https://img.shields.io/badge/tests-254%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![cost](https://img.shields.io/badge/real--mode-free%20(Ollama)-success) ![infra](https://img.shields.io/badge/storage-Postgres%20%2B%20Redis-336791) ![streaming](https://img.shields.io/badge/API-SSE%20streaming-informational) ![retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20FAISS%20multilingual-informational) ![orchestration](https://img.shields.io/badge/orchestration-n8n-EA4B71)

🔗 **Live demo:** https://ai-agent-hybrid-rag.onrender.com/

🧩 **Algorithmic foundation:** the self-correcting RAG strategy below started as a standalone, easier-to-read reference implementation — see [`local-agentic-rag`](https://github.com/BaoVo1126/local-agentic-rag) — before being reimplemented here against this repo's `AgentStrategy` interface, alongside three other reasoning strategies.

---

## 📌 What this is

Drop a PDF/TXT/MD file in, ask questions through a web UI, CLI, API, or an n8n workflow — the agent retrieves the relevant passages, reasons about the answer, and (in its most advanced mode) **checks its own answer for hallucination before returning it**, retrying automatically if it's wrong. Runs fully offline with zero setup in mock mode, or against a real, free, local model via [Ollama](https://ollama.com) with one environment variable. Retrieval defaults to BM25 fused with a real dense FAISS index for accurate semantic search, and documents are split with a paragraph/sentence-aware `RecursiveCharacterTextSplitter` instead of a fixed word count.

## ✨ What it does

- 🧠 **Four interchangeable reasoning strategies** over the same tools — ReAct, native function-calling, plan-and-execute, and the **self-correcting RAG agent** that is the default: it grades its own retrieved evidence, re-tries when the answer isn't grounded, falls back to the web, and refuses rather than guessing. The other three are baselines the benchmark measures it against.
- 🎯 **Structured prompt contracts, not loose text parsing** — the self-correcting RAG agent's graders/verifiers now speak a strict JSON schema (`{"grounded": true|false, "reason": "..."}`) instead of a bare "yes/no" that silently defaulted to a pass on anything unparseable — see [Prompt engineering](#prompt-engineering) below.
- 🔗 **n8n orchestration layer** — a workflow-level automation layer sitting outside the agent, wiring `/api/upload` + `/api/chat` into webhooks, file-ingestion pipelines, and human-review routing when the agent's self-check flags an answer — see [`n8n/`](n8n/).
- 🔍 **Hybrid retrieval** — BM25 fused via Reciprocal Rank Fusion with a **FAISS** dense-embedding index (zero extra infrastructure), with an optional cross-encoder reranker on top. Both halves are **multilingual** (English + Vietnamese), a mixed-language corpus is split into one sub-index per language so the smaller corpus is not crowded out of the top-k by the larger one, and BM25 folds Vietnamese diacritics into an extra term so OCR tone damage ("chat"/"chát" for "chất") stops hiding passages from search — worth 1.8x recall on scanned text, measured.
- 📄 **Format-aware chunking** — the splitter is chosen per document kind, because one strategy cannot serve them all: tables split on row boundaries with the header repeated on every chunk, Markdown splits on headings and carries a section breadcrumb into each chunk, and prose falls back to paragraph -> sentence -> word boundaries. Budgets are counted in **tokens**, not characters, so a Vietnamese document is not chunked far more finely than an English one from the same setting.
- 🔎 **Reads more than PDFs** — `.pdf .txt .md .csv .tsv .xlsx .docx .html .json .jsonl` plus image files, all through one dispatcher. Extraction that "succeeds" but produces garbled text is rejected and reported rather than silently indexed.
- 🖼️ **OCR fallback for scanned documents** — a page with no usable text layer is rendered and passed to a pluggable OCR engine (Tesseract by default, PaddleOCR opt-in). OCR runs *only* on pages that fail the quality check, so a clean text-layer PDF costs nothing extra.
- 💬 **Persistent chat sessions** — full conversation history saved to Postgres, Redis-cached for fast reads.
- ⚡ **Real-time streaming** — `/api/chat/stream` is true Server-Sent Events; the self-correcting agent streams each retrieval/verify/retry step live as it happens, not after the fact.
- 🔁 **Ingestion that checks itself** — after every build, a phrase is taken out of each document and searched for, to confirm the document can retrieve its own text (`src/ingestion/verification.py`). A build that finishes without raising says nothing about whether retrieval works, and that is precisely how every retrieval bug in [`docs/bugs-found.md`](docs/bugs-found.md) shipped: a chunk count, exit zero, and a broken index. A failing text-layer PDF can be re-read with OCR forced on, keeping whichever version verifies better.
- 📊 **Built-in benchmark** — quantitatively compares all four strategies on pass rate, latency, groundedness, and cost — not just a demo, a measurement tool.
- 🐳 **One-command infra** — `docker compose up` brings up Postgres + Redis for durable chat history alongside the app; `--profile with-n8n` adds the orchestration layer.

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Agent orchestration | Custom `AgentStrategy` interface — self-correcting RAG (default) plus ReAct / function-calling / plan-execute as benchmark baselines |
| Prompt engineering | `src/agents/prompts.py` — role + output-contract system prompts, separated from loop control flow |
| LLM | [Ollama](https://ollama.com) (`qwen2.5:7b`, local & free — chosen over llama3.1, which misspells Vietnamese even when merely copying the question into a tool call) — dual-mode with a deterministic offline mock for CI |
| Retrieval | BM25 (Unicode + Vietnamese diacritic folding) + FAISS dense embeddings (`sentence-transformers`, multilingual), RRF fusion, per-language sub-indexes, optional cross-encoder reranking |
| Ingestion | PyMuPDF text extraction + table detection, pandas/python-docx/stdlib loaders, Tesseract/PaddleOCR fallback (`ingestion/loaders.py`, `ingestion/ocr.py`) |
| Chunking | Per-format splitter router, token-based budgets via `tiktoken` (`ingestion/chunking.py`) |
| Vector storage | FAISS, persisted as a fingerprinted pickle that refuses to load when `data/` or the chunking/embedding settings have changed |
| Chat history | **Postgres**, **Redis** read-cache |
| API | **FastAPI**, Server-Sent Events streaming |
| Workflow orchestration | **n8n** — webhook → ingest → ask-agent → route-on-confidence (`n8n/document_qa_workflow.json`) |
| Frontend | Vanilla JS chat console with a live reasoning-trace view |
| Testing | Pytest — 254 tests (unit / integration / regression), including nineteen real bugs caught by testing against a genuinely fresh environment and against a real model |

## 🏗️ Architecture

```
                                    n8n (workflow orchestration)
                                    webhook → ingest? → ask → route-on-confidence
                                                    │        ▲
                                                    ▼        │
data/*.pdf,csv,docx,… → ingestion (extract → OCR → chunk → verify it can be found again)
                                                            │
                                    retrieval (BM25 + diacritic folding ⊕ FAISS multilingual,
                                               RRF, one sub-index per language)
                                                            │
                                  tools/ (document_search, web_search, calculator, summarize)
                                                            │
                     ┌───────────────┬───────────────┬──────┴──────────┐
                     ▼               ▼               ▼                 ▼
                  ReAct        Function-calling  Plan & execute   Self-correcting RAG ★
                     └── benchmark baselines ────┘                 (the default)
                                                             │
                                          agents/prompts.py (role + JSON output contracts)
                                                             │
                                                 core/llm_client (Mock ↔ Ollama)
                                                             │
                          api/main.py (FastAPI, SSE streaming)  ──▶  Postgres (chat history)
                                                                                   │                          ▲
                                                                              web/ (chat UI)          Redis (read cache)
```

Every agent, tool, and storage backend is swappable through abstract interfaces (`src/core/interfaces.py`) — adding a strategy, or moving from a pickle file to a real database, never touches the other layers. That's not a design claim, it's demonstrated: the self-correcting agent, the Postgres/Redis chat-history layer, the FAISS retrieval backend and the per-language routing layer were all added after the original three-strategy version, with zero changes to the agents or API that didn't need them — `RoutedRetriever` implements the exact same `.search(query, top_k)` interface as the retriever it wraps, so every agent strategy, the API, the CLI and the benchmark work with it unmodified. n8n follows the same principle from the outside: it's one more caller of `/api/chat`, not a rewrite of the agent loop.

## <a name="prompt-engineering"></a>🧠 Prompt engineering & system directives

The self-correcting RAG agent's accuracy depends entirely on its graders and verifiers judging correctly — and the original version asked for a bare "yes" or "no" in free text, parsed with a regex that **defaulted to a pass (`True`) whenever the model's response didn't match cleanly**. That's a silently optimistic failure mode: any hedge, caveat, or off-format response from the model waved a passage or answer through instead of being caught as a real failure.

Following the same separation used in [`AI_AGENT_FROM_ZERO`](https://github.com/breslee1707/AI_AGENT_FROM_ZERO) (prompts kept in their own module, tool/output shape stated explicitly rather than implied), `src/agents/prompts.py` now gives each grader/verifier:

- **An explicit ROLE line** — what kind of judge it is, before the task.
- **A strict output contract** — a one-line JSON object with a fixed schema (`{"grounded": true|false, "reason": "..."}`), not a word the model has to guess the exact phrasing of.
- **Fail-closed parsing** — `_parse_verdict()` in `self_correcting_rag_agent.py` only accepts a clean match against that schema (with a loose yes/no fallback for the mock LLM's canned test phrasing); anything else is now treated as the check **failing**, not passing.

This directly targets the "accuracy still very low" symptom: a low-accuracy self-correcting loop is often not a retrieval problem, it's the *grader itself* rubber-stamping bad passages/answers because its own output wasn't being parsed reliably.

## 🎯 The answering contract

Three of the four strategies in `src/agents/` are **benchmark baselines** —
they exist so the trade-offs between reasoning loops stay measurable. One is
the production path, and it is what a caller gets by default:
`self_correcting_rag` (`src/agents/self_correcting_rag_agent.py`).

Search → Understand & Reason → Check Itself → Answer with sources, under five
promises the other three do not make:

| Promise | Why it exists |
|---|---|
| **Answers in the language it was asked in** | An English system prompt reliably drags a model into answering a Vietnamese question in English. That is a failed answer even when every fact in it is correct — the person asking chose Vietnamese for a reason. The language is detected from the query with a character/stopword check (no model call) and selects a prompt written *in* that language. |
| **Refuses, exactly and detectably** | When no retrieved passage survives grading, the pipeline emits one constant sentence — `Xin lỗi, tôi không tìm thấy thông tin chính xác trong tài liệu.` / `Sorry, I could not find accurate information in the documents.` — so a caller can recognise a refusal by string comparison instead of parsing prose. The older strategy answered from the raw top results in that case, removing the guard in precisely the situation where a hallucination is most likely. |
| **Cites sources that exist** | Passages are numbered `[Source 1..N]` in the context *before* the prompt asks for tags — a prompt cannot ask for `[Source 2]` unless something is labelled Source 2. An answer citing a number that was never supplied is rejected: a fabricated citation is a hallucination wearing the costume of evidence, which makes it worse than an uncited claim, not better. |
| **Shows the model whole passages** | No truncation anywhere in the context builder. |
| **Falls back to the web, last** | When the documents and every retry have failed, `web_search` (DuckDuckGo, no API key) runs on the *original* question — rewrites are tuned to this corpus's vocabulary and searching the open web with them looks for the wrong thing. Web results are labelled with their URL and the answer is prefixed with a notice, because an answer that silently mixes document and web evidence is worse than either alone. A web answer passes the same self-check: search snippets are, if anything, easier to over-read than a retrieved passage, since they are already written as summaries. |

An answer that fails its own self-check is **not** shipped with an
`[unverified]` caveat — it becomes a web answer, or a refusal. A caveated
answer still reads as an answer, and is the shape a hallucination most easily
survives in.

**Cost matters as much as correctness here.** Relevance grading judges all
retrieved candidates in one LLM call, not one call each. Per-passage grading
was correct and unaffordable: measured against `qwen2.5:7b` on CPU at 8.5 s a
call, twelve candidates meant 102 seconds of filtering before the model wrote
a word of the answer — times up to three self-correction attempts. Every unit
test passed throughout, because a mock answers instantly and cannot tell one
call from twelve. That is why an end-to-end run against the real model is part
of the workflow and not an optional extra.

### It works end to end, measured

A Vietnamese question against a **scanned** textbook -- 147 pages, zero
characters of embedded text, recovered by OCR:

```
Q  Ở nhiệt độ thường chất béo tồn tại ở trạng thái nào?
A  Ở nhiệt độ thường, chất béo tồn tại ở trạng thái lỏng hoặc rắn [Source 1].
   [Source 1] = data/sgk_hoa12.pdf, page 13        (4 LLM calls, 0 retries)
```

Answered in the language it was asked in, with a citation that resolves to a
real page, and correct against the book. Retrieval `hit@4` on a six-probe set
spanning both documents and both languages is **6/6**
(`python scripts/diagnose_retrieval.py`) -- including a probe deliberately
typed without diacritics, which the pre-fix pipeline could not match at all.

### Measuring it: `RUN_EVAL`

Prefixing a query with `RUN_EVAL` runs the pipeline and returns a JSON verdict
instead of the answer:

```json
{"query": "...", "answer": "...", "refused": false,
 "faithfulness": 1, "answer_relevance": 1, "context_precision": 1}
```

The three metrics are judged by three separate calls, not one call returning
three fields — a small local model asked for three booleans at once decides
"good" or "bad" once and fills all three to match, destroying the distinction
they exist to draw. Kept separate, they point at different fixes:

- `context_precision = 0` → **the retriever missed.** Changing the prompt or the model will not help.
- `context_precision = 1, faithfulness = 0` → **the evidence arrived and the model ignored it.** Changing retrieval will not help.
- `answer_relevance = 0` → the model answered a different question than the one asked.

A refusal scores faithful, context-imprecise, and not relevant — declining to
answer without evidence is the pipeline working, and an eval that scored it as
a hallucination would push the system back towards guessing.

## 🔗 n8n orchestration layer

`n8n/document_qa_workflow.json` puts a workflow-level automation layer **outside** the agent: a webhook receives a question (and optionally a file to ingest), calls `/api/upload` if needed, calls `/api/chat`, and routes the result — including notifying a human reviewer when the agent's own self-check flags the answer as `unverified`. The agent's reasoning loop stays exactly where it is; n8n only ever talks to it over HTTP, the same way the CLI, web UI, and benchmark do. See [`n8n/README.md`](n8n/README.md) for the full design rationale and setup steps.

```bash
docker compose --profile with-n8n up -d
# n8n UI at http://localhost:5678 — import n8n/document_qa_workflow.json
```

## 🚀 Quickstart

```bash
git clone <this-repo> && cd ai-agent-hybrid-rag
pip install -r requirements.txt

cp "/path/to/your.pdf" data/
python scripts/build_index.py
uvicorn src.api.main:app --reload   # open http://localhost:8000
```

Runs fully offline out of the box (`faiss-cpu`/`sentence-transformers` are in `requirements.txt` by default now, for the FAISS retrieval backend). For a real model (free): install [Ollama](https://ollama.com), `ollama pull qwen2.5:7b`, set `LLM_BACKEND=ollama`. For durable chat history: `docker compose up -d postgres redis`, set `CHAT_HISTORY_BACKEND=postgres` — see [Production upgrade](#production-upgrade) below. For workflow orchestration: `docker compose --profile with-n8n up -d`.

## <a name="production-upgrade"></a>🐳 Production upgrade: real storage, not just a demo

| | Dev default (zero setup) | Production |
|---|---|---|
| Chat history | in-process dict, lost on restart | **Postgres**, read-cached through **Redis** |
| Index | FAISS pickle, rebuilt from `data/` | same — fingerprinted, so a stale one is rebuilt rather than served |
| Streaming | same endpoint | real **SSE** (`text/event-stream`) |
| Orchestration | direct API calls | optional **n8n** workflows (`--profile with-n8n`) |

```bash
docker compose up -d postgres redis
export CHAT_HISTORY_BACKEND=postgres
export POSTGRES_DSN="postgresql+psycopg2://agentlab:agentlab@localhost:5432/agentlab"
export REDIS_URL="redis://localhost:6379/0"
python scripts/build_index.py
uvicorn src.api.main:app --reload
```

Redis is a pure cache, not a hard dependency — if it's unreachable, reads just fall back to Postgres directly instead of failing.

## 📊 Benchmark: four strategies, measured not asserted

`python scripts/run_benchmark.py` runs every strategy over the same eval set and reports pass rate, latency, LLM/tool call count, and **groundedness** (LLM-as-judge check of whether the answer is actually supported by what it retrieved) side by side. The self-correcting agent trades more LLM calls and latency for measurably higher groundedness — the whole point of the comparison is making that trade-off visible instead of just claiming one strategy is "better."

## 🐛 Engineering rigor

Every backend swap in this project (reranker, faiss, chat history) was tested against a **genuinely fresh environment** — an empty database, a missing package — not just the happy path. A second round came from diagnosing bad answers end to end rather than unit-testing components in isolation, which is where the expensive ones were hiding: a BM25 tokenizer that deleted every Vietnamese diacritic, an `EMBEDDING_MODEL` setting nothing read, a `.env` file nothing loaded, and an index that had been stale for weeks with no way to notice. **Nineteen real bugs**, each documented with the exact failure, the fix, and a regression test that locks it in: see [`docs/bugs-found.md`](docs/bugs-found.md).

`python scripts/diagnose_retrieval.py` is the tool that found most of them. It reports what the index actually contains and scores hit@k on a probe set **without involving an LLM**, so "the answer is wrong" can be attributed to retrieval or to generation instead of guessed at.

## 📁 Project layout

```
src/
  core/         interfaces.py (Tool/LLMClient/AgentStrategy ABCs), llm_client.py, config.py
  ingestion/    loaders (pdf/csv/xlsx/docx/html/json/image), ocr (pluggable engines), quality (readability gate),
                chunking (per-format splitter router), indexer (memory + faiss), verification (self-retrieval check)
  retrieval/    bm25 (Unicode + diacritic folding), fusion (RRF), embeddings (multilingual, asymmetric),
                faiss_store + hybrid_faiss, reranker, language (detection) + routed (per-language sub-index)
  tools/        document_search, web_search (DuckDuckGo fallback), calculator, summarize, registry
  agents/       self_correcting_rag (the production contract: bilingual, cited, self-checking,
                web fallback, refuses when unsupported), react + function_calling + plan_execute
                (benchmark baselines), critic.py, prompts.py, factory
  db/           chat session models, Postgres repository
  cache/        Redis read-cache wrapper
  evaluation/   eval_dataset, metrics, groundedness, benchmark
  api/          main.py (FastAPI + SSE), schemas.py
web/            chat UI + live reasoning-trace console
n8n/            document_qa_workflow.json + setup/design notes
scripts/        build_index, run_agent_cli, run_benchmark, run_eval_gate, diagnose_retrieval
docs/           bugs-found.md
test/           unit / integration / regression (254 tests)
docker-compose.yml   Postgres + Redis (chat history) + n8n (optional) + the app
```
