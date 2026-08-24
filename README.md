# Trợ lý Hỏi-Đáp Pháp Luật Việt Nam — AI Agent Pipeline

Hệ thống RAG đa tác tử (multi-agent) cho pháp luật Việt Nam, xây dựng quanh hai nguyên
tắc bất biến:

- **Grounded-or-refuse** — chỉ trả lời từ đoạn luật đã truy xuất; không đủ căn cứ thì
  từ chối, không suy diễn.
- **Version-aware** — mọi trích dẫn đều được đối chiếu hiệu lực qua Legal Knowledge
  Graph trước khi đưa vào câu trả lời.

Mỗi câu trả lời bắt buộc kèm trích dẫn chuẩn `Điều – Khoản – Điểm + Số hiệu văn bản`
và disclaimer cố định.

---

## 1. Kiến trúc

### Pipeline A — Offline: Ingestion & Knowledge Base

```
văn bản thô (.txt/.json/HF dataset)
   └─► StructureAwareParser      Văn bản → Chương → Mục → Điều → Khoản → Điểm
        └─► MetadataExtractor    số hiệu, loại VB, cơ quan, ngày/trạng thái hiệu lực
             └─► RelationExtractor   THAY_THE / SUA_DOI / HUONG_DAN / BAI_BO / CAN_CU
                  ├─► LegalChunkBuilder ──► Qdrant (dense) + BM25 (sparse)
                  └─► KnowledgeGraphBuilder ──► Neo4j hoặc MemoryGraphStore (JSON)
```

### Pipeline B — Online: LangGraph Multi-Agent

```
          ┌─────────┐
          │ router  │──(ngoài phạm vi)───────────────┐
          └────┬────┘                                │
               ▼                                     │
        ┌──────────┐   ┌──────────────┐   ┌──────────▼───┐
   ┌───►│ retrieve │──►│ kg_validate  │──►│    verify    │
   │    └──────────┘   └──────────────┘   └──┬────────┬──┘
   │      ▲ self-correction (rewrite query)  │        │ đủ căn cứ
   │      └──────────────────────────────────┘        ▼
   │                                             ┌────────┐
   │                                             │ answer │
   │                                             └───┬────┘
   │                                    ┌────────────▼────────┐
   └──────────(gate thất bại)───────────│   citation_check    │──► END
                                        └──────────┬──────────┘
                                                   │ hết lượt thử
                                              ┌────▼───┐
                                              │ refuse │──► END
                                              └────────┘
```

| Node | Vai trò |
|---|---|
| `router` | Phân loại intent, viết lại query, tách sub-query, bắt số hiệu/số Điều |
| `retrieve` | BM25 + dense song song → **RRF** → cross-encoder rerank; nạp thẳng Điều được hỏi đích danh |
| `kg_validate` | Lọc văn bản hết hiệu lực; mở rộng multi-hop sang VB hướng dẫn/thay thế |
| `verify` | Chấm `grounding_score`; yếu → viết lại query và quay lại `retrieve` |
| `answer` | Sinh câu trả lời **chỉ** từ evidence, kèm trích dẫn chuẩn |
| `citation_check` | Chốt chặn: audit trích dẫn (deterministic) + kiểm chứng claim nguyên tử |
| `refuse` | Nhánh từ chối có giải thích lý do + liệt kê điều khoản bị loại vì hiệu lực |

Mọi vòng lặp đều bị chặn bởi `MAX_RETRIEVAL_ATTEMPTS`; nút `refuse` tiếp cận được từ 3
vị trí — đó là bảo đảm cấu trúc cho nguyên tắc grounded-or-refuse.

---

## 2. Chạy thử trong 60 giây (profile MVP, offline hoàn toàn)

```bash
pip install -r requirements.txt          # hoặc: pip install -e ".[dev,nlp]"
cp .env.example .env                     # mặc định đã là profile mvp

python scripts/ingest_priority.py         # nạp văn bản nền tảng từ corpus HF
python scripts/run_ui.py                  # Web UI  -> http://localhost:8501
python scripts/run_api.py --port 8080     # REST API -> http://localhost:8080/docs
python scripts/diagnose.py                # chẩn đoán 4 tầng lỗi (a/b/c/d)
python scripts/verify_goldens.py          # đối chiếu nhãn vàng với văn bản gốc
python scripts/run_eval.py --regression --diagnose-failures
pytest                                    # 82 test nhanh
pytest -m slow                            # 5 test tích hợp trên corpus thật
```

Web UI có 4 tab: **Hỏi đáp** (chat, hiện đầy đủ bằng chứng + trace), **Giám sát**
(chỉ số vận hành đọc từ run log), **Đánh giá** (chạy golden set ngay trên giao diện),
**Kho văn bản** (tra cứu hiệu lực trực tiếp trên Knowledge Graph).

Profile `mvp` không cần GPU, không cần tải model, không cần Qdrant/Neo4j server:
Qdrant chạy in-memory, KG lưu JSON, embedder/reranker/LLM dùng bản stub tất định.

Ví dụ gọi API:

```bash
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{
  "question": "Nghị định nào đang hướng dẫn Điều 26 của Luật Doanh nghiệp 59/2020/QH14 và còn hiệu lực?",
  "include_trace": true
}'
```

---

## 3. Cấu trúc thư mục

```
AIAgent_phapluatVN/
├── src/legal_agent/
│   ├── config.py                  # Settings (pydantic-settings), 1 nguồn cấu hình duy nhất
│   ├── logging_config.py
│   ├── domain/                    # tầng thuần dữ liệu, không phụ thuộc hạ tầng
│   │   ├── enums.py               # DocumentType, EffectStatus, RelationType, NodeLevel, QueryIntent
│   │   ├── citation.py            # Citation: render / parse / covers / parse_cited
│   │   ├── document.py            # LegalDocumentMeta, LegalRelation, status_as_of()
│   │   ├── node.py                # LegalNode - cây cấu trúc văn bản
│   │   └── chunk.py               # LegalChunk (đơn vị trích dẫn), RetrievedChunk
│   ├── ingestion/                 # ── MODULE 1 ──
│   │   ├── patterns.py            # toàn bộ regex tiếng Việt + chuẩn hoá Unicode
│   │   ├── parser.py              # StructureAwareParser (state machine + ancestor stack)
│   │   ├── metadata_extractor.py  # số hiệu, tiêu đề, cơ quan, ngày & trạng thái hiệu lực
│   │   ├── relation_extractor.py  # khai thác quan hệ văn bản → cạnh của KG
│   │   ├── chunker.py             # cắt theo Điều/Khoản, không bao giờ cắt giữa Điểm
│   │   ├── loaders.py             # .txt / .json / .jsonl / HuggingFace dataset
│   │   └── pipeline.py            # IngestionPipeline: parse → KG → index
│   ├── indexing/
│   │   ├── tokenizer.py           # tách từ tiếng Việt (underthesea/pyvi) cho BM25
│   │   ├── embedder.py            # vnlegal-lal / bge-m3 + HashingEmbedder (offline)
│   │   ├── bm25_index.py          # BM25Okapi + pickle sidecar
│   │   └── qdrant_store.py        # collection, payload filter theo hiệu lực, tra cứu theo Điều
│   ├── kg/
│   │   ├── base.py                # LegalGraphStore Protocol + GraphVerdict
│   │   ├── memory_store.py        # backend MVP (JSON snapshot)
│   │   ├── neo4j_store.py         # backend production (Cypher)
│   │   └── builder.py             # cạnh nghịch đảo + suy luận lan truyền hết hiệu lực
│   ├── retrieval/
│   │   ├── hybrid.py              # HybridRetriever + Reciprocal Rank Fusion
│   │   └── reranker.py            # bge-reranker-v2-m3 + stub theo độ phủ từ vựng
│   ├── llm/
│   │   ├── base.py                # LLMClient Protocol, JSON recovery
│   │   ├── prompts.py             # toàn bộ prompt (router/answer/verifier/claim)
│   │   ├── vllm_client.py         # vLLM qua API OpenAI-compatible
│   │   └── stub_client.py         # LLM tất định để chạy/kiểm thử offline
│   ├── agents/                    # ── MODULE 2 ──
│   │   ├── state.py               # AgentState + trace append-only
│   │   ├── nodes/                 # router, retrieval, kg_validator, verifier,
│   │   │                          # answer, citation_checker, refusal
│   │   ├── edges.py               # toàn bộ policy điều hướng (conditional edges)
│   │   ├── graph.py               # lắp ráp & compile LangGraph
│   │   └── service.py             # LegalAgentService: bootstrap + ask()
│   ├── monitoring/                # ── MONITOR ──
│   │   ├── run_logger.py          # ghi 1 dòng JSONL mỗi lượt hỏi + đo thời gian
│   │   ├── metrics.py             # tổng hợp refusal/retry rate, latency p50/p95 theo node
│   │   └── tracing.py             # bật/kiểm tra cấu hình LangSmith
│   ├── evaluation/                # ── EVAL ──
│   │   ├── dataset.py             # schema golden set + loader
│   │   ├── metrics.py             # retrieval/citation recall, precision, stale rate
│   │   └── runner.py              # chạy bộ case -> báo cáo JSON + Markdown
│   └── api/
│       ├── main.py                # FastAPI app + lifespan warm-up
│       ├── deps.py, schemas.py
│       └── routers/               # /ask, /health, /metrics, /runs, /admin
├── app/streamlit_app.py           # ── WEB UI ── 4 tab
├── scripts/                       # run_ui, run_api, run_ingestion, run_eval, ask_cli
├── tests/                         # 80 test (75 nhanh + 5 tích hợp)
├── data/
│   ├── raw/                       # 3 văn bản mẫu của MVP
│   └── eval/golden_set.jsonl      # 10 case đánh giá
├── .github/workflows/             # ci.yml (lint/test/eval/docker) + deploy.yml
├── docker/                        # Dockerfile + compose (Qdrant + Neo4j + API + UI)
└── render.yaml
```

---

## 4. Quyết định thiết kế đáng chú ý

**Không chunk theo số ký tự.** Đơn vị chunk là Khoản (hoặc cả Điều nếu không có Khoản).
Một chunk cắt ngang hai Khoản thì không thể trích dẫn, mà chunk không trích dẫn được thì
vô dụng với hệ grounded-or-refuse. Khoản quá dài chỉ được cắt tại ranh giới Điểm.

**Ngữ cảnh cha tách khỏi nội dung.** `context_header` (văn bản > chương > mục > điều +
câu dẫn) chỉ được ghép vào khi *embed*; văn bản hiển thị và văn bản đem đi kiểm chứng
vẫn là nguyên văn điều luật.

**Phân biệt ngữ cảnh khi parse.** `1.` chỉ mở Khoản khi đang mở một Điều; `a)` chỉ mở
Điểm khi đang ở trong Khoản/Điều. Nhờ vậy danh sách đánh số ở phần mở đầu không phá cây.

**RRF thay vì nội suy điểm số.** Điểm BM25 và cosine không cùng thang đo; RRF chỉ dùng
*thứ hạng* nên không cần hiệu chỉnh theo từng corpus.

**Audit trích dẫn là deterministic, không nhờ LLM.** Chỉ các trích dẫn nằm trong ngoặc
đơn (đúng định dạng prompt bắt buộc) mới bị audit — số hiệu xuất hiện *bên trong đoạn
luật được trích* là dẫn chiếu của luật, không phải khẳng định thẩm quyền của trợ lý.
Trích dẫn không khớp evidence ⇒ trượt gate ngay, không cần model.

**Bằng chứng nạp thêm qua KG cũng phải qua bộ lọc hiệu lực.** Một chuỗi quan hệ có thể
dẫn tới văn bản cũng đã hết hiệu lực; đi vòng qua graph không được trở thành đường tránh
kiểm tra phiên bản.

**Mọi thành phần nặng đều có stub tất định.** Toàn bộ graph chạy và test được offline,
không GPU, không tải model — đổi backend chỉ bằng biến môi trường.

---

## 5. Cấu hình (biến môi trường chính)

| Biến | MVP | Production | Ghi chú |
|---|---|---|---|
| `APP_PROFILE` | `mvp` | `prod` | |
| `LLM_BACKEND` | `stub` | `openai_compatible` | trỏ `LLM_BASE_URL` tới vLLM |
| `LLM_MODEL` | – | `Qwen/Qwen2.5-7B-Instruct` | |
| `EMBEDDING_BACKEND` | `stub` | `sentence_transformers` | `darklethelong/vnlegal-lal`, fallback `BAAI/bge-m3` |
| `RERANKER_BACKEND` | `stub` | `flag_embedding` | `BAAI/bge-reranker-v2-m3` |
| `QDRANT_MODE` | `memory` | `server` | |
| `GRAPH_BACKEND` | `memory` | `neo4j` | |
| `MAX_RETRIEVAL_ATTEMPTS` | `2` | `2` | trần vòng lặp self-correction |
| `GROUNDING_THRESHOLD` | `0.6` | `0.6` | ngưỡng của `verify` |
| `CLAIM_SUPPORT_THRESHOLD` | `0.6` | `0.6` | ngưỡng của post-hoc gate |

Chuyển sang production:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --port 8000
docker compose -f docker/docker-compose.yml up -d      # Qdrant + Neo4j + API
python scripts/run_ingestion.py                        # nạp KB vào server thật
```

---

## 6. Phạm vi MVP & dữ liệu

MVP đóng gói trọn vẹn lĩnh vực **Luật Doanh nghiệp** với 3 văn bản trong `data/raw/`:

| Văn bản | Trạng thái | Vai trò trong demo |
|---|---|---|
| Luật Doanh nghiệp 59/2020/QH14 | còn hiệu lực | văn bản gốc |
| Nghị định 01/2021/NĐ-CP | còn hiệu lực | `HUONG_DAN` Điều 26 → test multi-hop |
| Luật Doanh nghiệp 68/2014/QH13 | hết hiệu lực | bị `THAY_THE` → test version-aware |

Ba văn bản này phủ đủ cả ba loại quan hệ cần thiết để chứng minh pipeline. Thêm lĩnh vực
mới chỉ cần bỏ file `.txt` vào `data/raw/` rồi chạy lại `scripts/run_ingestion.py` —
không phải sửa code. File nguồn có thể khai báo front-matter để ghim metadata:

```
---
effect_status: het_hieu_luc
expiry_date: ngày 01 tháng 01 năm 2021
---
```

---

## 7. Kiểm thử

```bash
pytest            # 75 test nhanh
pytest -m slow    # 5 test tích hợp: ingest thật -> hỏi thật -> kiểm trích dẫn thật
ruff check src tests scripts app
```

| Nhóm | Nội dung được khoá bằng test |
|---|---|
| `test_parser` | cây Chương/Mục/Điều/Khoản/Điểm, bẫy dẫn chiếu "Điều X của Luật này", front-matter, ngày hiệu lực tự chiếu |
| `test_chunker_citation` | chunk = Khoản, Điểm không bị cắt rời, `parse_cited` bỏ qua số hiệu nằm trong đoạn luật được trích |
| `test_kg_and_fusion` | lan truyền hết hiệu lực, multi-hop theo Điều, hiệu lực theo mốc thời gian, công thức RRF |
| `test_agent_graph` | trả lời có căn cứ, từ chối ngoài phạm vi, trần self-correction, chặn trích dẫn bịa, giữ/loại văn bản hết hiệu lực |
| `test_monitoring` | ghi/đọc run log, cộng dồn latency khi retry, các tỷ lệ tổng hợp |
| `test_evaluation` | recall/precision trích dẫn, phát hiện trích dẫn hết hiệu lực, chấm ca từ chối |
| `test_api` | hợp đồng HTTP, từ chối là 200 chứ không phải lỗi |

---

## 8. Đánh giá (Eval)

```bash
python scripts/run_eval.py                  # in báo cáo + lưu JSON/Markdown
python scripts/run_eval.py --fail-under 0.8 # dùng làm cổng chặn trong CI
```

Golden set (`data/eval/golden_set.jsonl`) gắn kỳ vọng ở ba tầng cho mỗi câu hỏi:

```jsonc
{
  "case_id": "eval-003-multihop-huong-dan",
  "question": "Nghị định nào đang hướng dẫn Điều 26 của Luật Doanh nghiệp 59/2020/QH14...",
  "expected_citations": ["Điều 1, Nghị định 01/2021/NĐ-CP"],   // phải trích dẫn
  "forbidden_citations": [],                                    // không được trích
  "expected_status": "answered",                                // trả lời hay từ chối
  "allow_stale_citations": false                                // được phép dẫn VB hết hiệu lực?
}
```

Chỉ số và ý nghĩa:

| Chỉ số | Ý nghĩa |
|---|---|
| `retrieval_recall` | Điều khoản đúng có vào được pool bằng chứng không — trần trên của mọi chỉ số sau |
| `citation_recall` | Điều khoản đúng có thực sự được trích dẫn không |
| `citation_precision` | Mọi trích dẫn phát ra đều có bằng chứng chống lưng |
| `stale_citation_rate` | **Phải bằng 0** — trích dẫn văn bản hết hiệu lực là lỗi chết người của sản phẩm pháp lý |
| `status_accuracy` | Trả lời khi cần trả lời, từ chối khi cần từ chối — từ chối được **chấm điểm**, không phải cửa thoát |

Kết quả hiện tại (profile MVP, LLM stub):

```
pass_rate 0.90 · status_accuracy 1.00 · retrieval_recall 1.00
citation_recall 0.88 · citation_precision 1.00 · stale_citation_rate 0.00
retry_rate 0.20 · avg_latency 471 ms
```

---

## 9. Giám sát (Monitor)

Mỗi lượt hỏi ghi một dòng JSONL vào `data/processed/run_log.jsonl` (append-only, đọc
được bằng pandas một dòng, đẩy vào log pipeline nào cũng được):

```bash
curl http://localhost:8080/metrics       # tổng hợp
curl http://localhost:8080/runs?limit=20 # nhật ký gần nhất
```

Đo cả *chất lượng* chứ không chỉ độ trễ:

- `refusal_rate` — tăng đột ngột nghĩa là truy xuất/corpus vừa hỏng; **giảm về 0 không
  phải tin vui**, thường là chốt chặn đã ngừng bắt lỗi.
- `retry_rate` — proxy trực tiếp cho chất lượng truy xuất lượt đầu.
- `latency p50/p95` tổng thể **và theo từng node** (đo tại `agents/graph.py:instrument`),
  nên reranker chậm hay LLM dài dòng lộ ra ngay.

LangSmith là tuỳ chọn, bật bằng biến môi trường (`LANGSMITH_TRACING=true`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`); LangGraph tự gửi trace, module `tracing.py`
chỉ kiểm tra cấu hình để một API key sai báo lỗi ngay lúc khởi động thay vì im lặng.

---

## 10. CI/CD & Deploy

`.github/workflows/ci.yml` — 5 job:

| Job | Nội dung |
|---|---|
| `lint` | `ruff check src tests scripts app` |
| `test` | test nhanh trên Python 3.11 và 3.12 |
| `integration` | `pytest -m slow` (dựng index thật) |
| `evaluate` | `run_eval.py --fail-under 0.8` — **build fail nếu chất lượng tụt dưới ngưỡng**, báo cáo được upload làm artifact |
| `docker` | build image + smoke-test `/live` trong container |

`.github/workflows/deploy.yml` gọi Render deploy hook rồi chờ `/live` xanh (cần secret
`RENDER_DEPLOY_HOOK` và `SERVICE_URL`).

```bash
docker compose -f docker/docker-compose.yml up -d   # Qdrant + Neo4j + API + UI
```

---

## 11. Chẩn đoán bốn tầng lỗi

Một câu trả lời sai chỉ có thể phát sinh ở đúng bốn chỗ, và cách sửa ở mỗi chỗ hoàn toàn
khác nhau. Đo trước khi sửa là bắt buộc:

| Tầng | Câu hỏi | Nếu hỏng thì sửa ở đâu |
|---|---|---|
| **a. corpus** | Văn bản gốc có trong KB chưa? | `scripts/ingest_priority.py` |
| **b. parse** | Parser có tách đúng Điều chứa đáp án? | `ingestion/parser.py`, `patterns.py` |
| **c. retrieval** | Chunk đúng có lọt top-k? | `retrieval/hybrid.py`, embedder, reranker |
| **d. generation** | Có bằng chứng rồi, câu trả lời có dùng đúng? | `agents/nodes/answer.py`, prompt |

```bash
python scripts/diagnose.py                                  # 2 câu canary
python scripts/run_eval.py --regression --diagnose-failures # quy mọi case fail về 1 tầng
```

Tầng đầu tiên hỏng là thủ phạm; mọi tầng sau được đánh dấu `BLOCKED` vì kết quả của
chúng không còn mang thông tin gì.

---

## 12. Priority ingestion - văn bản nền tảng

Nguồn: `th1nhng0/vietnamese-legal-documents` (171.556 văn bản crawl từ vbpl.vn), gồm
`metadata`, `content` (HTML) và `relationships`.

Hai quyết định rút ra từ việc **đo**, không phải phỏng đoán:

- **Lọc theo loại văn bản + số hiệu, không lọc theo từ khoá tiêu đề.** Tìm tiêu đề chứa
  "Xử lý vi phạm hành chính" trả về 340 kết quả mà đa số là Quyết định UBND tỉnh. Văn bản
  nền tảng chỉ nằm trong `Hiến pháp` (6), `Bộ luật` (17), `Luật` (610).
- **Metadata của corpus là nguồn chân lý.** Ngày hiệu lực, cơ quan ban hành và trạng thái
  hiệu lực được ghi vào front-matter của file thô để parser dùng luôn. Bản trích đoạn tự
  soạn trước đây ghi Nghị định 01/2021/NĐ-CP là "còn hiệu lực", trong khi corpus ghi
  **"Hết hiệu lực toàn bộ"** - đúng loại sai lệch mà một trợ lý pháp lý không được mắc.

`content.parquet` nặng 785 MB nhưng chỉ đọc theo row-group với column pruning, nên chỉ
vài chục MB thực sự đi qua mạng.

Kết quả parse trên văn bản thật (tỷ lệ parse ra 0 Điều: **0.0%**, nên **không cần Docling**
theo đúng ngưỡng 10% đã đặt ra):

| Văn bản | Điều | Khoản | Điểm |
|---|---|---|---|
| Hiến pháp 2013 | 120 | 244 | 0 |
| Bộ luật Hình sự 2015 | 426 | 1.397 | 2.932 |
| Bộ luật Dân sự 2015 | 689 | 1.418 | 312 |
| Bộ luật Lao động 2019 | 220 | 648 | 287 |
| Luật XLVPHC 2012 | 142 | 432 | 418 |

---

## 13. Bộ regression nhiều lĩnh vực

`src/legal_agent/evaluation/datasets/legal_qa_regression.py` - **30 case** trải Hiến pháp,
Hình sự, Dân sự, Lao động, Doanh nghiệp, Hành chính, Hôn nhân gia đình, cộng 3 case bắt
buộc từ chối. Hai câu canary chỉ là 2 trong số đó.

Mọi nhãn vàng đều được đối chiếu ngược với văn bản gốc bằng `scripts/verify_goldens.py`.
Quá trình này đã bắt được nhãn sai của chính tôi (Hiến pháp viết "năm năm" chứ không phải
"05 năm") và một bug trong chính script kiểm tra (khớp mờ theo tên khiến nhãn của Luật
Doanh nghiệp 2020 bị kiểm chứng nhầm trên bản 2014 đã hết hiệu lực).

Kết quả hiện tại (12 văn bản, 7.826 chunk, LLM stub):

```
pass_rate 0.80 · status_accuracy 0.90 · retrieval_recall 0.85
citation_recall 0.78 · citation_precision 0.89 · stale_citation_rate 0.00
retry_rate 0.20 · avg_latency 445 ms

Phân bố tầng lỗi của 6 case chưa đạt: a_corpus 4 · c_retrieval 1 · d_generation 1
```

---

## 14. Hạn chế đã biết

- **Thiếu Luật Doanh nghiệp 59/2020/QH14 và Luật Đất đai 2024 trong corpus.** Bản ghi ưu
  tiên của hai văn bản này rỗng nội dung; với Luật Doanh nghiệp đã có cơ chế ứng viên dự
  phòng (id 142881), chạy `python scripts/ingest_priority.py --only "Doanh nghiệp 2020"`
  là nạp được. Luật Đất đai 2024 chỉ có một bản ghi duy nhất và nó rỗng - đây là khoảng
  trống của nguồn dữ liệu, không phải của pipeline. 4/6 case chưa đạt đến từ đây.
- **LLM stub không phải baseline chất lượng.** Câu trả lời được ghép từ 3 khối bằng chứng
  đầu tiên, nên câu hỏi dạng liệt kê ("những hình phạt chính nào") dễ trượt dù điều luật
  đúng đã nằm trong context. Chạy với vLLM + Qwen2.5 để đo chất lượng thật.
- **Trích dẫn buộc phải nằm trong ngoặc đơn**, theo đúng định dạng bắt buộc ở mục 7.
- **Quan hệ KG khai thác bằng luật (rule-based).** Đã có hai lớp chặn cho bẫy nguy hiểm
  nhất (luật sửa đổi bị hiểu nhầm thành luật thay thế), nhưng văn bản viết lắt léo vẫn có
  thể bị bỏ sót quan hệ; khi đó hệ thống trả `khong_xac_dinh` và không trích dẫn.

---

⚖️ **Lưu ý:** Đây là công cụ hỗ trợ tra cứu pháp luật, không thay thế ý kiến tư vấn của
luật sư hoặc cơ quan nhà nước có thẩm quyền.
