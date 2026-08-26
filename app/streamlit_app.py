from __future__ import annotations
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.agents import LegalAgentService
from legal_agent.config import get_settings
from legal_agent.logging_config import setup_logging

STATUS_BADGE = {
    "con_hieu_luc": ("🟢", "Còn hiệu lực"),
    "het_hieu_luc": ("🔴", "Hết hiệu lực"),
    "het_hieu_luc_mot_phan": ("🟠", "Hết hiệu lực một phần"),
    "chua_co_hieu_luc": ("🔵", "Chưa có hiệu lực"),
    "khong_xac_dinh": ("⚪", "Chưa xác định"),
}

SAMPLE_QUESTIONS = [
    "Ai không có quyền thành lập và quản lý doanh nghiệp tại Việt Nam?",
    "Nghị định nào đang hướng dẫn Điều 26 của Luật Doanh nghiệp 59/2020/QH14 và còn hiệu lực?",
    "Luật Doanh nghiệp 68/2014/QH13 còn hiệu lực không?",
    "Vốn điều lệ được hiểu như thế nào?",
    "Điều 32 Nghị định 01/2021/NĐ-CP quy định điều kiện nào để hồ sơ được tiếp nhận?",
    "Cách nấu phở bò ngon nhất là gì?",
]

st.set_page_config(page_title="Trợ lý Pháp luật Việt Nam", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def load_service() -> LegalAgentService:
    """Khởi tạo service một lần cho cả phiên (model + index đều tốn kém)."""
    setup_logging("WARNING")
    service = LegalAgentService(get_settings())
    service.bootstrap()
    return service


def status_chip(effect_status: str) -> str:
    icon, label = STATUS_BADGE.get(effect_status, STATUS_BADGE["khong_xac_dinh"])
    return f"{icon} {label}"


def render_answer(result: dict) -> None:
    """Hiển thị một lượt trả lời: câu trả lời, chỉ số, bằng chứng, trace."""
    answer = result["answer"]

    if answer.status == "refused":
        st.warning("**Hệ thống từ chối trả lời** — không đủ căn cứ pháp lý.", icon="⛔")
    else:
        st.success("**Đã trả lời dựa trên căn cứ đã đối chiếu**", icon="✅")

    st.markdown(answer.answer)

    columns = st.columns(5)
    columns[0].metric("Grounding", f"{answer.grounding_score:.2f}",
                      help="Điểm thẩm định bằng chứng của node verify (ngưỡng 0.60)")
    columns[1].metric("Support ratio", f"{answer.support_ratio:.2f}",
                      help="Tỷ lệ luận điểm được chứng minh ở chốt chặn cuối")
    columns[2].metric("Số lượt truy xuất", answer.attempts,
                      help="Lớn hơn 1 nghĩa là vòng self-correction đã kích hoạt")
    columns[3].metric("Bằng chứng", len(answer.evidence))
    columns[4].metric("Độ trễ", f"{answer.latency_ms:,.0f} ms")

    if answer.graph_notes:
        with st.expander(" Ghi chú hiệu lực từ Knowledge Graph", expanded=True):
            for note in answer.graph_notes:
                st.markdown(f"- {note}")

    if answer.excluded_chunks:
        with st.expander(f" Điều khoản bị loại vì hiệu lực ({len(answer.excluded_chunks)})"):
            for item in answer.excluded_chunks:
                st.markdown(f"- **{item['citation']}** — {item['reason']}")

    if answer.evidence:
        with st.expander(f"📄 Bằng chứng đã truy xuất ({len(answer.evidence)})"):
            for index, item in enumerate(answer.evidence, start=1):
                st.markdown(
                    f"**[{index}] {item['citation']}** &nbsp; {status_chip(item['effect_status'])}"
                    f" &nbsp;·&nbsp; `{item['source']}` &nbsp;·&nbsp; score `{item['score']:.4f}`"
                )
                if item.get("graph_note"):
                    st.caption(f"↳ {item['graph_note']}")
                st.text(item["text"][:900])
                st.divider()

    if answer.trace:
        with st.expander("🔍 Trace của LangGraph"):
            frame = pd.DataFrame([
                {
                    "bước": index,
                    "node": entry.get("node", ""),
                    "ms": entry.get("elapsed_ms", 0.0),
                    "chi tiết": ", ".join(
                        f"{key}={value}" for key, value in entry.items()
                        if key not in {"node", "elapsed_ms", "hits", "scores", "queries"}
                    )[:180],
                }
                for index, entry in enumerate(answer.trace, start=1)
            ])
            st.dataframe(frame, use_container_width=True, hide_index=True)


def tab_chat(service: LegalAgentService, as_of: str | None) -> None:
    """Tab hỏi đáp: lịch sử hội thoại + ô nhập câu hỏi."""
    st.caption("Mỗi câu trả lời chỉ dựa trên điều khoản đã truy xuất. Không đủ căn cứ, "
               "hệ thống sẽ từ chối thay vì suy diễn.")

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant", avatar="⚖️"):
            render_answer(entry)

    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Nhập câu hỏi pháp luật...") or pending
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất, đối chiếu hiệu lực và kiểm chứng trích dẫn..."):
            started = time.perf_counter()
            answer = service.ask(question, session_id="streamlit", as_of=as_of)
            elapsed = (time.perf_counter() - started) * 1000
        if not answer.latency_ms:
            answer.latency_ms = elapsed
        result = {"question": question, "answer": answer}
        render_answer(result)
    st.session_state.history.append(result)


def tab_monitoring(service: LegalAgentService) -> None:
    """Tab giám sát: đọc trực tiếp từ run log JSONL."""
    summary = service.metrics()
    if not summary.total_runs:
        st.info("Chưa có lượt hỏi nào được ghi nhận. Hãy đặt câu hỏi ở tab Hỏi đáp.")
        return

    columns = st.columns(4)
    columns[0].metric("Tổng lượt hỏi", summary.total_runs)
    columns[1].metric("Tỷ lệ từ chối", f"{summary.refusal_rate:.0%}",
                      help="Giảm về 0 không phải tin vui: thường là chốt chặn đã ngừng bắt lỗi")
    columns[2].metric("Tỷ lệ self-correct", f"{summary.retry_rate:.0%}",
                      help="Proxy trực tiếp cho chất lượng truy xuất lượt đầu")
    columns[3].metric("Độ trễ p95", f"{summary.latency_p95_ms:,.0f} ms")

    columns = st.columns(4)
    columns[0].metric("Trả lời được", summary.answered)
    columns[1].metric("Grounding TB", f"{summary.avg_grounding:.2f}")
    columns[2].metric("Support TB", f"{summary.avg_support:.2f}")
    columns[3].metric("Độ trễ p50", f"{summary.latency_p50_ms:,.0f} ms")

    left, right = st.columns(2)
    with left:
        st.subheader("Độ trễ trung bình theo node")
        if summary.node_latency_ms:
            st.bar_chart(pd.Series(summary.node_latency_ms, name="ms"))
    with right:
        st.subheader("Phân bố intent")
        if summary.intents:
            st.bar_chart(pd.Series(summary.intents, name="lượt"))

    if summary.top_cited_documents:
        st.subheader("Văn bản được trích dẫn nhiều nhất")
        st.dataframe(pd.DataFrame(summary.top_cited_documents,
                                  columns=["Số hiệu", "Số lần trích dẫn"]),
                     use_container_width=True, hide_index=True)

    st.subheader("Nhật ký gần nhất")
    records = service.recorder.read_all(limit=25)[::-1]
    if records:
        frame = pd.DataFrame(records)[
            ["timestamp", "question", "status", "intent", "grounding_score",
             "support_ratio", "attempts", "latency_ms"]
        ]
        st.dataframe(frame, use_container_width=True, hide_index=True)


def tab_evaluation(service: LegalAgentService) -> None:
    """Tab đánh giá: chạy golden set và hiển thị báo cáo."""
    from legal_agent.evaluation import EvaluationRunner, load_golden_set

    settings = get_settings()
    st.caption("Golden set gắn kỳ vọng cho từng câu hỏi: điều khoản phải truy xuất được, "
               "điều khoản phải trích dẫn, điều khoản KHÔNG được trích, và hành vi mong "
               "đợi (trả lời hay từ chối).")

    try:
        cases = load_golden_set(settings.abs_eval_dataset_path)
    except FileNotFoundError as error:
        st.error(str(error))
        return

    st.write(f"**{len(cases)} case** trong `{settings.eval_dataset_path}`")
    if st.button(" Chạy đánh giá", type="primary"):
        progress = st.progress(0.0, text="Đang chạy...")
        report = EvaluationRunner(service).run(
            cases,
            progress=lambda index, total, _: progress.progress(
                index / total, text=f"Đang chạy case {index}/{total}"),
        )
        progress.empty()
        st.session_state.eval_report = report

    report = st.session_state.get("eval_report")
    if report is None:
        return

    aggregate = report.aggregate()
    columns = st.columns(4)
    columns[0].metric("Pass rate", f"{aggregate['pass_rate']:.0%}")
    columns[1].metric("Retrieval recall", f"{aggregate['retrieval_recall']:.2f}")
    columns[2].metric("Citation precision", f"{aggregate['citation_precision']:.2f}")
    columns[3].metric("Trích dẫn hết hiệu lực", f"{aggregate['stale_citation_rate']:.0%}",
                      help="Chỉ số sống còn của sản phẩm pháp lý - phải bằng 0")

    st.dataframe(
        pd.DataFrame([
            {
                "case": result.case_id,
                "đạt": "✅" if result.passed else "❌",
                "kỳ vọng": result.expected_status,
                "thực tế": result.actual_status,
                "retr. recall": result.retrieval_recall,
                "cit. recall": result.citation_recall,
                "cit. precision": result.citation_precision,
                "thiếu trích dẫn": ", ".join(result.missing_citations) or "-",
            }
            for result in report.results
        ]),
        use_container_width=True, hide_index=True,
    )


def tab_documents(service: LegalAgentService, as_of: str | None) -> None:
    """Tab kho văn bản: danh sách VB trong KG + công cụ tra cứu hiệu lực."""
    documents = service.graph_store.all_documents()
    st.dataframe(
        pd.DataFrame([
            {
                "Số hiệu": document.doc_number,
                "Tên văn bản": document.title,
                "Loại": document.doc_type.value,
                "Hiệu lực": status_chip(document.effect_status.value),
                "Ngày hiệu lực": document.effective_date,
                "Ngày hết hiệu lực": document.expiry_date,
                "Cơ quan": document.issuing_body,
                "Quan hệ": len(document.relations),
            }
            for document in documents
        ]),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Tra cứu hiệu lực trên Knowledge Graph")
    left, right = st.columns([2, 1])
    doc_number = left.selectbox("Số hiệu văn bản",
                                [document.doc_number for document in documents])
    dieu = right.text_input("Số Điều (tuỳ chọn)", placeholder="ví dụ: 26")

    if doc_number:
        verdict = service.graph_store.validate(
            doc_number, dieu.strip() or None, service.context.as_of_date(as_of))
        payload = verdict.as_dict()
        st.markdown(f"**Trạng thái:** {status_chip(payload['status'])}")
        if payload["replaced_by"]:
            st.markdown(f"**Bị thay thế bởi:** {', '.join(payload['replaced_by'])}")
        if payload["amended_by"]:
            st.markdown(f"**Bị sửa đổi bởi:** {', '.join(payload['amended_by'])}")
        if payload["guided_by"]:
            st.markdown("**Được hướng dẫn bởi:** " + ", ".join(
                f"{entry['doc_number']}"
                + (f" (Điều {entry['dieu']})" if entry.get("dieu") else "")
                for entry in payload["guided_by"]))
        if not any([payload["replaced_by"], payload["amended_by"], payload["guided_by"]]):
            st.caption("Không có quan hệ nào được ghi nhận cho phạm vi đã chọn.")


def main() -> None:
    st.session_state.setdefault("history", [])

    st.title("⚖️ Trợ lý Hỏi-Đáp Pháp Luật Việt Nam")
    st.caption("Grounded-or-refuse · Version-aware · Trích dẫn Điều – Khoản – Điểm")

    with st.spinner("Đang nạp Knowledge Base (parse văn bản, dựng index và graph)..."):
        service = load_service()
    settings = get_settings()

    with st.sidebar:
        st.header("Cấu hình")
        st.caption(f"Profile: **{settings.app_profile}**")
        st.code(
            f"LLM        : {settings.llm_backend}\n"
            f"Embedding  : {settings.embedding_backend}\n"
            f"Reranker   : {settings.reranker_backend}\n"
            f"Qdrant     : {settings.qdrant_mode}\n"
            f"Graph      : {settings.graph_backend}",
            language="text",
        )
        columns = st.columns(2)
        columns[0].metric("Chunks", service.vector_store.count())
        columns[1].metric("Văn bản", len(service.graph_store.all_documents()))

        st.divider()
        st.subheader("Ngày tham chiếu hiệu lực")
        use_as_of = st.checkbox("Hỏi theo một mốc thời gian khác", value=False,
                                help="Kiểm tra hiệu lực tại thời điểm chỉ định "
                                     "thay vì hôm nay")
        as_of = st.date_input("Ngày", value=date.today()).isoformat() if use_as_of else None

        st.divider()
        st.subheader("Câu hỏi mẫu")
        for index, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample-{index}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

        st.divider()
        if st.button("🗑️ Xoá lịch sử hội thoại", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        if settings.llm_backend == "stub":
            st.warning("Đang dùng LLM stub (offline). Câu trả lời được ghép trực tiếp từ "
                       "điều luật truy xuất được. Đặt `LLM_BACKEND=openai_compatible` để "
                       "dùng vLLM/Qwen2.5.", icon="⚠️")

    chat, monitoring, evaluation, documents = st.tabs(
        [" Hỏi đáp", " Giám sát", " Đánh giá", " Kho văn bản"])
    with chat:
        tab_chat(service, as_of)
    with monitoring:
        tab_monitoring(service)
    with evaluation:
        tab_evaluation(service)
    with documents:
        tab_documents(service, as_of)

    st.divider()
    st.caption(" Đây là công cụ hỗ trợ tra cứu pháp luật, không thay thế ý kiến tư vấn "
               "của luật sư hoặc cơ quan nhà nước có thẩm quyền.")


if __name__ == "__main__":
    main()
