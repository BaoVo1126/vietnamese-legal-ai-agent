from __future__ import annotations
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.agents import LegalAgentService
from legal_agent.agents.service import AgentAnswer
from legal_agent.config import get_settings
from legal_agent.logging_config import setup_logging
from legal_agent.storage import ConversationStore, group_by_day

STATUS_LABELS = {
    "con_hieu_luc": "còn hiệu lực",
    "het_hieu_luc": "hết hiệu lực",
    "het_hieu_luc_mot_phan": "hết hiệu lực một phần",
    "chua_co_hieu_luc": "chưa có hiệu lực",
    "khong_xac_dinh": "chưa xác định hiệu lực",
}

SAMPLE_QUESTIONS = [
    "Ai không có quyền thành lập và quản lý doanh nghiệp tại Việt Nam?",
    "Người từ đủ bao nhiêu tuổi phải chịu trách nhiệm hình sự về mọi tội phạm?",
    "Nghị định nào hướng dẫn Điều 26 của Luật Doanh nghiệp 59/2020/QH14?",
    "Luật Doanh nghiệp 68/2014/QH13 còn hiệu lực không?",
    "Hợp đồng lao động chấm dứt trong những trường hợp nào?",
]

VIEWS = ("Hỏi đáp", "Giám sát", "Đánh giá", "Kho văn bản")

STYLES = """
<style>
  :root {
    /* Đúng màu nhấn mà theme Streamlit đang dùng. Bản teal sẫm hơn chỉ đạt khoảng
       3.3:1 trên nền tối, dưới ngưỡng WCAG AA cho chữ nhỏ. */
    --accent: #4fb3a5;
    --radius-bubble: 14px;
    --radius-control: 8px;
    --hairline: rgba(128, 128, 128, 0.22);
    --surface-user: rgba(128, 128, 128, 0.12);
    --surface-quiet: rgba(128, 128, 128, 0.07);
  }

  /* Vùng chính rộng vừa mắt đọc, chừa đáy cho thanh nhập cố định. */
  .main .block-container {
    max-width: 820px;
    padding-top: 2.2rem;
    padding-bottom: 7rem;
  }

  /* Bỏ avatar: khung chat không cần hình đại diện khi chỉ có hai bên nói chuyện. */
  [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
  }

  [data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.15rem 0;
    gap: 0;
  }

  /* Câu hỏi của người dùng: canh phải, có nền, không quá rộng để còn ra dáng lượt nói. */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    justify-content: flex-end;
  }

  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
    background: var(--surface-user);
    border-radius: var(--radius-bubble);
    padding: 0.7rem 1rem;
    max-width: 76%;
    width: fit-content;
    /* margin-left tự động đẩy sang phải bất kể hàng là flex hay block; chỉ dựa vào
       justify-content thì hỏng khi Streamlit đổi kiểu container. */
    margin-left: auto;
    margin-right: 0;
  }

  /* Câu trả lời: canh trái, không bọc nền, để phần trích dẫn dài dễ đọc. */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] {
    max-width: 100%;
    padding: 0.2rem 0 0.5rem 0;
  }

  /* Một hiệu ứng duy nhất: tin nhắn mới hiện lên. */
  @media (prefers-reduced-motion: no-preference) {
    [data-testid="stChatMessage"] {
      animation: message-in 180ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes message-in {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  }

  /* Thanh nhập cố định ở đáy. */
  [data-testid="stChatInput"] textarea {
    border-radius: var(--radius-control);
  }

  /* Nhãn nhỏ đi kèm câu trả lời: trạng thái, số hiệu, chỉ số. */
  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.25rem 0 0.55rem 0;
  }

  .meta-item {
    font-size: 0.74rem;
    line-height: 1.5;
    padding: 0.12rem 0.5rem;
    border: 1px solid var(--hairline);
    border-radius: var(--radius-control);
    opacity: 0.85;
  }

  .meta-item.is-refused {
    border-color: var(--accent);
    color: var(--accent);
  }

  .evidence-cite {
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.1rem;
  }

  .evidence-body {
    font-size: 0.82rem;
    line-height: 1.55;
    opacity: 0.82;
    white-space: pre-wrap;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--hairline);
    margin-bottom: 0.7rem;
  }

  .disclaimer {
    font-size: 0.76rem;
    line-height: 1.5;
    opacity: 0.62;
    border-top: 1px solid var(--hairline);
    padding-top: 0.7rem;
    margin-top: 0.4rem;
  }

  /* Sidebar: điều hướng dạng danh sách phẳng, không viền hộp. */
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: 0.15rem; }

  /* Chữ trong sidebar to hơn một nhịp so với mặc định: đây là vùng điều hướng chính,
     người dùng quét mắt qua nó liên tục. */
  [data-testid="stSidebar"] { font-size: 1.02rem; }
  [data-testid="stSidebar"] .stRadio label p { font-size: 1.02rem; }

  /* Tên trang. Đặt ở đầu sidebar và đủ lớn để đóng vai nhãn hiệu, không phải một dòng
     tiêu đề phụ. */
  .wordmark {
    font-size: 1.62rem;
    font-weight: 650;
    line-height: 1.18;
    letter-spacing: -0.02em;
    margin: 0.1rem 0 0.35rem 0;
  }

  .wordmark-sub {
    font-size: 0.86rem;
    line-height: 1.5;
    opacity: 0.6;
    margin-bottom: 0.2rem;
  }

  /* Nhãn nhóm ngày trong danh sách hội thoại. */
  .history-day {
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    opacity: 0.55;
    margin: 0.75rem 0 0.2rem 0;
  }

  /* Mục hội thoại: nút tràn chiều rộng, chữ canh trái, nền chỉ hiện khi rê chuột. */
  [data-testid="stSidebar"] .stButton button {
    text-align: left;
    justify-content: flex-start;
    font-weight: 400;
    border: none;
    background: transparent;
    padding: 0.34rem 0.5rem;
    border-radius: var(--radius-control);
    transition: background 120ms ease;
  }

  [data-testid="stSidebar"] .stButton button:hover {
    background: var(--surface-user);
  }

  [data-testid="stSidebar"] .stButton button p {
    font-size: 0.95rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Nút mở hội thoại mới: giữ viền để tách khỏi danh sách bên dưới. */
  [data-testid="stSidebar"] .st-key-new-chat .stButton button {
    border: 1px solid var(--hairline);
    margin-bottom: 0.35rem;
  }

  /* Giờ đặt câu hỏi, bám sát ngay dưới tiêu đề hội thoại. */
  .history-time {
    font-size: 0.74rem;
    opacity: 0.42;
    margin: -0.55rem 0 0.35rem 0.55rem;
  }

  /* Nút xoá chỉ hiện khi rê chuột vào dòng, để danh sách không bị nhiễu. Nó vẫn hiện
     khi được focus bằng bàn phím và luôn hiện trên thiết bị cảm ứng: ẩn theo hover là
     một cách làm loại người dùng không có chuột ra khỏi tính năng. */
  [data-testid="stSidebar"] [class*="st-key-del-"] {
    opacity: 0;
    transition: opacity 120ms ease;
  }

  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover
    [class*="st-key-del-"],
  [data-testid="stSidebar"] [class*="st-key-del-"]:focus-within {
    opacity: 1;
  }

  @media (hover: none) {
    [data-testid="stSidebar"] [class*="st-key-del-"] { opacity: 1; }
  }

  [data-testid="stSidebar"] [class*="st-key-del-"] .stButton button p {
    font-size: 0.82rem;
    opacity: 0.65;
  }

  /* Dòng hỏi xác nhận trước khi xoá. */
  .history-confirm {
    font-size: 0.86rem;
    line-height: 1.45;
    padding: 0.1rem 0.5rem 0.35rem 0.5rem;
  }

  /* Nút xác nhận xoá: mang màu nhấn để tách khỏi nút huỷ ngay cạnh. */
  [data-testid="stSidebar"] [class*="st-key-confirm-"] .stButton button {
    border: 1px solid var(--accent);
    justify-content: center;
    text-align: center;
  }

  [data-testid="stSidebar"] [class*="st-key-confirm-"] .stButton button p {
    color: var(--accent);
    font-weight: 500;
  }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_store() -> ConversationStore:
    return ConversationStore(get_settings().abs_conversation_db_path)


def start_new_conversation() -> None:
    st.session_state.history = []
    st.session_state.conversation_id = None
    st.session_state.pending_delete = None


def open_conversation(conversation_id: str) -> None:
    store = load_store()
    history: list[dict] = []
    pending_question: str | None = None
    for message in store.messages(conversation_id):
        if message.role == "user":
            pending_question = message.content
        elif pending_question is not None:
            history.append({
                "question": pending_question,
                "answer": AgentAnswer(**message.payload) if message.payload
                else AgentAnswer(question=pending_question, answer=message.content,
                                 status="answered"),
            })
            pending_question = None
    st.session_state.history = history
    st.session_state.conversation_id = conversation_id
    st.session_state.pending_delete = None


def persist_turn(question: str, answer) -> None:
    store = load_store()
    conversation_id = st.session_state.get("conversation_id")
    if not conversation_id:
        conversation_id = store.create(question)
        st.session_state.conversation_id = conversation_id
    store.append(conversation_id, "user", question)
    store.append(conversation_id, "assistant", answer.answer, asdict(answer))


@st.cache_resource(show_spinner=False)
def load_service() -> LegalAgentService:
    setup_logging("WARNING")
    service = LegalAgentService(get_settings())
    service.bootstrap()
    return service


def status_label(effect_status: str) -> str:
    return STATUS_LABELS.get(effect_status, STATUS_LABELS["khong_xac_dinh"])


def scroll_to_latest() -> None:
    components.html(
        "<script>const d=window.parent.document;"
        "d.scrollingElement.scrollTo({top:d.scrollingElement.scrollHeight,"
        "behavior:'smooth'});</script>",
        height=0,
    )


def render_answer(answer, show_disclaimer: bool = True) -> None:
    meta = [
        f"grounding {answer.grounding_score:.2f}",
        f"support {answer.support_ratio:.2f}",
        f"{len(answer.evidence)} bằng chứng",
        f"{answer.latency_ms:,.0f} ms",
    ]
    if answer.attempts > 1:
        meta.append(f"{answer.attempts} lượt truy xuất")
    classes = "meta-item is-refused" if answer.status == "refused" else "meta-item"
    head = f'<span class="{classes}">' + (
        "không đủ căn cứ" if answer.status == "refused" else "có căn cứ") + "</span>"
    st.markdown(
        '<div class="meta-row">' + head
        + "".join(f'<span class="meta-item">{item}</span>' for item in meta)
        + "</div>",
        unsafe_allow_html=True,
    )

   
    body, _, disclaimer = answer.answer.rpartition("Lưu ý:")
    if not body:
        body, disclaimer = answer.answer, ""
    body = body.split("**Căn cứ pháp lý đã đối chiếu:**")[0]
    body = body.split("**Ghi chú hiệu lực (Knowledge Graph):**")[0]
    st.markdown(body.strip())
    if show_disclaimer and disclaimer.strip():
        st.markdown(f'<div class="disclaimer">Lưu ý: {disclaimer.strip()}</div>',
                    unsafe_allow_html=True)

    if answer.graph_notes:
        with st.expander("Ghi chú hiệu lực từ Knowledge Graph"):
            for note in answer.graph_notes:
                st.markdown(f"- {note}")

    if answer.excluded_chunks:
        with st.expander(f"Điều khoản bị loại vì hiệu lực ({len(answer.excluded_chunks)})"):
            for item in answer.excluded_chunks:
                st.markdown(f"- **{item['citation']}**: {item['reason']}")

    if answer.evidence:
        with st.expander(f"Bằng chứng đã truy xuất ({len(answer.evidence)})"):
            for index, item in enumerate(answer.evidence, start=1):
                st.markdown(
                    f'<div class="evidence-cite">{index}. {item["citation"]}</div>'
                    f'<div class="meta-row">'
                    f'<span class="meta-item">{status_label(item["effect_status"])}</span>'
                    f'<span class="meta-item">{item["source"]}</span>'
                    f'<span class="meta-item">score {item["score"]:.4f}</span>'
                    f"</div>"
                    f'<div class="evidence-body">{item["text"][:900]}</div>',
                    unsafe_allow_html=True,
                )

    if answer.trace:
        with st.expander("Trace của LangGraph"):
            st.dataframe(
                pd.DataFrame([
                    {"bước": index, "node": entry.get("node", ""),
                     "ms": entry.get("elapsed_ms", 0.0)}
                    for index, entry in enumerate(answer.trace, start=1)
                ]),
                use_container_width=True, hide_index=True,
            )


def view_chat(service: LegalAgentService, as_of: str | None) -> None:
    if not st.session_state.history:
        st.markdown(
            "Đặt câu hỏi về các văn bản đang có trong kho. Mỗi câu trả lời chỉ dựa trên "
            "điều khoản truy xuất được; không đủ căn cứ thì hệ thống từ chối thay vì "
            "suy diễn."
        )

    last_index = len(st.session_state.history) - 1
    for index, entry in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry["answer"], show_disclaimer=index == last_index)
    if st.session_state.history:
        scroll_to_latest()

    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Nhập câu hỏi pháp luật") or pending
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Đang truy xuất và đối chiếu hiệu lực"):
        started = time.perf_counter()
        answer = service.ask(question, session_id="streamlit", as_of=as_of)
        if not answer.latency_ms:
            answer.latency_ms = (time.perf_counter() - started) * 1000

    st.session_state.history.append({"question": question, "answer": answer})
    persist_turn(question, answer)
    st.rerun()


def view_monitoring(service: LegalAgentService) -> None:
    st.subheader("Giám sát")
    summary = service.metrics()
    if not summary.total_runs:
        st.info("Chưa có lượt hỏi nào được ghi nhận.")
        return

    columns = st.columns(4)
    columns[0].metric("Tổng lượt hỏi", summary.total_runs)
    columns[1].metric("Tỷ lệ từ chối", f"{summary.refusal_rate:.0%}",
                      help="Giảm về 0 không phải tin vui: thường là chốt chặn đã ngừng "
                           "bắt lỗi")
    columns[2].metric("Tỷ lệ tự sửa", f"{summary.retry_rate:.0%}",
                      help="Phản ánh chất lượng truy xuất ở lượt đầu")
    columns[3].metric("Độ trễ p95", f"{summary.latency_p95_ms:,.0f} ms")

    columns = st.columns(4)
    columns[0].metric("Trả lời được", summary.answered)
    columns[1].metric("Grounding trung bình", f"{summary.avg_grounding:.2f}")
    columns[2].metric("Support trung bình", f"{summary.avg_support:.2f}")
    columns[3].metric("Độ trễ p50", f"{summary.latency_p50_ms:,.0f} ms")

    left, right = st.columns(2)
    with left:
        st.caption("Độ trễ trung bình theo node")
        if summary.node_latency_ms:
            st.bar_chart(pd.Series(summary.node_latency_ms, name="ms"))
    with right:
        st.caption("Phân bố intent")
        if summary.intents:
            st.bar_chart(pd.Series(summary.intents, name="lượt"))

    st.caption("Nhật ký gần nhất")
    records = service.recorder.read_all(limit=25)[::-1]
    if records:
        st.dataframe(
            pd.DataFrame(records)[["timestamp", "question", "status", "intent",
                                   "grounding_score", "support_ratio", "latency_ms"]],
            use_container_width=True, hide_index=True,
        )


def view_evaluation(service: LegalAgentService) -> None:
    from legal_agent.evaluation import EvaluationRunner
    from legal_agent.evaluation.datasets import as_eval_cases

    st.subheader("Đánh giá")
    cases = as_eval_cases()
    st.caption(
        f"Bộ regression {len(cases)} case, trải nhiều lĩnh vực. Mỗi case gắn kỳ vọng ở "
        "ba tầng: điều khoản phải truy xuất được, điều khoản phải trích dẫn, và hành vi "
        "mong đợi (trả lời hay từ chối)."
    )

    if st.button("Chạy đánh giá", type="primary"):
        progress = st.progress(0.0, text="Đang chạy")
        report = EvaluationRunner(service).run(
            cases,
            progress=lambda index, total, _: progress.progress(
                index / total, text=f"Case {index}/{total}"),
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
                      help="Chỉ số sống còn của sản phẩm pháp lý, phải bằng 0")

    st.dataframe(
        pd.DataFrame([
            {"case": result.case_id,
             "đạt": "đạt" if result.passed else "chưa đạt",
             "kỳ vọng": result.expected_status,
             "thực tế": result.actual_status,
             "retr. recall": result.retrieval_recall,
             "cit. recall": result.citation_recall,
             "thiếu trích dẫn": ", ".join(result.missing_citations) or ""}
            for result in report.results
        ]),
        use_container_width=True, hide_index=True,
    )


def view_documents(service: LegalAgentService, as_of: str | None) -> None:
    st.subheader("Kho văn bản")
    documents = service.graph_store.all_documents()
    st.dataframe(
        pd.DataFrame([
            {"Số hiệu": document.doc_number or "không có số hiệu",
             "Tên văn bản": document.title,
             "Loại": document.doc_type.display_name,
             "Hiệu lực": status_label(document.effect_status.value),
             "Ngày hiệu lực": document.effective_date,
             "Quan hệ": len(document.relations)}
            for document in documents
        ]),
        use_container_width=True, hide_index=True,
    )

    st.caption("Tra cứu hiệu lực")
    left, right = st.columns([2, 1])
    doc_key = left.selectbox("Văn bản",
                             [document.doc_key for document in documents],
                             label_visibility="collapsed")
    dieu = right.text_input("Số Điều", placeholder="Số Điều, ví dụ 26",
                            label_visibility="collapsed")

    if not doc_key:
        return
    verdict = service.graph_store.validate(doc_key, dieu.strip() or None,
                                           service.context.as_of_date(as_of))
    payload = verdict.as_dict()
    st.markdown(f"Trạng thái: **{status_label(payload['status'])}**")
    if payload["replaced_by"]:
        st.markdown(f"Bị thay thế bởi: {', '.join(payload['replaced_by'])}")
    if payload["amended_by"]:
        st.markdown(f"Bị sửa đổi bởi: {', '.join(payload['amended_by'])}")
    if payload["guided_by"]:
        st.markdown("Được hướng dẫn bởi: " + ", ".join(
            entry["doc_number"] + (f" (Điều {entry['dieu']})" if entry.get("dieu") else "")
            for entry in payload["guided_by"]))
    if not any([payload["replaced_by"], payload["amended_by"], payload["guided_by"]]):
        st.caption("Không có quan hệ nào được ghi nhận cho phạm vi đã chọn.")


def render_history(store: ConversationStore) -> None:
    conversations = store.list_recent()
    if not conversations:
        st.caption("Chưa có hội thoại nào được lưu.")
        return

    current = st.session_state.get("conversation_id")
    for label, group in group_by_day(conversations):
        st.markdown(f'<div class="history-day">{label}</div>', unsafe_allow_html=True)
        for conversation in group:
            if st.session_state.get("pending_delete") == conversation.id:
                render_delete_confirm(store, conversation)
            else:
                render_history_row(conversation, current)


def render_history_row(conversation, current: str | None) -> None:
    open_column, delete_column = st.columns([0.82, 0.18], gap="small")
    with open_column:
        mark = "• " if conversation.id == current else ""
        if st.button(f"{mark}{conversation.title}", key=f"conv-{conversation.id}",
                     use_container_width=True,
                     help=f"{conversation.updated_at.strftime('%d/%m/%Y %H:%M')} · "
                          f"{conversation.message_count} lượt nói"):
            open_conversation(conversation.id)
            st.rerun()
    with delete_column, st.container(key=f"del-{conversation.id}"):
        if st.button("Xoá", key=f"del-btn-{conversation.id}",
                     use_container_width=True, help="Xoá hội thoại này"):
            st.session_state.pending_delete = conversation.id
            st.rerun()
    st.markdown(f'<div class="history-time">'
                f'{conversation.updated_at.strftime("%H:%M")}</div>',
                unsafe_allow_html=True)


def render_delete_confirm(store: ConversationStore, conversation) -> None:
    st.markdown(f'<div class="history-confirm">Xoá hội thoại '
                f'<strong>{conversation.title}</strong> cùng '
                f'{conversation.message_count} lượt nói?</div>',
                unsafe_allow_html=True)
    confirm_column, cancel_column = st.columns(2, gap="small")
    with confirm_column, st.container(key=f"confirm-{conversation.id}"):
        if st.button("Xoá", key=f"confirm-btn-{conversation.id}",
                     use_container_width=True):
            store.delete(conversation.id)
            if st.session_state.get("conversation_id") == conversation.id:
                start_new_conversation()
            st.session_state.pending_delete = None
            st.rerun()
    if cancel_column.button("Giữ lại", key=f"cancel-{conversation.id}",
                            use_container_width=True):
        st.session_state.pending_delete = None
        st.rerun()


def render_sidebar(service: LegalAgentService, store: ConversationStore
                   ) -> tuple[str, str | None]:
    settings = get_settings()
    with st.sidebar:
        st.markdown('<div class="wordmark">Trợ lý Pháp luật<br>Việt Nam</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="wordmark-sub">Chỉ trả lời từ điều khoản truy xuất được, '
                    'kèm trích dẫn Điều, Khoản, Điểm và số hiệu văn bản.</div>',
                    unsafe_allow_html=True)

        view = st.radio("Khu vực", VIEWS, label_visibility="collapsed")
        st.divider()

        if view == "Hỏi đáp":
            with st.container(key="new-chat"):
                if st.button("Cuộc trò chuyện mới", use_container_width=True):
                    start_new_conversation()
                    st.rerun()

            with st.expander("Câu hỏi mẫu"):
                for index, question in enumerate(SAMPLE_QUESTIONS):
                    if st.button(question, key=f"sample-{index}",
                                 use_container_width=True):
                        st.session_state.pending_question = question
                        st.rerun()

            render_history(store)
            st.divider()

        use_as_of = st.checkbox("Hỏi theo mốc thời gian khác", value=False,
                                help="Kiểm tra hiệu lực tại ngày chỉ định thay vì hôm nay")
        as_of = st.date_input("Ngày", value=date.today()).isoformat() if use_as_of else None

        st.divider()
        columns = st.columns(2)
        columns[0].metric("Văn bản", len(service.graph_store.all_documents()))
        columns[1].metric("Chunk", f"{service.vector_store.count():,}")
        st.caption(
            f"LLM {settings.llm_backend} · embedding {settings.embedding_backend} · "
            f"Qdrant {settings.qdrant_mode} · graph {settings.graph_backend}"
        )
        if settings.llm_backend == "stub":
            st.caption("Đang chạy mô hình giả lập offline: câu trả lời được ghép từ "
                       "chính điều luật truy xuất được. Đặt LLM_BACKEND thành "
                       "openai_compatible để dùng mô hình thật.")
    return view, as_of


def main() -> None:
    st.set_page_config(page_title="Trợ lý pháp luật Việt Nam", layout="centered",
                       initial_sidebar_state="expanded")
    st.markdown(STYLES, unsafe_allow_html=True)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("pending_delete", None)

    with st.spinner("Đang nạp kho tri thức"):
        service = load_service()

    store = load_store()
    view, as_of = render_sidebar(service, store)
    if view == "Hỏi đáp":
        view_chat(service, as_of)
    elif view == "Giám sát":
        view_monitoring(service)
    elif view == "Đánh giá":
        view_evaluation(service)
    else:
        view_documents(service, as_of)


if __name__ == "__main__":
    main()
