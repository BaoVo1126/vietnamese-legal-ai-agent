# Văn bản mẫu viết tay (fixtures)

Ba file này là **trích đoạn tự soạn** dùng làm fixture cho test parser ở giai đoạn đầu,
không phải văn bản đầy đủ. Chúng đã được thay bằng bản đầy đủ tải từ corpus
`th1nhng0/vietnamese-legal-documents` (xem `scripts/ingest_priority.py`).

Giữ lại vì hai lý do:

- test parser chạy offline được, không phụ thuộc mạng;
- đối chiếu: metadata tự đoán từ trích đoạn cho kết quả **sai** ở Nghị định 01/2021/NĐ-CP
  (file mẫu ghi "còn hiệu lực", thực tế corpus ghi "Hết hiệu lực toàn bộ"). Đây là bằng
  chứng cụ thể cho việc metadata phải lấy từ nguồn chính thống thay vì suy đoán.
