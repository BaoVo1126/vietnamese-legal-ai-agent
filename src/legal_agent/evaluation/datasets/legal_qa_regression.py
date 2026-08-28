from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..dataset import EvalCase

REGRESSION_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "hp-001-co-quan-quyen-luc", "field": "Hiến pháp",
     "question": "Theo Hiến pháp 2013, cơ quan nào là cơ quan quyền lực nhà nước cao "
                 "nhất của nước Cộng hòa xã hội chủ nghĩa Việt Nam?",
     "expected_citations": ["Điều 69 Hiến pháp năm 2013"],
     "answer_terms": ["Quốc hội"], "tags": ["canary", "hien_phap"]},
    {"case_id": "hp-002-nhiem-ky-quoc-hoi", "field": "Hiến pháp",
     "question": "Nhiệm kỳ của mỗi khóa Quốc hội theo Hiến pháp 2013 là bao nhiêu năm?",
     "expected_citations": ["Điều 71 Hiến pháp năm 2013"],
     "answer_terms": ["năm năm"], "tags": ["hien_phap"]},
    {"case_id": "hp-003-tuoi-bau-cu", "field": "Hiến pháp",
     "question": "Công dân đủ bao nhiêu tuổi có quyền bầu cử và bao nhiêu tuổi có quyền "
                 "ứng cử vào Quốc hội theo Hiến pháp 2013?",
     "expected_citations": ["Điều 27 Hiến pháp năm 2013"],
     "answer_terms": ["đủ mười tám tuổi"], "tags": ["hien_phap"]},
    {"case_id": "hp-004-chu-tich-nuoc", "field": "Hiến pháp",
     "question": "Chủ tịch nước do cơ quan nào bầu ra theo Hiến pháp 2013?",
     "expected_citations": ["Điều 87 Hiến pháp năm 2013"],
     "answer_terms": ["Quốc hội bầu"], "tags": ["hien_phap"]},


    {"case_id": "hs-001-tuoi-chiu-tnhs", "field": "Hình sự",
     "question": "Người từ đủ bao nhiêu tuổi trở lên phải chịu trách nhiệm hình sự về "
                 "mọi tội phạm theo Bộ luật Hình sự 2015?",
     "expected_citations": ["Điều 12 Bộ luật Hình sự 100/2015/QH13"],
     "answer_terms": ["đủ 16 tuổi"], "tags": ["canary", "hinh_su"]},
    {"case_id": "hs-002-hinh-phat-chinh", "field": "Hình sự",
     "question": "Bộ luật Hình sự 2015 quy định những hình phạt chính nào đối với người "
                 "phạm tội?",
     "expected_citations": ["Điều 32 Bộ luật Hình sự 100/2015/QH13"],
     "answer_terms": ["cảnh cáo"], "tags": ["hinh_su"]},
    {"case_id": "hs-003-phong-ve-chinh-dang", "field": "Hình sự",
     "question": "Phòng vệ chính đáng được Bộ luật Hình sự 2015 định nghĩa như thế nào?",
     "expected_citations": ["Điều 22 Bộ luật Hình sự 100/2015/QH13"],
     "answer_terms": ["phòng vệ chính đáng"], "tags": ["hinh_su", "khai_niem"]},
    {"case_id": "hs-004-thoi-hieu-truy-cuu", "field": "Hình sự",
     "question": "Thời hiệu truy cứu trách nhiệm hình sự đối với tội phạm ít nghiêm "
                 "trọng là bao lâu?",
     "expected_citations": ["Điều 27 Bộ luật Hình sự 100/2015/QH13"],
     "answer_terms": ["05 năm"], "tags": ["hinh_su"]},


    {"case_id": "ds-001-nguoi-thanh-nien", "field": "Dân sự",
     "question": "Theo Bộ luật Dân sự 2015, người thành niên là người từ bao nhiêu tuổi?",
     "expected_citations": ["Điều 20 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["đủ mười tám tuổi"], "tags": ["dan_su", "khai_niem"]},
    {"case_id": "ds-002-hang-thua-ke", "field": "Dân sự",
     "question": "Hàng thừa kế thứ nhất theo pháp luật gồm những ai?",
     "expected_citations": ["Điều 651 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["vợ, chồng"], "tags": ["dan_su"]},
    {"case_id": "ds-003-thoi-hieu-hop-dong", "field": "Dân sự",
     "question": "Thời hiệu khởi kiện để yêu cầu Tòa án giải quyết tranh chấp hợp đồng "
                 "là bao lâu?",
     "expected_citations": ["Điều 429 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["03 năm"], "tags": ["dan_su"]},
    {"case_id": "ds-004-nang-luc-hanh-vi", "field": "Dân sự",
     "question": "Người chưa thành niên có năng lực hành vi dân sự như thế nào theo Bộ "
                 "luật Dân sự 2015?",
     "expected_citations": ["Điều 21 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["chưa thành niên"], "tags": ["dan_su"]},

    {"case_id": "ld-001-thoi-gio-lam-viec", "field": "Lao động",
     "question": "Thời giờ làm việc bình thường của người lao động tối đa bao nhiêu giờ "
                 "một ngày và một tuần?",
     "expected_citations": ["Điều 105 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["08 giờ"], "tags": ["lao_dong"]},
    {"case_id": "ld-002-thu-viec", "field": "Lao động",
     "question": "Thời gian thử việc tối đa đối với người lao động là bao lâu?",
     "expected_citations": ["Điều 25 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["thử việc"], "tags": ["lao_dong"]},
    {"case_id": "ld-003-loai-hop-dong", "field": "Lao động",
     "question": "Hợp đồng lao động gồm những loại nào theo Bộ luật Lao động 2019?",
     "expected_citations": ["Điều 20 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["không xác định thời hạn"], "tags": ["lao_dong"]},
    {"case_id": "ld-004-nghi-hang-nam", "field": "Lao động",
     "question": "Người lao động làm việc đủ 12 tháng trong điều kiện bình thường được "
                 "nghỉ hằng năm bao nhiêu ngày?",
     "expected_citations": ["Điều 113 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["12 ngày"], "tags": ["lao_dong"]},
    {"case_id": "ld-005-tuoi-nghi-huu", "field": "Lao động",
     "question": "Tuổi nghỉ hưu của người lao động trong điều kiện lao động bình thường "
                 "được quy định thế nào?",
     "expected_citations": ["Điều 169 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["nghỉ hưu"], "tags": ["lao_dong"]},


    {"case_id": "dn-001-quyen-thanh-lap", "field": "Doanh nghiệp",
     "question": "Những tổ chức, cá nhân nào không có quyền thành lập và quản lý doanh "
                 "nghiệp tại Việt Nam?",
     "expected_citations": ["Điều 17 Luật Doanh nghiệp 59/2020/QH14"],
     "forbidden_citations": ["Điều 18 Luật Doanh nghiệp 68/2014/QH13"],
     "answer_terms": ["cán bộ, công chức"], "tags": ["doanh_nghiep", "version_aware"]},
    {"case_id": "dn-002-thu-tuc-dang-ky", "field": "Doanh nghiệp",
     "question": "Thủ tục đăng ký doanh nghiệp theo Điều 26 Luật Doanh nghiệp "
                 "59/2020/QH14 gồm những phương thức nào?",
     "expected_citations": ["Điều 26 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["qua mạng thông tin điện tử"],
     "tags": ["doanh_nghiep", "exact_citation"]},
    {"case_id": "dn-003-quyen-doanh-nghiep", "field": "Doanh nghiệp",
     "question": "Doanh nghiệp có những quyền gì theo pháp luật hiện hành?",
     "expected_citations": ["Điều 7 Luật Doanh nghiệp 59/2020/QH14"],
     "forbidden_citations": ["Điều 7 Luật Doanh nghiệp 68/2014/QH13"],
     "answer_terms": ["tự do kinh doanh"], "tags": ["doanh_nghiep", "version_aware"]},
    {"case_id": "dn-004-von-dieu-le", "field": "Doanh nghiệp",
     "question": "Vốn điều lệ của doanh nghiệp được hiểu như thế nào?",
     "expected_citations": ["Điều 4 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["vốn điều lệ"], "tags": ["doanh_nghiep", "khai_niem"]},

   
    {"case_id": "hc-001-tuoi-xu-phat", "field": "Hành chính",
     "question": "Người từ bao nhiêu tuổi trở lên bị xử phạt vi phạm hành chính?",
     "expected_citations": ["Điều 5 Luật Xử lý vi phạm hành chính 15/2012/QH13"],
     "answer_terms": ["đủ 14 tuổi"], "tags": ["hanh_chinh"]},
    {"case_id": "hc-002-thoi-hieu-xu-phat", "field": "Hành chính",
     "question": "Thời hiệu xử phạt vi phạm hành chính thông thường là bao lâu?",
     "expected_citations": ["Điều 6 Luật Xử lý vi phạm hành chính 15/2012/QH13"],
     "answer_terms": ["01 năm"], "tags": ["hanh_chinh"]},
    {"case_id": "hc-003-hinh-thuc-xu-phat", "field": "Hành chính",
     "question": "Các hình thức xử phạt vi phạm hành chính gồm những gì?",
     "expected_citations": ["Điều 21 Luật Xử lý vi phạm hành chính 15/2012/QH13"],
     "answer_terms": ["cảnh cáo"], "tags": ["hanh_chinh"]},

  
    {"case_id": "hn-001-tuoi-ket-hon", "field": "Hôn nhân gia đình",
     "question": "Nam nữ đủ bao nhiêu tuổi thì được kết hôn theo Luật Hôn nhân và gia "
                 "đình 2014?",
     "expected_citations": ["Điều 8 Luật Hôn nhân và gia đình 52/2014/QH13"],
     "answer_terms": ["đủ 20 tuổi"], "tags": ["hon_nhan"]},

 
    {"case_id": "kg-001-huong-dan-dieu-26", "field": "Doanh nghiệp",
     "question": "Nghị định nào hướng dẫn Điều 26 của Luật Doanh nghiệp 59/2020/QH14?",
     "expected_citations": ["Nghị định 01/2021/NĐ-CP"],
     "allow_stale_citations": True,
     "expected_status": "answered", "tags": ["multi_hop", "knowledge_graph"]},
    {"case_id": "kg-002-hieu-luc-luat-cu", "field": "Doanh nghiệp",
     "question": "Luật Doanh nghiệp 68/2014/QH13 còn hiệu lực không?",
     "expected_citations": [], "allow_stale_citations": True,
     "expected_status": "answered", "tags": ["version_aware"]},

    {"case_id": "rf-001-ngoai-pham-vi", "field": "-",
     "question": "Cách nấu phở bò ngon nhất là gì?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_scope"]},
    {"case_id": "rf-002-ngoai-corpus", "field": "-",
     "question": "Điều kiện để một công ty được chào bán chứng khoán lần đầu ra công "
                 "chúng theo Luật Chứng khoán là gì?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_corpus"]},
    {"case_id": "rf-003-tu-van-ca-nhan", "field": "-",
     "question": "Tôi nên thuê luật sư nào ở Hà Nội để kiện hàng xóm?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_scope"]},
  
    {"case_id": "ds-005-dieu-kien-giao-dich", "field": "Dân sự",
     "question": "Giao dịch dân sự có hiệu lực khi đáp ứng những điều kiện nào?",
     "expected_citations": ["Điều 117 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["giao dịch dân sự"], "tags": ["dan_su", "liet_ke"]},
    {"case_id": "ds-006-phap-nhan", "field": "Dân sự",
     "question": "Một tổ chức được công nhận là pháp nhân khi có đủ những điều kiện nào?",
     "expected_citations": ["Điều 74 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["pháp nhân"], "tags": ["dan_su", "liet_ke"]},
    {"case_id": "ds-007-quyen-so-huu", "field": "Dân sự",
     "question": "Quyền sở hữu bao gồm những quyền nào?",
     "expected_citations": ["Điều 158 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["chiếm hữu"], "tags": ["dan_su", "liet_ke"]},
    {"case_id": "ds-008-thoi-hieu-thua-ke", "field": "Dân sự",
     "question": "Thời hiệu để người thừa kế yêu cầu chia di sản là bao lâu?",
     "expected_citations": ["Điều 623 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["thời hiệu"], "tags": ["dan_su"]},
    {"case_id": "ds-009-boi-thuong", "field": "Dân sự",
     "question": "Căn cứ phát sinh trách nhiệm bồi thường thiệt hại ngoài hợp đồng là gì?",
     "expected_citations": ["Điều 584 Bộ luật Dân sự 91/2015/QH13"],
     "answer_terms": ["bồi thường"], "tags": ["dan_su"]},

    {"case_id": "ld-006-cham-dut-hdld", "field": "Lao động",
     "question": "Hợp đồng lao động chấm dứt trong những trường hợp nào?",
     "expected_citations": ["Điều 34 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["chấm dứt hợp đồng lao động"], "tags": ["lao_dong", "liet_ke"]},
    {"case_id": "ld-007-quyen-nguoi-lao-dong", "field": "Lao động",
     "question": "Người lao động có những quyền nào theo Bộ luật Lao động?",
     "expected_citations": ["Điều 5 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["người lao động"], "tags": ["lao_dong", "liet_ke"]},
    {"case_id": "ld-008-noi-dung-hdld", "field": "Lao động",
     "question": "Hợp đồng lao động phải có những nội dung chủ yếu nào?",
     "expected_citations": ["Điều 21 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["hợp đồng lao động"], "tags": ["lao_dong", "liet_ke"]},
    {"case_id": "ld-009-lam-them-gio", "field": "Lao động",
     "question": "Thời gian làm thêm giờ được quy định như thế nào?",
     "expected_citations": ["Điều 107 Bộ luật Lao động 45/2019/QH14"],
     "answer_terms": ["làm thêm giờ"], "tags": ["lao_dong"]},

    {"case_id": "hc-004-nguyen-tac-xu-ly", "field": "Hành chính",
     "question": "Việc xử lý vi phạm hành chính phải tuân theo những nguyên tắc nào?",
     "expected_citations": ["Điều 3 Luật Xử lý vi phạm hành chính 15/2012/QH13"],
     "answer_terms": ["nguyên tắc"], "tags": ["hanh_chinh", "liet_ke"]},

    {"case_id": "dn-005-nguoi-dai-dien", "field": "Doanh nghiệp",
     "question": "Người đại diện theo pháp luật của doanh nghiệp được quy định thế nào?",
     "expected_citations": ["Điều 12 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["người đại diện theo pháp luật"], "tags": ["doanh_nghiep"]},
    {"case_id": "dn-006-giai-the", "field": "Doanh nghiệp",
     "question": "Doanh nghiệp bị giải thể trong những trường hợp nào?",
     "expected_citations": ["Điều 207 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["giải thể"], "tags": ["doanh_nghiep", "liet_ke"]},
    {"case_id": "dn-007-quyen-co-dong", "field": "Doanh nghiệp",
     "question": "Cổ đông phổ thông có những quyền nào?",
     "expected_citations": ["Điều 115 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["cổ đông phổ thông"], "tags": ["doanh_nghiep", "liet_ke"]},
    {"case_id": "dn-008-von-cong-ty-co-phan", "field": "Doanh nghiệp",
     "question": "Vốn điều lệ của công ty cổ phần được xác định như thế nào?",
     "expected_citations": ["Điều 112 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["vốn điều lệ"], "tags": ["doanh_nghiep"]},
    {"case_id": "dn-009-hanh-vi-bi-cam", "field": "Doanh nghiệp",
     "question": "Luật Doanh nghiệp nghiêm cấm những hành vi nào?",
     "expected_citations": ["Điều 16 Luật Doanh nghiệp 59/2020/QH14"],
     "answer_terms": ["nghiêm cấm"], "tags": ["doanh_nghiep", "liet_ke"]},

    {"case_id": "hp-005-chinh-phu", "field": "Hiến pháp",
     "question": "Theo Hiến pháp 2013, Chính phủ là cơ quan gì?",
     "expected_citations": ["Điều 94 Hiến pháp năm 2013"],
     "answer_terms": ["cơ quan hành chính nhà nước cao nhất"], "tags": ["hien_phap"]},
    {"case_id": "hp-006-nhiem-vu-quoc-hoi", "field": "Hiến pháp",
     "question": "Quốc hội có những nhiệm vụ và quyền hạn nào theo Hiến pháp 2013?",
     "expected_citations": ["Điều 70 Hiến pháp năm 2013"],
     "answer_terms": ["nhiệm vụ và quyền hạn"], "tags": ["hien_phap", "liet_ke"]},
    {"case_id": "hp-007-toa-an", "field": "Hiến pháp",
     "question": "Theo Hiến pháp 2013, cơ quan nào thực hiện quyền xét xử?",
     "expected_citations": ["Điều 102 Hiến pháp năm 2013"],
     "answer_terms": ["cơ quan xét xử"], "tags": ["hien_phap"]},

    {"case_id": "hs-005-bien-phap-tu-phap", "field": "Hình sự",
     "question": "Các biện pháp tư pháp đối với người phạm tội gồm những gì?",
     "expected_citations": ["Điều 46 Bộ luật Hình sự 100/2015/QH13"],
     "answer_terms": ["biện pháp tư pháp"], "tags": ["hinh_su", "liet_ke"]},
    {"case_id": "hs-006-tinh-tiet-tang-nang", "field": "Hình sự",
     "question": "Những tình tiết nào là tình tiết tăng nặng trách nhiệm hình sự?",
     "expected_citations": ["Điều 52 Bộ luật Hình sự 100/2015/QH13"],
     "answer_terms": ["tăng nặng"], "tags": ["hinh_su", "liet_ke"]},

    {"case_id": "hn-002-tai-san-chung", "field": "Hôn nhân gia đình",
     "question": "Tài sản chung của vợ chồng gồm những gì?",
     "expected_citations": ["Điều 33 Luật Hôn nhân và gia đình 52/2014/QH13"],
     "answer_terms": ["tài sản chung"], "tags": ["hon_nhan", "liet_ke"]},
    {"case_id": "hn-003-thuan-tinh-ly-hon", "field": "Hôn nhân gia đình",
     "question": "Thuận tình ly hôn được giải quyết như thế nào?",
     "expected_citations": ["Điều 55 Luật Hôn nhân và gia đình 52/2014/QH13"],
     "answer_terms": ["thuận tình ly hôn"], "tags": ["hon_nhan"]},

    {"case_id": "kg-003-nghi-dinh-ho-so", "field": "Doanh nghiệp",
     "question": "Hồ sơ đăng ký doanh nghiệp đối với công ty hợp danh gồm những gì?",
     "expected_citations": ["Điều 22 Nghị định 01/2021/NĐ-CP"],
     "allow_stale_citations": True,
     "answer_terms": ["công ty hợp danh"], "tags": ["multi_hop", "liet_ke"]},
    {"case_id": "kg-004-blhs-bi-sua-doi", "field": "Hình sự",
     "question": "Bộ luật Hình sự 2015 đã bị văn bản nào sửa đổi, bổ sung?",
     "expected_citations": ["Điều 1 Luật 12/2017/QH14"],
     "answer_terms": ["sửa đổi"], "tags": ["multi_hop", "version_aware"]},
    {"case_id": "kg-005-thu-tuc-hien-hanh", "field": "Doanh nghiệp",
     "question": "Trình tự, thủ tục đăng ký doanh nghiệp hiện nay được quy định ở đâu?",
     "expected_citations": ["Điều 26 Luật Doanh nghiệp 59/2020/QH14"],
     "forbidden_citations": ["Điều 27 Luật Doanh nghiệp 68/2014/QH13"],
     "answer_terms": ["đăng ký doanh nghiệp"],
     "tags": ["multi_hop", "version_aware"]},

    {"case_id": "rf-004-luat-thue-tncn", "field": "-",
     "question": "Thuế thu nhập cá nhân được tính như thế nào theo Luật Thuế thu nhập "
                 "cá nhân?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_corpus"],
     "note": "Văn bản được gọi đích danh nhưng không có trong kho."},
    {"case_id": "rf-005-so-huu-tri-tue", "field": "-",
     "question": "Thời hạn bảo hộ sáng chế theo Luật Sở hữu trí tuệ là bao lâu?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_corpus"]},
    {"case_id": "rf-006-khong-phai-phap-luat", "field": "-",
     "question": "Ngày mai Hà Nội có mưa không, tôi có nên mang ô đi làm?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_scope"]},
    {"case_id": "rf-007-du-doan-ket-qua", "field": "-",
     "question": "Nếu tôi kiện hàng xóm về tranh chấp đất thì tôi có thắng không?",
     "expected_citations": [], "expected_status": "refused",
     "tags": ["refusal", "out_of_scope"],
     "note": "Dự đoán kết quả vụ việc không phải tra cứu quy phạm."},

)


def as_eval_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    for index, raw in enumerate(REGRESSION_CASES):
        cases.append(EvalCase(
            case_id=raw["case_id"],
            question=raw["question"],
            expected_citations=list(raw.get("expected_citations", [])),
            forbidden_citations=list(raw.get("forbidden_citations", [])),
            expected_status=raw.get("expected_status", "answered"),
            expected_intent=raw.get("expected_intent", ""),
            allow_stale_citations=bool(raw.get("allow_stale_citations", False)),
            tags=[*raw.get("tags", []), f"field:{raw.get('field', '-')}"],
            note=raw.get("note", ""),
        ))
        _ = index
    return cases


def export_jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for raw in REGRESSION_CASES:
            record = {key: value for key, value in raw.items()
                      if key not in {"field", "answer_terms"}}
            record["tags"] = [*raw.get("tags", []), f"field:{raw.get('field', '-')}"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def answer_terms(case_id: str) -> tuple[str, ...]:
    for raw in REGRESSION_CASES:
        if raw["case_id"] == case_id:
            return tuple(raw.get("answer_terms", []))
    return ()
