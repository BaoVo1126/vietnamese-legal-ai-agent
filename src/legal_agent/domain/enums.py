from __future__ import annotations
from enum import StrEnum


class DocumentType(StrEnum):
    HIEN_PHAP = "hien_phap"         
    BO_LUAT = "bo_luat"               
    LUAT = "luat"                   
    PHAP_LENH = "phap_lenh"           
    NGHI_QUYET = "nghi_quyet"        
    NGHI_DINH = "nghi_dinh"          
    QUYET_DINH = "quyet_dinh"        
    THONG_TU = "thong_tu"             
    CONG_VAN = "cong_van"             
    KHAC = "khac"

    @property
    def display_name(self) -> str:
        return _DOCUMENT_TYPE_LABELS[self]


_DOCUMENT_TYPE_LABELS: dict[DocumentType, str] = {
    DocumentType.HIEN_PHAP: "Hiến pháp",
    DocumentType.BO_LUAT: "Bộ luật",
    DocumentType.LUAT: "Luật",
    DocumentType.PHAP_LENH: "Pháp lệnh",
    DocumentType.NGHI_QUYET: "Nghị quyết",
    DocumentType.NGHI_DINH: "Nghị định",
    DocumentType.QUYET_DINH: "Quyết định",
    DocumentType.THONG_TU: "Thông tư",
    DocumentType.CONG_VAN: "Công văn",
    DocumentType.KHAC: "Văn bản",
}

ISSUER_SUFFIX_TO_TYPE: dict[str, DocumentType] = {
    "QH": DocumentType.LUAT,          
    "UBTVQH": DocumentType.PHAP_LENH,
    "NĐ-CP": DocumentType.NGHI_DINH,  
    "NQ-CP": DocumentType.NGHI_QUYET,
    "QĐ-TTG": DocumentType.QUYET_DINH,
    "TT-BTC": DocumentType.THONG_TU,
    "TT-BKHĐT": DocumentType.THONG_TU,
    "TT": DocumentType.THONG_TU,
}


class EffectStatus(StrEnum):
    CON_HIEU_LUC = "con_hieu_luc"          
    HET_HIEU_LUC = "het_hieu_luc"          
    HET_HIEU_LUC_MOT_PHAN = "het_hieu_luc_mot_phan"  
    CHUA_CO_HIEU_LUC = "chua_co_hieu_luc"   
    KHONG_XAC_DINH = "khong_xac_dinh"      

    @property
    def is_citable(self) -> bool:
        return self in {EffectStatus.CON_HIEU_LUC, EffectStatus.HET_HIEU_LUC_MOT_PHAN}

    @property
    def display_name(self) -> str:
        return _EFFECT_LABELS[self]


_EFFECT_LABELS: dict[EffectStatus, str] = {
    EffectStatus.CON_HIEU_LUC: "còn hiệu lực",
    EffectStatus.HET_HIEU_LUC: "hết hiệu lực",
    EffectStatus.HET_HIEU_LUC_MOT_PHAN: "hết hiệu lực một phần",
    EffectStatus.CHUA_CO_HIEU_LUC: "chưa có hiệu lực",
    EffectStatus.KHONG_XAC_DINH: "chưa xác định hiệu lực",
}


class RelationType(StrEnum):
    THAY_THE = "THAY_THE"          
    BI_THAY_THE_BOI = "BI_THAY_THE_BOI"
    SUA_DOI = "SUA_DOI"           
    BI_SUA_DOI_BOI = "BI_SUA_DOI_BOI"
    HUONG_DAN = "HUONG_DAN"       
    DUOC_HUONG_DAN_BOI = "DUOC_HUONG_DAN_BOI"
    BAI_BO = "BAI_BO"              
    BI_BAI_BO_BOI = "BI_BAI_BO_BOI"
    CAN_CU = "CAN_CU"             
    DAN_CHIEU = "DAN_CHIEU"       
    THUOC_VE = "THUOC_VE"         

    @property
    def inverse(self) -> RelationType:
        return _INVERSE_RELATIONS.get(self, self)


_INVERSE_RELATIONS: dict[RelationType, RelationType] = {
    RelationType.THAY_THE: RelationType.BI_THAY_THE_BOI,
    RelationType.BI_THAY_THE_BOI: RelationType.THAY_THE,
    RelationType.SUA_DOI: RelationType.BI_SUA_DOI_BOI,
    RelationType.BI_SUA_DOI_BOI: RelationType.SUA_DOI,
    RelationType.HUONG_DAN: RelationType.DUOC_HUONG_DAN_BOI,
    RelationType.DUOC_HUONG_DAN_BOI: RelationType.HUONG_DAN,
    RelationType.BAI_BO: RelationType.BI_BAI_BO_BOI,
    RelationType.BI_BAI_BO_BOI: RelationType.BAI_BO,
}


class NodeLevel(StrEnum):
    VAN_BAN = "van_ban"
    PHAN = "phan"
    CHUONG = "chuong"
    MUC = "muc"
    TIEU_MUC = "tieu_muc"
    DIEU = "dieu"
    KHOAN = "khoan"
    DIEM = "diem"

    @property
    def label(self) -> str:
        return _NODE_LEVEL_LABELS[self]


_NODE_LEVEL_LABELS: dict[NodeLevel, str] = {
    NodeLevel.VAN_BAN: "Văn bản",
    NodeLevel.PHAN: "Phần",
    NodeLevel.CHUONG: "Chương",
    NodeLevel.MUC: "Mục",
    NodeLevel.TIEU_MUC: "Tiểu mục",
    NodeLevel.DIEU: "Điều",
    NodeLevel.KHOAN: "Khoản",
    NodeLevel.DIEM: "Điểm",
}

NODE_LEVEL_DEPTH: dict[NodeLevel, int] = {
    NodeLevel.VAN_BAN: 0,
    NodeLevel.PHAN: 1,
    NodeLevel.CHUONG: 2,
    NodeLevel.MUC: 3,
    NodeLevel.TIEU_MUC: 4,
    NodeLevel.DIEU: 5,
    NodeLevel.KHOAN: 6,
    NodeLevel.DIEM: 7,
}


class QueryIntent(StrEnum):
    TRA_CUU_DIEU_KHOAN = "tra_cuu_dieu_khoan"  
    HOI_DAP_KHAI_NIEM = "hoi_dap_khai_niem"   
    THU_TUC_HANH_CHINH = "thu_tuc_hanh_chinh"  
    CHE_TAI_XU_PHAT = "che_tai_xu_phat"         
    HIEU_LUC_VAN_BAN = "hieu_luc_van_ban"       
    SO_SANH_DOI_CHIEU = "so_sanh_doi_chieu"    
    NGOAI_PHAM_VI = "ngoai_pham_vi"             

    @property
    def needs_graph_expansion(self) -> bool:
        return self in {
            QueryIntent.HIEU_LUC_VAN_BAN,
            QueryIntent.SO_SANH_DOI_CHIEU,
            QueryIntent.THU_TUC_HANH_CHINH,
        }
