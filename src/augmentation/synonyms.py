"""Từ điển đồng nghĩa cho tiếng Việt - bao gồm teencode GenZ"""

SYNONYMS = {
    'tốt': ['giỏi', 'xuất_sắc', 'ổn', 'tuyệt', 'ngon', 'xịn'],
    'xấu': ['tệ', 'kém', 'dở', 'tồi', 'tệ_hại'],
    'người': ['con_người', 'cá_nhân', 'nhân_vật'],
    'nói': ['bảo', 'phát_biểu', 'kêu', 'trình_bày', 'chia_sẻ'],
    'ghét': ['căm', 'thù', 'hận', 'không_ưa', 'khinh'],
    'đánh': ['đấm', 'đập', 'tấn_công', 'ra_đòn'],
    'đẹp': ['xinh', 'ưa_nhìn', 'kiều_diễm', 'xinh_xắn', 'bảnh'],
    'thích': ['ưa', 'mê', 'khoái', 'yêu_thích', 'hứng'],
    'yêu': ['quý', 'mến', 'thương', 'si_mê'],
    'biết': ['hiểu', 'nắm', 'rõ', 'thông_thuộc'],
    
    'ngu': ['ngốc', 'khờ', 'dốt', 'đần', 'ngu_ngốc', 'dốt_nát', 'đần_độn'],
    'điên': ['khùng', 'mất_trí', 'loạn', 'điên_rồ', 'rồ_dại', 'tâm_thần'],
    'khốn': ['khốn_nạn', 'đê_tiện', 'xấu_xa', 'tồi_tệ', 'bỉ_ổi', 'vô_lương'],
    'chết': ['chết_tiệt', 'đi_đời', 'tiêu_đời', 'chết_mẹ', 'toi_đời'],
    'chửi': ['mắng', 'nguyền_rủa', 'đay_nghiến', 'sỉ_vả', 'phỉ_báng'],
    'xúc_phạm': ['sỉ_nhục', 'lăng_mạ', 'miệt_thị', 'xỉa_xói'],
    'đe_dọa': ['dọa', 'hăm_dọa', 'uy_hiếp', 'đe_nẹt'],
    'phân_biệt': ['kỳ_thị', 'phân_biệt_đối_xử', 'kì_thị'],
    'căm': ['thù', 'hận', 'căm_thù', 'căm_hận', 'phẫn_nộ'],
    'súc_vật': ['thú_vật', 'cầm_thú', 'không_ra_gì'],
    
    'vcl': ['vãi_lúa', 'vãi_cả', 'quá', 'rất', 'cực_kỳ'],
    'đmm': ['địt_mẹ_mày', 'đm', 'đ*t_mẹ'],
    'cmm': ['cảm_mến', 'thương', 'quý', 'yêu'],
    'sml': ['sóc_mẹ_luôn', 'chết_mẹ', 'xỉu', 'hết_hồn'],
    'vl': ['vãi_lúa', 'rất', 'quá', 'cực_kỳ'],
    'vãi': ['quá', 'rất', 'cực', 'kinh'],
    
    'mày': ['bạn', 'cậu', 'bồ', 'cu', 'mày_ạ'],
    'tao': ['tôi', 'mình', 'tớ', 'tau'],
    'con': ['nhóc', 'trẻ_con', 'bé'],
    'thằng': ['gã', 'chàng', 'anh_ta'],
    'con_kia': ['cô_nàng', 'ả', 'mụ'],
    
    'cc': ['c*t', 'gì', 'quần_què', 'cặc'],
    'địt': ['chửi', 'mắng', 'xúc_phạm'],
    'cứt': ['phân', 'rác_rưởi', 'vớ_vẩn'],
    'khốn_nạn': ['đê_tiện', 'bỉ_ổi', 'vô_lương_tâm'],
    'đồ_ngu': ['đồ_ngốc', 'đồ_dốt', 'đồ_đần', 'ngu_xuẩn'],
    
    'đỉnh': ['tuyệt', 'xuất_sắc', 'quá_hay', 'top', 'đỉnh_cao'],
    'xịn': ['tốt', 'chất_lượng', 'ok', 'chuẩn', 'ngon'],
    'chuẩn': ['đúng', 'chính_xác', 'đúng_đắn'],
    'ngon': ['tốt', 'xuất_sắc', 'tuyệt_vời'],
    'gắt': ['căng', 'quá', 'rất', 'kinh'],
    'kịch': ['cực', 'tột_cùng', 'tối_đa'],
    
    'cày': ['làm_việc_chăm_chỉ', 'nỗ_lực', 'cần_cù'],
    'lướt': ['xem', 'duyệt', 'đọc'],
    'chat': ['trò_chuyện', 'nhắn_tin', 'nói_chuyện'],
    'like': ['thích', 'ủng_hộ', 'đồng_ý'],
    'share': ['chia_sẻ', 'lan_tỏa', 'đăng'],
    'follow': ['theo_dõi', 'quan_tâm', 'đăng_ký'],
    'tag': ['gắn_thẻ', 'đề_cập', 'nhắc_đến'],
    'trend': ['xu_hướng', 'hot', 'thịnh_hành'],
    
    'hum': ['hôm_nay', 'ngày_hôm_nay'],
    'sao': ['thế_nào', 'ra_sao', 'tại_sao'],
    'zui': ['vui', 'vui_vẻ', 'hứng_thú'],
    'bùn': ['buồn', 'chán', 'thất_vọng'],
    'ngu': ['ngu_ngốc', 'dốt', 'đần'],
    'iu': ['yêu', 'quý', 'mến'],
    'iu_iu': ['yêu_yêu', 'thương_thương'],
    'hum_nay': ['hôm_nay', 'ngày_này'],
    'mai_mốt': ['ngày_mai', 'tương_lai', 'sau_này'],
    
    'k': ['không', 'hông', 'ko'],
    'ko': ['không', 'hông', 'k'],
    'hok': ['không', 'ko', 'k'],
    'dc': ['được', 'đc', 'ok'],
    'đc': ['được', 'dc', 'ok'],
    'oke': ['ok', 'được', 'ổn', 'đồng_ý'],
    'ok': ['được', 'ổn', 'đồng_ý'],
    '2k': ['teen', 'thế_hệ_mới', 'genz'],
    'gấu': ['người_yêu', 'bạn_trai', 'bạn_gái'],
}

EMOTION_SYNONYMS = {
    'tích_cực': {
        'vui': ['hạnh_phúc', 'sung_sướng', 'phấn_khởi', 'hân_hoan'],
        'hạnh_phúc': ['sung_sướng', 'toại_nguyện', 'mãn_nguyện'],
        'tuyệt': ['tuyệt_vời', 'xuất_sắc', 'đỉnh', 'ngon'],
    },
    'tiêu_cực': {
        'buồn': ['chán', 'thất_vọng', 'sầu_khổ', 'đau_khổ'],
        'tức': ['giận', 'bực', 'cáu', 'phẫn_nộ'],
        'ghét': ['căm_thù', 'hận', 'khinh', 'khó_ưa'],
        'sợ': ['lo_sợ', 'sợ_hãi', 'khiếp_sợ', 'ám_ảnh'],
    }
}

def build_synonym_dict():
    """Trả về từ điển đồng nghĩa"""
    return SYNONYMS

def get_synonyms(word: str) -> list[str]:
    """Lấy từ đồng nghĩa của một từ"""
    return SYNONYMS.get(word, [])

def add_synonym(word: str, synonyms: list[str]):
    """Thêm từ đồng nghĩa mới"""
    if word in SYNONYMS:
        SYNONYMS[word].extend(synonyms)
        SYNONYMS[word] = list(set(SYNONYMS[word]))
    else:
        SYNONYMS[word] = synonyms