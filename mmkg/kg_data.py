# -*- coding: utf-8 -*-
"""MMKG Ẩm thực Việt Nam — phần tri thức (không gồm ảnh).

Nhóm tự biên soạn (có hỗ trợ của công cụ AI) từ kiến thức phổ thông về món ăn
Việt Nam; đã đối chiếu với các bài Wikipedia tiếng Việt / tiếng Anh của từng món
(xem DATA_CARD.md). Địa danh dùng tên truyền thống, trước đợt sắp xếp tỉnh 2025.

Mỗi món: (id, tên, loại, xuất xứ, [cách chế biến], [nguyên liệu], [từ khoá ảnh])
- xuất xứ là một tỉnh/thành, hoặc một vùng khi món phổ biến khắp vùng và không có
  một nơi gốc được thừa nhận rộng rãi.
- nguyên liệu là các nguyên liệu đặc trưng, không phải công thức đầy đủ.
- từ khoá ảnh (không dấu, chữ thường) dùng để lọc kết quả tìm ảnh trên Wikimedia
  Commons: tiêu đề/mô tả/danh mục của tệp phải chứa ít nhất một từ khoá.
"""

REGIONS = ["Bắc Bộ", "Trung Bộ", "Nam Bộ", "Tây Nguyên"]

PROVINCES = {
    "Hà Nội": "Bắc Bộ", "Hải Phòng": "Bắc Bộ", "Thanh Hóa": "Trung Bộ",
    "Huế": "Trung Bộ", "Đà Nẵng": "Trung Bộ", "Quảng Nam": "Trung Bộ",
    "Bình Định": "Trung Bộ", "Ninh Thuận": "Trung Bộ", "Lâm Đồng": "Tây Nguyên",
    "TP. Hồ Chí Minh": "Nam Bộ", "Bà Rịa – Vũng Tàu": "Nam Bộ", "Cần Thơ": "Nam Bộ",
    "Sóc Trăng": "Nam Bộ",
}

DISH_TYPES = ["món sợi nước", "món sợi trộn", "món cơm và xôi", "bánh mặn",
              "bánh truyền thống", "bánh ngọt", "món cuốn và nem",
              "món kho", "canh và lẩu", "đồ uống", "món ăn vặt", "món nướng"]

METHODS = ["nấu", "luộc", "nướng", "chiên", "hấp", "kho", "xào", "trộn",
           "gói lá", "pha", "lên men"]

INGREDIENT_GROUPS = {
    "tinh bột": ["bánh phở", "bún", "sợi hủ tiếu", "sợi mì gạo", "gạo", "gạo nếp",
                 "bột gạo", "bột năng", "bột mì", "bánh tráng"],
    "thịt và trứng": ["thịt bò", "thịt lợn", "thịt gà", "trứng", "trứng cút"],
    "hải sản": ["tôm", "cua", "cá", "ốc", "hến", "mực"],
    "đậu và hạt": ["đậu xanh", "đậu phụ", "đậu phộng"],
    "rau củ quả": ["giá đỗ", "rau muống", "cà chua", "dứa", "cà tím", "xà lách",
                   "dưa chuột", "cà rốt", "mộc nhĩ", "sầu riêng"],
    "rau thơm và lá": ["rau thơm", "hành lá", "rau răm", "thì là", "lá lốt"],
    "gia vị": ["sả", "gừng", "nghệ", "ớt", "hoa hồi", "me"],
    "mắm và nước chấm": ["nước mắm", "mắm tôm", "mắm ruốc", "mắm cá"],
    "lá gói": ["lá dong", "lá chuối"],
    "sữa và đồ uống": ["nước cốt dừa", "sữa đặc", "cà phê"],
}

D = [
    # id, tên, loại, xuất xứ, cách chế biến, nguyên liệu, từ khoá ảnh
    ("pho_bo", "Phở bò", "món sợi nước", "Hà Nội", ["nấu"],
     ["bánh phở", "thịt bò", "hành lá", "gừng", "hoa hồi", "rau thơm"],
     ["pho bo", "beef pho", "pho tai", "pho chin", "pho ha noi", "pho hanoi"]),
    ("pho_ga", "Phở gà", "món sợi nước", "Hà Nội", ["nấu", "luộc"],
     ["bánh phở", "thịt gà", "hành lá", "gừng"],
     ["pho ga", "chicken pho"]),
    ("bun_cha", "Bún chả", "món sợi trộn", "Hà Nội", ["nướng"],
     ["bún", "thịt lợn", "nước mắm", "ớt", "rau thơm"],
     ["bun cha"]),
    ("bun_bo_hue", "Bún bò Huế", "món sợi nước", "Huế", ["nấu"],
     ["bún", "thịt bò", "thịt lợn", "sả", "mắm ruốc", "ớt", "hành lá"],
     ["bun bo hue", "bun bo"]),
    ("bun_rieu", "Bún riêu cua", "món sợi nước", "Hà Nội", ["nấu"],
     ["bún", "cua", "cà chua", "đậu phụ", "mắm tôm", "hành lá", "rau thơm"],
     ["bun rieu"]),
    ("bun_thit_nuong", "Bún thịt nướng", "món sợi trộn", "TP. Hồ Chí Minh", ["nướng"],
     ["bún", "thịt lợn", "sả", "đậu phộng", "nước mắm", "xà lách", "rau thơm"],
     ["bun thit nuong"]),
    ("bun_dau", "Bún đậu mắm tôm", "món sợi trộn", "Hà Nội", ["chiên"],
     ["bún", "đậu phụ", "mắm tôm", "thịt lợn", "rau thơm", "dưa chuột"],
     ["bun dau"]),
    ("banh_da_cua", "Bánh đa cua", "món sợi nước", "Hải Phòng", ["nấu"],
     ["cua", "rau muống", "lá lốt", "thịt lợn", "hành lá"],
     ["banh da cua"]),
    ("bun_oc", "Bún ốc", "món sợi nước", "Hà Nội", ["nấu", "luộc"],
     ["bún", "ốc", "cà chua", "đậu phụ", "hành lá", "rau thơm"],
     ["bun oc"]),
    ("bun_mam", "Bún mắm", "món sợi nước", "Sóc Trăng", ["nấu"],
     ["bún", "mắm cá", "tôm", "mực", "thịt lợn", "cà tím", "sả"],
     ["bun mam"]),
    ("hu_tieu_nam_vang", "Hủ tiếu Nam Vang", "món sợi nước", "TP. Hồ Chí Minh", ["nấu"],
     ["sợi hủ tiếu", "thịt lợn", "tôm", "trứng cút"],
     ["nam vang", "hu tieu nam vang", "hu tiu nam vang"]),
    ("mi_quang", "Mì Quảng", "món sợi trộn", "Quảng Nam", ["nấu"],
     ["sợi mì gạo", "tôm", "thịt lợn", "nghệ", "đậu phộng", "bánh tráng", "rau thơm"],
     ["mi quang"]),
    ("cao_lau", "Cao lầu", "món sợi trộn", "Quảng Nam", ["luộc", "kho"],
     ["sợi mì gạo", "thịt lợn", "giá đỗ", "rau thơm"],
     ["cao lau"]),
    ("bun_cha_ca", "Bún chả cá", "món sợi nước", "Đà Nẵng", ["chiên", "nấu"],
     ["bún", "cá", "cà chua", "dứa", "hành lá"],
     ["bun cha ca"]),
    ("banh_canh_cua", "Bánh canh cua", "món sợi nước", "TP. Hồ Chí Minh", ["nấu"],
     ["cua", "tôm", "bột năng", "hành lá"],
     ["banh canh"]),
    ("com_tam", "Cơm tấm", "món cơm và xôi", "TP. Hồ Chí Minh", ["nướng"],
     ["gạo", "thịt lợn", "trứng", "nước mắm", "hành lá", "dưa chuột"],
     ["com tam", "broken rice"]),
    ("com_ga", "Cơm gà Hội An", "món cơm và xôi", "Quảng Nam", ["luộc"],
     ["gạo", "thịt gà", "nghệ", "rau răm", "gừng"],
     ["com ga"]),
    ("com_hen", "Cơm hến", "món cơm và xôi", "Huế", ["xào"],
     ["gạo", "hến", "đậu phộng", "mắm ruốc", "ớt", "rau thơm"],
     ["com hen"]),
    ("banh_mi", "Bánh mì", "bánh mặn", "TP. Hồ Chí Minh", ["nướng"],
     ["bột mì", "thịt lợn", "dưa chuột", "cà rốt", "rau thơm", "ớt"],
     ["banh mi"]),
    ("banh_xeo", "Bánh xèo", "bánh mặn", "Nam Bộ", ["chiên"],
     ["bột gạo", "nghệ", "nước cốt dừa", "tôm", "thịt lợn", "giá đỗ", "xà lách", "nước mắm"],
     ["banh xeo"]),
    ("banh_khot", "Bánh khọt", "bánh mặn", "Bà Rịa – Vũng Tàu", ["chiên"],
     ["bột gạo", "tôm", "nước cốt dừa", "hành lá", "xà lách"],
     ["banh khot"]),
    ("banh_cuon", "Bánh cuốn", "bánh mặn", "Hà Nội", ["hấp"],
     ["bột gạo", "thịt lợn", "mộc nhĩ", "nước mắm"],
     ["banh cuon"]),
    ("banh_beo", "Bánh bèo", "bánh mặn", "Huế", ["hấp"],
     ["bột gạo", "tôm", "hành lá", "nước mắm"],
     ["banh beo"]),
    ("banh_bot_loc", "Bánh bột lọc", "bánh mặn", "Huế", ["hấp", "gói lá"],
     ["bột năng", "tôm", "thịt lợn", "lá chuối", "nước mắm"],
     ["bot loc"]),
    ("banh_chung", "Bánh chưng", "bánh truyền thống", "Bắc Bộ", ["luộc", "gói lá"],
     ["gạo nếp", "đậu xanh", "thịt lợn", "lá dong"],
     ["banh chung"]),
    ("banh_tet", "Bánh tét", "bánh truyền thống", "Nam Bộ", ["luộc", "gói lá"],
     ["gạo nếp", "đậu xanh", "thịt lợn", "lá chuối"],
     ["banh tet"]),
    ("banh_can", "Bánh căn", "bánh mặn", "Ninh Thuận", ["nướng"],
     ["bột gạo", "trứng", "tôm", "mực", "hành lá", "nước mắm"],
     ["banh can"]),
    ("banh_hoi", "Bánh hỏi", "bánh mặn", "Bình Định", ["hấp"],
     ["bột gạo", "hành lá", "thịt lợn", "nước mắm"],
     ["banh hoi"]),
    ("banh_it", "Bánh ít", "bánh truyền thống", "Bình Định", ["hấp", "gói lá"],
     ["gạo nếp", "đậu xanh", "lá chuối"],
     ["banh it"]),
    ("banh_trang_nuong", "Bánh tráng nướng", "món ăn vặt", "Lâm Đồng", ["nướng"],
     ["bánh tráng", "trứng cút", "hành lá", "ớt"],
     ["banh trang nuong"]),
    ("banh_pia", "Bánh pía", "bánh ngọt", "Sóc Trăng", ["nướng"],
     ["bột mì", "đậu xanh", "sầu riêng", "trứng"],
     ["banh pia"]),
    ("banh_da_lon", "Bánh da lợn", "bánh ngọt", "Nam Bộ", ["hấp"],
     ["bột năng", "đậu xanh", "nước cốt dừa"],
     ["banh da lon"]),
    ("goi_cuon", "Gỏi cuốn", "món cuốn và nem", "Nam Bộ", ["luộc"],
     ["bánh tráng", "tôm", "thịt lợn", "bún", "rau thơm", "xà lách"],
     ["goi cuon", "summer roll", "fresh spring roll"]),
    ("nem_ran", "Nem rán", "món cuốn và nem", "Hà Nội", ["chiên"],
     ["bánh tráng", "thịt lợn", "mộc nhĩ", "trứng", "cà rốt"],
     ["nem ran", "cha gio", "fried spring roll"]),
    ("cha_ca", "Chả cá Lã Vọng", "món nướng", "Hà Nội", ["nướng", "chiên"],
     ["cá", "nghệ", "thì là", "mắm tôm", "bún", "đậu phộng", "hành lá"],
     ["cha ca la vong", "cha ca"]),
    ("ca_kho_to", "Cá kho tộ", "món kho", "Nam Bộ", ["kho"],
     ["cá", "nước mắm", "ớt", "hành lá"],
     ["ca kho"]),
    ("thit_kho", "Thịt kho trứng", "món kho", "Nam Bộ", ["kho"],
     ["thịt lợn", "trứng", "nước mắm"],
     ["thit kho"]),
    ("canh_chua", "Canh chua cá", "canh và lẩu", "Nam Bộ", ["nấu"],
     ["cá", "me", "dứa", "cà chua", "giá đỗ", "rau thơm"],
     ["canh chua"]),
    ("lau_mam", "Lẩu mắm", "canh và lẩu", "Cần Thơ", ["nấu"],
     ["mắm cá", "cá", "tôm", "mực", "cà tím", "sả", "rau muống"],
     ["lau mam"]),
    ("xoi_xeo", "Xôi xéo", "món cơm và xôi", "Hà Nội", ["hấp"],
     ["gạo nếp", "đậu xanh", "nghệ"],
     ["xoi xeo"]),
    ("ca_phe_sua_da", "Cà phê sữa đá", "đồ uống", "TP. Hồ Chí Minh", ["pha"],
     ["cà phê", "sữa đặc"],
     ["ca phe sua da", "cafe sua da", "iced coffee"]),
    ("ca_phe_trung", "Cà phê trứng", "đồ uống", "Hà Nội", ["pha"],
     ["cà phê", "trứng", "sữa đặc"],
     ["ca phe trung", "egg coffee"]),
    ("nem_chua", "Nem chua", "món cuốn và nem", "Thanh Hóa", ["lên men", "gói lá"],
     ["thịt lợn", "ớt", "lá chuối"],
     ["nem chua"]),
    ("bo_la_lot", "Bò lá lốt", "món nướng", "Nam Bộ", ["nướng"],
     ["thịt bò", "lá lốt", "sả", "đậu phộng"],
     ["la lot"]),
    ("gio_lua", "Giò lụa", "món cuốn và nem", "Bắc Bộ", ["luộc", "gói lá"],
     ["thịt lợn", "nước mắm", "lá chuối"],
     ["gio lua", "cha lua"]),
]

# Tên tiếng Anh ngắn để tìm ảnh và để so sánh với CLIP tiếng Anh.
EN_NAME = {
    "pho_bo": "beef pho noodle soup", "pho_ga": "chicken pho noodle soup",
    "bun_cha": "bun cha grilled pork with rice noodles",
    "bun_bo_hue": "bun bo Hue spicy beef noodle soup",
    "bun_rieu": "bun rieu crab noodle soup",
    "bun_thit_nuong": "bun thit nuong grilled pork vermicelli",
    "bun_dau": "bun dau mam tom fried tofu with noodles and shrimp paste",
    "banh_da_cua": "banh da cua red noodle crab soup",
    "bun_oc": "bun oc snail noodle soup", "bun_mam": "bun mam fermented fish noodle soup",
    "hu_tieu_nam_vang": "hu tieu Nam Vang noodle soup",
    "hu_tieu_my_tho": "hu tieu My Tho noodle soup", "mi_quang": "mi Quang turmeric noodles",
    "cao_lau": "cao lau Hoi An noodles", "bun_cha_ca": "bun cha ca fish cake noodle soup",
    "banh_canh_cua": "banh canh cua thick crab noodle soup",
    "com_tam": "com tam broken rice with grilled pork",
    "com_ga": "com ga Hoi An chicken rice", "com_hen": "com hen baby clam rice",
    "banh_mi": "banh mi Vietnamese sandwich", "banh_xeo": "banh xeo sizzling crepe",
    "banh_khot": "banh khot mini rice pancakes", "banh_cuon": "banh cuon steamed rice rolls",
    "banh_beo": "banh beo steamed rice cakes", "banh_bot_loc": "banh bot loc tapioca dumplings",
    "banh_chung": "banh chung square sticky rice cake", "banh_tet": "banh tet cylindrical sticky rice cake",
    "banh_can": "banh can mini rice pancakes with egg", "banh_hoi": "banh hoi woven rice vermicelli",
    "banh_it": "banh it sticky rice dumpling", "banh_trang_nuong": "banh trang nuong grilled rice paper",
    "banh_trang_tron": "banh trang tron rice paper salad", "banh_pia": "banh pia durian pastry",
    "banh_da_lon": "banh da lon steamed layer cake", "goi_cuon": "goi cuon fresh spring rolls",
    "nem_ran": "nem ran fried spring rolls", "cha_ca": "cha ca La Vong turmeric fish with dill",
    "ca_kho_to": "ca kho to caramelized fish in clay pot",
    "thit_kho": "thit kho braised pork and eggs", "canh_chua": "canh chua sour fish soup",
    "lau_mam": "lau mam fermented fish hot pot", "xoi_xeo": "xoi xeo sticky rice with mung bean",
    "ca_phe_sua_da": "ca phe sua da iced milk coffee", "ca_phe_trung": "ca phe trung egg coffee",
    "nem_chua": "nem chua fermented pork roll", "bo_la_lot": "bo la lot beef wrapped in betel leaves",
    "gio_lua": "gio lua Vietnamese pork sausage",
}

REL = {
    "type": "thuộc_loại", "origin": "xuất_xứ", "method": "chế_biến_bằng",
    "ingredient": "có_nguyên_liệu", "group": "thuộc_nhóm", "region": "thuộc_vùng",
    "depicts": "mô_tả",
}


def build_kg():
    """Trả về (entities, triples). entities: list dict(id, name, kind)."""
    import unicodedata, re

    def slug(s):
        s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
        return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

    ents, seen = [], set()

    def add(eid, name, kind):
        if eid in seen:
            return eid
        seen.add(eid)
        ents.append(dict(id=eid, name=name, kind=kind))
        return eid

    for r in REGIONS:
        add("reg_" + slug(r), r, "vùng")
    for p, r in PROVINCES.items():
        add("prov_" + slug(p), p, "tỉnh")
    for t in DISH_TYPES:
        add("type_" + slug(t), t, "loại món")
    for m in METHODS:
        add("meth_" + slug(m), m, "cách chế biến")
    ing2grp = {}
    for g, items in INGREDIENT_GROUPS.items():
        add("grp_" + slug(g), g, "nhóm nguyên liệu")
        for it in items:
            assert it not in ing2grp, it
            ing2grp[it] = g
    for d in D:
        add("dish_" + d[0], d[1], "món")

    tri = []
    used_ing = set()
    for (did, name, typ, origin, meths, ings, _) in D:
        h = "dish_" + did
        assert typ in DISH_TYPES, typ
        tri.append((h, REL["type"], "type_" + slug(typ)))
        if origin in PROVINCES:
            tri.append((h, REL["origin"], "prov_" + slug(origin)))
        else:
            assert origin in REGIONS, origin
            tri.append((h, REL["origin"], "reg_" + slug(origin)))
        for m in meths:
            assert m in METHODS, m
            tri.append((h, REL["method"], "meth_" + slug(m)))
        for it in ings:
            assert it in ing2grp, it
            used_ing.add(it)
            tri.append((h, REL["ingredient"], "ing_" + slug(it)))
    for it, g in ing2grp.items():
        if it in used_ing:
            add("ing_" + slug(it), it, "nguyên liệu")
            tri.append(("ing_" + slug(it), REL["group"], "grp_" + slug(g)))
    for p, r in PROVINCES.items():
        tri.append(("prov_" + slug(p), REL["region"], "reg_" + slug(r)))
    # bỏ thực thể không có cạnh nào
    deg = {}
    for h, _, t in tri:
        deg[h] = deg.get(h, 0) + 1
        deg[t] = deg.get(t, 0) + 1
    ents = [e for e in ents if deg.get(e["id"], 0) > 0]
    assert len(set(tri)) == len(tri)
    return ents, tri, slug


if __name__ == "__main__":
    ents, tri, _ = build_kg()
    from collections import Counter
    print(len(ents), "thực thể", Counter(e["kind"] for e in ents))
    print(len(tri), "triple", Counter(r for _, r, _ in tri))


# Bản dịch tiếng Anh của các thực thể không phải món, dùng cho phân tích chuyển
# giao (truy vấn tiếng Anh + CLIP tiếng Anh). Tên tỉnh giữ nguyên, bỏ dấu.
EN_ENTITY = {
    # vùng
    "Bắc Bộ": "Northern Vietnam", "Trung Bộ": "Central Vietnam",
    "Nam Bộ": "Southern Vietnam", "Tây Nguyên": "the Central Highlands",
    # loại món
    "món sợi nước": "noodle soup", "món sợi trộn": "dry noodle dish",
    "món cơm và xôi": "rice dish", "bánh mặn": "savory cake",
    "bánh truyền thống": "traditional festive cake", "bánh ngọt": "sweet cake",
    "món cuốn và nem": "roll or cured meat dish", "món kho": "braised dish",
    "canh và lẩu": "soup or hot pot", "đồ uống": "drink", "món ăn vặt": "street snack",
    "món nướng": "grilled dish",
    # cách chế biến
    "nấu": "simmered", "luộc": "boiled", "nướng": "grilled", "chiên": "fried",
    "hấp": "steamed", "kho": "braised", "xào": "stir-fried", "gói lá": "wrapped in leaves",
    "pha": "brewed", "lên men": "fermented",
    # nhóm nguyên liệu
    "tinh bột": "starch", "thịt và trứng": "meat and egg", "hải sản": "seafood",
    "đậu và hạt": "beans and nuts", "rau củ quả": "vegetables", "rau thơm và lá": "herbs",
    "gia vị": "spices", "mắm và nước chấm": "fish sauce and pastes", "lá gói": "wrapping leaves",
    "sữa và đồ uống": "milk and drinks",
    # nguyên liệu
    "bánh phở": "flat rice noodles", "bún": "rice vermicelli", "sợi hủ tiếu": "hu tieu noodles",
    "sợi mì gạo": "thick rice noodles", "gạo": "rice", "gạo nếp": "sticky rice",
    "bột gạo": "rice flour", "bột năng": "tapioca starch", "bột mì": "wheat flour",
    "bánh tráng": "rice paper", "thịt bò": "beef", "thịt lợn": "pork", "thịt gà": "chicken",
    "trứng": "egg", "trứng cút": "quail egg", "tôm": "shrimp", "cua": "crab", "cá": "fish",
    "ốc": "snails", "hến": "baby clams", "mực": "squid", "đậu xanh": "mung beans",
    "đậu phụ": "tofu", "đậu phộng": "peanuts", "giá đỗ": "bean sprouts",
    "rau muống": "water spinach", "cà chua": "tomato", "dứa": "pineapple",
    "cà tím": "eggplant", "xà lách": "lettuce", "dưa chuột": "cucumber", "cà rốt": "carrot",
    "mộc nhĩ": "wood ear mushroom", "sầu riêng": "durian", "rau thơm": "fresh herbs",
    "hành lá": "scallions", "rau răm": "Vietnamese coriander", "thì là": "dill",
    "lá lốt": "betel leaves", "sả": "lemongrass", "gừng": "ginger", "nghệ": "turmeric",
    "ớt": "chili", "hoa hồi": "star anise", "me": "tamarind", "nước mắm": "fish sauce",
    "mắm tôm": "shrimp paste", "mắm ruốc": "fermented krill paste", "mắm cá": "fermented fish",
    "lá dong": "dong leaves", "lá chuối": "banana leaves", "nước cốt dừa": "coconut milk",
    "sữa đặc": "condensed milk", "cà phê": "coffee",
}
