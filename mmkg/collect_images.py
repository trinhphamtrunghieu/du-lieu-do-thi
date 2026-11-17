# -*- coding: utf-8 -*-
"""Thu thập ảnh cho MMKG từ Wikimedia Commons (đã chạy ngày 17/11/2026).

Quy trình (đúng như lần chạy đã tạo images.csv):
1. Với mỗi món: lấy tệp trong Category:<tên món tiếng Việt>, rồi tìm kiếm Commons (namespace File)
   với 3 câu: "<tên món>" (có ngoặc kép), tên không dấu, 4 từ đầu của tên tiếng Anh.
2. Lọc: chỉ JPEG/PNG, rộng ≥ 300 px; giấy phép CC0 / CC BY / CC BY-SA / Public domain;
   tiêu đề / danh mục / mô tả phải chứa một từ khoá của món (kg_data.D, cột cuối, so khớp
   theo ranh giới từ trên chuỗi không dấu). Khi một tệp khớp nhiều món, chọn món có từ khoá dài
   nhất (ưu tiên khớp ở tiêu đề/danh mục hơn mô tả). Bỏ tệp có tiêu đề chứa từ chỉ cửa hàng,
   biển hiệu, chợ, thực đơn...
3. Chọn tối đa 12 ứng viên/món, ưu tiên khớp ở tiêu đề và đến từ danh mục; không quá 2 tệp cùng
   gốc tên, không quá 4 tệp cùng tác giả. Tải ảnh thu nhỏ rộng 384 px.
4. Duyệt bằng mắt từng ứng viên (bảng ảnh theo món), loại ảnh không phải món đó hoặc không
   thấy rõ món (tháp Chăm "Bánh Ít", biển hiệu, quầy hàng, người, tranh vẽ...). Loại ảnh gần trùng
   (dHash, khoảng cách ≤ 40/256), giữ tối đa 10 ảnh/món. Quyết định từng ứng viên nằm trong
   image_candidates_review.csv.

Chạy lại: python collect_images.py --out cand/   (cần mạng tới commons.wikimedia.org).
Kết quả tìm kiếm của Commons thay đổi theo thời gian, nên lần chạy sau có thể ra tập ứng viên khác;
tập ảnh dùng trong báo cáo là tập đã lưu trong images/ và images.csv.
"""
import argparse, html, json, os, re, time, unicodedata
from collections import Counter, defaultdict

import requests

import kg_data

UA = {"User-Agent": "Project3-MMKG-VN/0.1 (student research project; "
                    "https://github.com/trinhphamtrunghieu/du-lieu-do-thi)"}
API = "https://commons.wikimedia.org/w/api.php"
II = dict(action="query", format="json", prop="imageinfo|categories",
          iiprop="url|extmetadata|mime|size", iiurlwidth=384, cllimit="max", clshow="!hidden")
BAD = re.compile(r"\b(sign|signboard|menu|shop|store|stall|restaurant|counter|street|quan|bien|cua hang|map|"
                 r"logo|vendor|seller|market|cho|tiem|nha hang|recipe card|poster|packag|box|bag)\b")
OKLIC = re.compile(r"^(cc0|cc by|cc-by|public domain|pd)", re.I)


def norm(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def strip_html(s):
    return html.unescape(re.sub("<[^>]+>", " ", s or ""))


def api(params):
    for a in range(4):
        try:
            r = requests.get(API, params=params, headers=UA, timeout=40)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2 + 3 * a)
    return {}


def pages(params):
    return list(api(dict(II, **params)).get("query", {}).get("pages", {}).values())


def candidates():
    kw = {d[0]: (d[1], d[6], kg_data.EN_NAME[d[0]]) for d in kg_data.D}
    kw["ca_phe_sua_da"] = (kw["ca_phe_sua_da"][0], ["ca phe sua da", "cafe sua da", "ca phe sua"], kw["ca_phe_sua_da"][2])
    raw = {}
    for did, (name, _, en) in kw.items():
        ps = pages(dict(generator="categorymembers", gcmtitle="Category:" + name, gcmtype="file", gcmlimit=100))
        src = {p["title"]: "category" for p in ps}
        for q in [f'"{name}"', norm(name), " ".join(en.split()[:4])]:
            for p in pages(dict(generator="search", gsrnamespace=6, gsrsearch=q, gsrlimit=50)):
                if p["title"] not in src:
                    src[p["title"]] = "search"
                    ps.append(p)
        raw[did] = [(p, src[p["title"]]) for p in ps]
    return kw, raw


def filter_and_select(kw, raw):
    pat = {d: [re.compile(r"\b" + re.escape(k) + r"\b") for k in v[1]] for d, v in kw.items()}
    allc = {}
    for did, lst in raw.items():
        for p, src in lst:
            c = allc.setdefault(p["title"], dict(p=p, srcs=set()))
            c["srcs"].add((did, src))
    rows = []
    for t, c in allc.items():
        p = c["p"]; ii = (p.get("imageinfo") or [{}])[0]; md = ii.get("extmetadata", {})
        if ii.get("mime") not in ("image/jpeg", "image/png") or (ii.get("width") or 0) < 300:
            continue
        lic = md.get("LicenseShortName", {}).get("value") or ""
        if not OKLIC.match(lic.strip()):
            continue
        ttl = norm(t[5:].rsplit(".", 1)[0])
        cats = norm(" ".join(x["title"][9:] for x in p.get("categories", [])))
        desc = norm(strip_html(md.get("ImageDescription", {}).get("value", ""))[:400])
        best = None
        for d, ps in pat.items():
            for k, pp in zip(kw[d][1], ps):
                where = "title" if pp.search(ttl) else "category" if pp.search(cats) else "desc" if pp.search(desc) else None
                if where:
                    sc = (where != "desc", len(k))
                    if best is None or sc > best[0]:
                        best = (sc, d, k, where)
        if best is None or BAD.search(ttl):
            continue
        artist = norm(strip_html(md.get("Artist", {}).get("value", "")))[:80]
        rows.append(dict(title=t, dish=best[1], kw=best[2], where=best[3],
                         src=sorted(s for d, s in c["srcs"] if d == best[1]) or ["other_dish_query"],
                         lic=lic, licurl=md.get("LicenseUrl", {}).get("value", ""), artist=artist,
                         thumb=ii.get("thumburl"), page=ii.get("descriptionurl"), w=ii.get("width"), h=ii.get("height")))
    by = defaultdict(list)
    for r in rows:
        by[r["dish"]].append(r)
    stem = lambda t: re.sub(r"\b\d+\b", "", norm(t[5:].rsplit(".", 1)[0])).strip()[:40]
    sel = []
    for d in sorted(by):
        lst = sorted(by[d], key=lambda r: (r["where"] != "title", "category" not in r["src"], -(r["w"] or 0), r["title"]))
        st, au, pick = Counter(), Counter(), []
        for r in lst:
            if st[stem(r["title"])] >= 2 or au[r["artist"]] >= 4:
                continue
            st[stem(r["title"])] += 1; au[r["artist"]] += 1; pick.append(r)
            if len(pick) >= 12:
                break
        sel += pick
    for i, r in enumerate(sel):
        r["cid"] = i
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cand")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kw, raw = candidates()
    sel = filter_and_select(kw, raw)
    json.dump(sel, open(os.path.join(a.out, "selected.json"), "w"), ensure_ascii=False, indent=0)
    for r in sel:
        fn = os.path.join(a.out, f"{r['dish']}__{r['cid']:04d}.jpg")
        if not os.path.exists(fn):
            x = requests.get(r["thumb"], headers=UA, timeout=20)
            if x.status_code == 200:
                open(fn, "wb").write(x.content)
    print(len(sel), "ứng viên; duyệt bằng mắt rồi ghi quyết định vào image_candidates_review.csv")
