# -*- coding: utf-8 -*-
"""Xuất bản demo TĨNH cho GitHub Pages: tính sẵn top-k + giải thích + fidelity cho một danh sách truy vấn.

GitHub Pages không chạy được Python/CLIP, nên trang tĩnh chỉ cho chọn trong các truy vấn đã tính sẵn
(bộ truy vấn đánh giá val/test của MMKG + ví dụ). Muốn nhập truy vấn tự do, dùng demo/app.py (HF Spaces/Colab).

Chạy:  python demo/export_static.py [--out build/static_demo] [--k 5] [--no-fidelity]
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mmkg_engine import MMKGSearch

EXTRA = ["bánh mặn của Huế, có tôm", "món sợi nước của Nam Bộ, có mắm cá", "đồ uống của Hà Nội có trứng",
         "bánh gói lá dong ngày Tết", "món kho của Nam Bộ có cá", "chả giò chiên giòn", "Phở bò", "bun bo hue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "build", "static_demo"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fid-top", type=int, default=1, help="số kết quả đầu mỗi truy vấn được đo fidelity (0 = bỏ)")
    a = ap.parse_args()
    eng = MMKGSearch()
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)

    qlist = []
    seen = set()
    for t in EXTRA:
        qlist.append(dict(text=t, group="ví dụ", gold=None)); seen.add(t)
    for q in eng.test_queries:
        if q["vi"] not in seen:
            qlist.append(dict(text=q["vi"], group={"direct": "trực tiếp", "indirect": "gián tiếp",
                                                   "paraphrase": "diễn đạt tự nhiên"}[q["group"]], gold=q["gold"]))
            seen.add(q["vi"])

    used = set()
    out_q = []
    for n, q in enumerate(qlist):
        R = eng.search(q["text"], k=a.k, gallery="test")
        items = []
        for it in R["results"]:
            G = eng.explanation_graph(it, q["text"])
            fid = eng.fidelity(R, it) if it["rank"] <= a.fid_top else None
            used.add(it["img"]); used.update(it["evidence"])
            items.append(dict(rank=it["rank"], img=it["image_id"], dish=it["dish_name"],
                              correct=None if q["gold"] is None else it["dish"] in q["gold"],
                              d_star=eng.name(it["d_star"]), via_ok=it["dish"] == it["d_star"],
                              score=round(it["score"], 3), part_clip=round(it["part_clip"], 3),
                              part_graph=round(it["part_graph"], 3), cos=round(it["clip_cos"], 4),
                              share=round(it["share"], 3),
                              paths=[eng.path_str(p) for p in it["paths"]],
                              unmatched=[eng.name(e) for e in it["unmatched"]],
                              evidence=[eng.IMG_IDS[j] for j in it["evidence"]],
                              author=str(it["author"]), license=str(it["license"]), source=str(it["source"]),
                              fidelity=fid,
                              graph=dict(nodes=[dict(id=k, **v) for k, v in G.nodes(data=True)],
                                         edges=[dict(s=u, t=v, **d) for u, v, d in G.edges(data=True)])))
        out_q.append(dict(query=q["text"], group=q["group"],
                          gold=None if q["gold"] is None else [eng.name(d) for d in q["gold"]],
                          entities=[f"{eng.name(e)} [{eng.ENT[e]['kind']}]" for e in R["entities"]],
                          query_dishes=[[eng.name(d), round(p, 3)] for d, p in R["query_dishes"][:3]],
                          results=items))
        if n % 20 == 0:
            print(f"{n + 1}/{len(qlist)}", flush=True)

    from PIL import Image
    for j in sorted(used):
        im = Image.open(eng.imgs.path[j]).convert("RGB")
        im.thumbnail((240, 240))
        im.save(os.path.join(a.out, "img", eng.IMG_IDS[j] + ".jpg"), quality=82)

    meta = dict(model=eng.model_key, hp=eng.hp, val_mrr=eng.val_mrr, seed=eng.seed,
                gallery_size=len(eng.gal["test"]), k=a.k)
    with open(os.path.join(a.out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.DEMO = ")
        json.dump(dict(meta=meta, queries=out_q), f, ensure_ascii=False)
        f.write(";\n")
    import shutil
    shutil.copy(os.path.join(HERE, "static_template", "index.html"), os.path.join(a.out, "index.html"))
    print(f"Xong: {len(out_q)} truy vấn, {len(used)} ảnh → {a.out}")


if __name__ == "__main__":
    main()
