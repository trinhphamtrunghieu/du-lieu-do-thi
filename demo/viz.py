# -*- coding: utf-8 -*-
"""Vẽ đồ thị con giải thích và thẻ kết quả (HTML) cho demo."""
import base64, html, io

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

COLORS = {"query": "#f4a261", "result": "#2a9d8f", "món": "#e76f51", "evidence": "#8ab17d",
          "unmatched": "#bbbbbb", "loại món": "#457b9d", "tỉnh": "#6d597a", "vùng": "#9d4edd",
          "nguyên liệu": "#e9c46a", "nhóm nguyên liệu": "#f2cc8f", "cách chế biến": "#81b29a"}


def _layout(G):
    """Bố cục theo cột: truy vấn | thực thể nhắc tới | trung gian | món d* | ảnh."""
    col = {}
    for n, a in G.nodes(data=True):
        k = a["kind"]
        if k == "query":
            col[n] = 0
        elif k in ("result", "evidence"):
            col[n] = 4
        elif k == "món":
            col[n] = 3
        else:
            col[n] = 1
    # node trung gian của đường đi 2 bước (nguyên liệu/tỉnh nằm giữa món và nhóm/vùng)
    for n, a in G.nodes(data=True):
        if col[n] == 1 and any(G.nodes[p]["kind"] == "món" for p in G.predecessors(n)) \
                and any(G.nodes[s]["kind"] not in ("món",) for s in G.successors(n)):
            col[n] = 2
    pos = {}
    for c in range(5):
        ns = [n for n in G if col[n] == c]
        for i, n in enumerate(ns):
            pos[n] = (c, -(i - (len(ns) - 1) / 2))
    return pos


def draw_explanation(G, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.2))
    pos = _layout(G)
    kinds = nx.get_node_attributes(G, "kind")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1700,
                           node_color=[COLORS.get(kinds[n], "#cccccc") for n in G], edgecolors="#333", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, "label"), font_size=7.5)
    styles = {"mention": "dashed", "pred": "dotted", "kg": "solid"}
    for kind, st in styles.items():
        es = [(u, v) for u, v, a in G.edges(data=True) if a["kind"] == kind]
        nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax, style=st, arrows=True, arrowsize=12,
                               node_size=1500, edge_color="#555")
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=nx.get_edge_attributes(G, "label"), font_size=7,
                                 rotate=False, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_axis_off()
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6); ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_to_b64(path, width=220):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def result_html(eng, R, with_graph=True, fid=None):
    """Bảng kết quả HTML: ảnh, điểm (CLIP + đồ thị), đường đi giải thích, đồ thị con."""
    e = html.escape
    hp = R["hp"]
    head = (f"<div style='font-family:sans-serif'><h3 style='margin:4px 0'>Truy vấn: «{e(R['query'])}»</h3>"
            f"<div><b>Thực thể nhận ra trong KG:</b> "
            + (", ".join(f"{e(eng.name(x))} <i>[{e(eng.ENT[x]['kind'])}]</i>" for x in R["entities"]) or "— (chỉ dùng CLIP + mô tả node)")
            + f"</div><div><b>P(món | truy vấn):</b> "
            + ", ".join(f"{e(eng.name(d))} {p:.2f}" for d, p in R["query_dishes"][:3])
            + f"</div><div style='color:#666;font-size:12px'>gallery = {e(R['gallery'])} ({len(R['_gal'])} ảnh) · "
              f"α={hp['alpha']:.2f}, λ={hp['lam']:.2f}, η={hp['eta']:.2f}, T={hp['T']}, w2={hp['w2']} (chọn trên val)</div>")
    rows = []
    for it in R["results"]:
        ok = it["dish"] == it["d_star"]
        paths = "".join(f"<li>{e(eng.path_str(p))}</li>" for p in it["paths"])
        paths += "".join(f"<li style='color:#a00'>ràng buộc «{e(eng.name(x))}» không thỏa với món này</li>" for x in it["unmatched"])
        if it["evidence"]:
            paths += (f"<li>ảnh train gần nhất —mô_tả→ {e(eng.name(it['d_star']))}: "
                      + ", ".join(e(eng.IMG_IDS[j]) for j in it["evidence"]) + "</li>")
        graph = ""
        if with_graph:
            fig, ax = plt.subplots(figsize=(9, 3.6))
            draw_explanation(eng.explanation_graph(it, R["query"]), ax)
            graph = f"<img style='max-width:100%' src='data:image/png;base64,{fig_to_b64(fig)}'/>"
        fhtml = ""
        if fid and it["rank"] in fid:
            f = fid[it["rank"]]
            fhtml = (f"<div style='font-size:12px;background:#f6f6f6;padding:4px'><b>Kiểm tra fidelity</b> "
                     f"(xoá {f['n_edges']} cạnh của giải thích, không huấn luyện lại): điểm {f['score_before']:+.2f} → "
                     f"{f['score_after_expl']:+.2f}, hạng {f['rank_before']} → {f['rank_after_expl']} · "
                     f"xoá ngẫu nhiên cùng số cạnh: điểm {f['score_after_rand']:+.2f}, hạng {f['rank_after_rand']:.1f}</div>")
        rows.append(
            f"<div style='display:flex;gap:12px;border-top:1px solid #ddd;padding:8px 0'>"
            f"<div style='flex:0 0 200px'><img style='width:200px;border-radius:4px' src='data:image/jpeg;base64,{img_to_b64(it['path'])}'/>"
            f"<div style='font-size:11px;color:#666'>{e(it['image_id'])} · {e(str(it['author']))} · {e(str(it['license']))}</div></div>"
            f"<div style='flex:1'><div><b>#{it['rank']} — ảnh là «{e(it['dish_name'])}»</b> "
            f"<span style='color:{'#2a9d8f' if ok else '#c1121f'}'>{'✓' if ok else '✗'} qua món «{e(eng.name(it['d_star']))}»</span></div>"
            f"<div>điểm <b>{it['score']:+.2f}</b> = CLIP {it['part_clip']:+.2f} + đồ thị {it['part_graph']:+.2f} "
            f"<span style='color:#666'>(cos {it['clip_cos']:.3f}; d* chiếm {it['share']:.0%} điểm đồ thị)</span></div>"
            f"<ul style='margin:4px 0'>{paths}</ul>{fhtml}{graph}</div></div>")
    return head + "".join(rows) + "</div>"
