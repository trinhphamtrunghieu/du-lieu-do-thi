# -*- coding: utf-8 -*-
"""Thêm checkpoint + bundle demo vào notebooks/components/partA_du-lieu-do-thi.ipynb (chạy một lần, idempotent).

- Cell mới "1.17 Checkpoint và bundle demo" (sau self_test): save_ckpt / train_or_load / export_demo_bundle.
- run_experiment: train_graph_scorer → train_or_load; lưu checkpoint sau khi chọn (α, λ) trên VAL.
- Cell Action Genome (bước 9/9): đặt CKPT_TAG = "ag" và xuất bundle demo cho gallery test.
"""
import json, sys
from pathlib import Path

NB = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/components/partA_du-lieu-do-thi.ipynb")
MARK = "# 1.17 CHECKPOINT VÀ BUNDLE DEMO"

CELL = r'''# =====================================================================================
# 1.17 CHECKPOINT VÀ BUNDLE DEMO
#   • save_ckpt / train_or_load: lưu trọng số GraphScorer + (α, λ) của từng (bộ dữ liệu, backbone, ablation, seed)
#     vào CFG.out_dir/checkpoints/<tag>_<clip>_<ablation>_s<seed>.pt. Lần chạy sau tìm file cùng tên trong
#     CFG.out_dir/checkpoints, các thư mục trong biến môi trường P3_RESUME_DIR (phân cách ':') và mọi
#     /kaggle/input/**/checkpoints ⇒ nạp lại, bỏ qua huấn luyện. Chỉ nạp khi số node + số cạnh của đồ thị khớp
#     (đồ thị dựng lại từ cùng dữ liệu + split). Đặt P3_RESUME=0 để luôn huấn luyện lại.
#   • export_demo_bundle: xuất mọi thứ cần để chấm TRUY VẤN TỰ DO trên CPU đúng như demo_query (đặc trưng CLIP +
#     nhúng GNN của gallery test, ma trận đường đi, đồ thị cho giải thích, đầu chiếu truy vấn q_proj, trọng số mẫu
#     đường đi, α, λ) + kết quả tham chiếu của CHÍNH notebook cho một danh sách truy vấn (để kiểm tra bản port)
#     + thumbnail. Dùng cho demo/ (Hugging Face Space).
# =====================================================================================
import pickle
import scipy.sparse as _sp_ck

CKPT_TAG = "flickr8k"
P3_RESUME = os.environ.get("P3_RESUME", "1") == "1"


def _ckpt_name(ablation, seed):
    return f"{CKPT_TAG}_{CFG.clip_model}_{ablation}_s{seed}.pt"


def _ckpt_dirs():
    ds = [CFG.out_dir / "checkpoints"] + [Path(p) for p in os.environ.get("P3_RESUME_DIR", "").split(":") if p]
    ds += [Path(p) for p in sorted(glob.glob("/kaggle/input/**/checkpoints", recursive=True))]
    return ds


def find_ckpt(ablation, seed):
    if not P3_RESUME:
        return None
    for d in _ckpt_dirs():
        p = Path(d) / _ckpt_name(ablation, seed)
        if p.exists():
            return p
    return None


def save_ckpt(model, graph, ablation, seed, alpha=None, lam=None, hist=None):
    d = CFG.out_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    p = d / _ckpt_name(ablation, seed)
    # bỏ buffer img_feat (= đặc trưng CLIP đóng băng, nạp lại từ cache) cho file nhỏ
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != "img_feat"}
    torch.save(dict(state=state, n_nodes=int(graph.n_nodes), n_edges=int(graph.edge_index.shape[1]),
                    alpha=alpha, lam=lam, hist=list(hist or []), clip=[CFG.clip_model, CFG.clip_weights],
                    gnn_type=CFG.gnn_type, learn_clip_w=bool(CFG.learn_clip_w), tag=CKPT_TAG, ablation=ablation,
                    seed=seed), p)
    return p


def train_or_load(engine, train_q_idx, train_img_idx, ablation, seed=0, verbose=True):
    """Nạp checkpoint nếu có và khớp đồ thị; nếu không thì huấn luyện như cũ (train_graph_scorer hiện hành)."""
    trainer = globals()["train_graph_scorer"]
    p = find_ckpt(ablation, seed)
    if p is not None:
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        g = engine.graph
        if ck["n_nodes"] == g.n_nodes and ck["n_edges"] == int(g.edge_index.shape[1]):
            ep = CFG.epochs
            CFG.epochs = 0                                 # dựng mô hình đúng kiến trúc, không huấn luyện
            try:
                model, _ = trainer(engine, train_q_idx, train_img_idx, seed=seed, verbose=False)
            finally:
                CFG.epochs = ep
            miss, extra = model.load_state_dict(ck["state"], strict=False)
            assert not extra and set(miss) <= {"img_feat"}, (miss, extra)
            model.eval()
            print(f"  ↺ nạp checkpoint {p} — bỏ qua huấn luyện")
            return model, ck.get("hist") or []
        print(f"  ⚠ checkpoint {p.name} không khớp đồ thị (node/cạnh {ck['n_nodes']}/{ck['n_edges']} ≠ "
              f"{g.n_nodes}/{int(g.edge_index.shape[1])}) → huấn luyện lại")
    return trainer(engine, train_q_idx, train_img_idx, seed=seed, verbose=verbose)


@torch.no_grad()
def demo_query_dict(text, art, k=5, parse=None):
    """Giống hệt demo_query (1.15) nhưng TRẢ VỀ kết quả thay vì in — làm tham chiếu cho bản port của demo/."""
    parse = parse or globals()["parse_query"]
    eng, graph, model = art["engine"], art["graph"], art["model"]
    model.eval()
    gallery = np.asarray(art["parts"]["test"]["img_idx"])
    hits = parse(text, graph)
    cids = [n for n, _, _ in hits]
    q_emb = encode_free_text(text, eng)
    sim = (q_emb @ eng.img_feat[gallery].T).numpy()[0]
    top = np.argsort(-sim)[:CFG.rerank_k]
    cand = gallery[top]
    pm = torch.tensor(np.stack([_path_feats_free(graph, cand, cids)])).to(DEVICE)
    h = model.encode_graph(graph.edge_index.to(DEVICE))
    s_gnn = model.gnn_score(h, q_emb.to(DEVICE), torch.tensor(cand[None], device=DEVICE)).cpu().numpy()
    s_path = model.path_score(pm).cpu().numpy()
    s = combine(znorm_per_row(sim[top][None]), znorm_per_row(s_gnn), znorm_per_row(s_path),
                art["alpha"], art["lam"])[0]
    order = np.argsort(-s)[:k]
    res = []
    for rank, j in enumerate(order, 1):
        i = int(cand[j])
        expl, full, det = explain_and_score(model, graph, i, cids)
        res.append(dict(rank=rank, pos=i, clip_cos=float(sim[top][j]), clip_rank=int(j) + 1, score=float(s[j]),
                        s_gnn=float(s_gnn[0, j]), s_path=float(s_path[0, j]), explanation=expl,
                        n_paths=det["n_paths"]))
    return dict(query=text, concepts=[[int(n), g_] for n, g_, _ in hits], results=res,
                q_vec=[float(x) for x in q_emb.numpy().ravel()])


def _sp_save(path, M):
    _sp_ck.save_npz(path, _sp_ck.csr_matrix(M).astype(np.float32))


@torch.no_grad()
def export_demo_bundle(art, records, name, demo_texts=(), k=5, thumb_px=256, parse=None, parse_cfg=None,
                       frames_of=None):
    """Ghi CFG.out_dir/demo_bundle/<name>/ (xem demo/README.md). frames_of(record) → list đường dẫn frame/ảnh."""
    out = CFG.out_dir / "demo_bundle" / name
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    eng, g, model = art["engine"], art["graph"], art["model"]
    model.eval()
    gal = np.asarray(art["parts"]["test"]["img_idx"])
    h = model.encode_graph(g.edge_index.to(DEVICE)).cpu().numpy()
    qp = {f"qproj.{k_}": v.detach().cpu().numpy() for k_, v in model.q_proj.state_dict().items()}
    pw = (model.pattern_weights() * F.softplus(model.path_gain)).detach().cpu().numpy()
    np.savez_compressed(out / "arrays.npz", gallery=gal, img_feat=eng.img_feat[gal].numpy().astype(np.float32),
                        h_img=h[gal].astype(np.float32), path_w=pw.astype(np.float32),
                        idf=np.asarray(g.idf, np.float32), **qp)
    for p_ in PATTERNS:
        _sp_save(out / f"R_{p_}.npz", g.R[p_][gal])
    for nm in ("A_scene", "A_related", "A_isa"):
        _sp_save(out / f"{nm}.npz", getattr(g, nm))
    A_t = getattr(g, "A_triple", None)
    if A_t is not None:
        _sp_save(out / "A_triple.npz", A_t[gal])
    tri2id = getattr(g, "tri2id", None) or {}
    meta = dict(name=name, clip=[CFG.clip_model, CFG.clip_weights], alpha=float(art["alpha"]), lam=float(art["lam"]),
                rerank_k=int(CFG.rerank_k), top_m_paths=int(CFG.top_m_paths), patterns=list(PATTERNS),
                pattern_desc=dict(PATTERN_DESC), drop_redundant_indirect=bool(CFG.drop_redundant_indirect),
                n_img=int(g.n_img), n_nodes=int(g.n_nodes), stop=sorted(STOP),
                name2id={k_: int(v) for k_, v in g.name2id.items()},
                id2name={str(int(k_)): v for k_, v in g.id2name.items()},
                scene_pred=[[int(a), int(b), r] for (a, b), r in g.scene_pred.items()],
                tri2id=[[list(t), int(i)] for t, i in tri2id.items()], n_tri=int(getattr(g, "n_tri", 0) or 0),
                parse=parse_cfg or dict(kind="default"),
                items=[dict(pos=int(i), id=str(records[i].get("video_id", records[i]["image_id"])),
                            caption=records[i]["caption"], objects=list(records[i]["objects"])[:40],
                            img_objects=[int(x) for x in g.img_objects[i]]) for i in gal])
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False)
    ref = [demo_query_dict(t, art, k=k, parse=parse) for t in demo_texts]
    json.dump(ref, open(out / "reference_outputs.json", "w"), ensure_ascii=False, indent=1)
    n_th = 0
    if frames_of is not None:
        from PIL import Image
        for i in gal:
            fr = [f for f in frames_of(records[i]) if os.path.exists(f)]
            if not fr:
                continue
            try:
                im = Image.open(fr[len(fr) // 2]).convert("RGB")
                im.thumbnail((thumb_px, thumb_px))
                im.save(out / "thumbs" / f"{meta['items'][list(gal).index(i)]['id']}.jpg", quality=82)
                n_th += 1
            except Exception as ex:
                print("  thumbnail lỗi:", ex)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2 ** 20
    print(f"Bundle demo '{name}': {len(gal)} mục gallery, {len(ref)} truy vấn tham chiếu, {n_th} thumbnail, "
          f"{mb:.1f} MB → {out}")
    return out
'''


def main():
    nb = json.load(open(NB))
    src = lambda c: "".join(c["source"])
    if any(MARK in src(c) for c in nb["cells"]):
        print("Đã patch rồi:", NB)
        return
    # 1) cell mới ngay sau cell self_test (định nghĩa 1.16)
    i_self = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and src(c).startswith("def self_test"))
    md = dict(cell_type="markdown", metadata={}, source="## 1.17 Checkpoint và bundle demo\n\n"
              "Lưu / nạp lại trọng số mô hình (bỏ qua huấn luyện khi chạy lại) và xuất dữ liệu cho demo truy vấn tự do "
              "(`demo/`, Hugging Face Space).")
    code = dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=CELL)
    nb["cells"][i_self + 1:i_self + 1] = [md, code]

    # 2) run_experiment: train_or_load + save_ckpt
    i_run = next(i for i, c in enumerate(nb["cells"]) if "def run_experiment(" in src(c))
    s = src(nb["cells"][i_run])
    old_train = ('            model, hist = train_graph_scorer(eng, parts["train"]["q_idx"],\n'
                 '                                             parts["train"]["img_idx"], seed=seed,\n'
                 '                                             verbose=(si == 0))\n')
    new_train = ('            model, hist = train_or_load(eng, parts["train"]["q_idx"],\n'
                 '                                        parts["train"]["img_idx"], ab, seed=seed,\n'
                 '                                        verbose=(si == 0))\n')
    assert old_train in s
    s = s.replace(old_train, new_train)
    old_tune = ('            (alpha, lam), grid = tune_on_val(model, eng, parts["val"], parts["val"]["img_idx"],\n'
                '                                             verbose=(si == 0))\n')
    assert old_tune in s
    s = s.replace(old_tune, old_tune + '            save_ckpt(model, graph, ab, seed, alpha, lam, hist)\n')
    nb["cells"][i_run]["source"] = s

    # 3) Action Genome: tag checkpoint + xuất bundle ở bước 9/9
    i_ag = next(i for i, c in enumerate(nb["cells"]) if "def load_action_genome" in src(c))
    s = src(nb["cells"][i_ag])
    old_with = 'AG_OUT = Path("/kaggle/working/output/action_genome")\n'
    assert old_with in s
    s = s.replace(old_with, old_with + 'CKPT_TAG = "ag"                      # tên checkpoint: ag_<clip>_<ablation>_s<seed>.pt\n')
    old_demo = '        demo_query("someone watching tv and holding a phone", ag_art, k=3)\n'
    assert old_demo in s
    s = s.replace(old_demo, old_demo + r'''    with step("9b/9 bundle demo (truy vấn tự do trên Space)"):
        _ag_demo_texts = ["a person drinking from a glass while sitting on the couch",
                          "someone watching tv and holding a phone", "a person opens the door and walks in",
                          "someone is cooking food in the kitchen", "a person holding a book and reading on the bed",
                          "a person takes a towel then wipes the table"]
        _ag_demo_texts += [q["text"] for q in json.load(open(R / "eval_queries_20.json"))["queries"]]
        export_demo_bundle(ag_art, ag_records, "action_genome", _ag_demo_texts, k=5, parse=ag_parse_query,
                           parse_cfg=dict(kind="ag", verb_alias=AG_VERB_ALIAS, noun_alias=AG_NOUN_ALIAS,
                                          verbs=sorted(_AG_VERBS), action_nodes=sorted(CH_ACTION_NODES)),
                           frames_of=lambda r: r["frames"])
''')
    nb["cells"][i_ag]["source"] = s
    json.dump(nb, open(NB, "w"), ensure_ascii=False, indent=1)
    print("Đã patch:", NB)


if __name__ == "__main__":
    main()
