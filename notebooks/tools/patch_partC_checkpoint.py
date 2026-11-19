# -*- coding: utf-8 -*-
"""Nạp lại checkpoint + bundle demo cho notebooks/components/partC_project03-level4b-v2-graphfix.ipynb (idempotent).

- run_exp: nếu đã có checkpoints/<exp_id>_s<seed>.pt (trong CFG.out_dir/checkpoints, MSVD_RESUME_DIR hoặc
  /kaggle/input/**/checkpoints) thì dựng mô hình cùng kiến trúc và nạp trọng số, bỏ qua huấn luyện
  (MSVD_RESUME=0 để tắt). Checkpoint vẫn được lưu như cũ sau mỗi lần huấn luyện.
- Cell mới sau cell demo (69): export_msvd_demo_bundle — xuất dữ liệu cho demo truy vấn tự do trên CPU
  (mô hình G3 seed đầu, tensor đồ thị, từ vựng node/vị từ, đặc trưng gallery, thumbnail) + kết quả tham chiếu
  của chính demo_query để kiểm tra bản port trong demo/.
"""
import json, sys
from pathlib import Path

NB = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/components/partC_project03-level4b-v2-graphfix.ipynb")
MARK = "4(b) — BUNDLE DEMO: dữ liệu cho truy vấn tự do"

RESUME_HELPERS = r'''
MSVD_RESUME = os.environ.get("MSVD_RESUME", "1") == "1"


def find_ckpt_c(exp_id, seed):
    """checkpoints/<exp_id>_s<seed>.pt của lần chạy trước (output Kaggle được Add Input) — None nếu không có / tắt."""
    if not MSVD_RESUME:
        return None
    dirs = [CFG.out_dir / "checkpoints"] + [Path(p) for p in os.environ.get("MSVD_RESUME_DIR", "").split(":") if p]
    dirs += [Path(p) for p in sorted(glob.glob("/kaggle/input/**/checkpoints", recursive=True))]
    for d_ in dirs:
        p = Path(d_) / f"{exp_id}_s{seed}.pt"
        if p.exists():
            return p
    return None


def load_ckpt_c(eng, design, path):
    """Dựng mô hình đúng kiến trúc của `design` rồi nạp trọng số (checkpoint lưu half ⇒ ép về dtype của mô hình)."""
    model = (make_model_v4(eng, design) if design.get("model") == "v4" else
             make_model(eng, SYM_FEAT, design["mp_weight"], design["init"], design["pred_mode"]))
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    ref = model.state_dict()
    assert set(sd) == set(ref), f"checkpoint {path.name} khác kiến trúc: {sorted(set(sd) ^ set(ref))[:5]}"
    for k_, v in sd.items():
        assert tuple(v.shape) == tuple(ref[k_].shape), f"{path.name}: {k_} {tuple(v.shape)} ≠ {tuple(ref[k_].shape)} (đồ thị khác)"
    model.load_state_dict({k_: v.to(ref[k_].dtype) for k_, v in sd.items()})
    model.eval()
    return model


'''

OLD_TRAIN = '''        if reuse is not None:
            model, hist = EXPS[reuse]["runs"][seed]["model"], EXPS[reuse]["runs"][seed]["hist"]
        elif design.get("model") == "v4":'''
NEW_TRAIN = '''        ck_in = find_ckpt_c(exp_id, seed) if reuse is None else None
        if reuse is not None:
            model, hist = EXPS[reuse]["runs"][seed]["model"], EXPS[reuse]["runs"][seed]["hist"]
        elif ck_in is not None:                                  # nạp checkpoint, bỏ qua huấn luyện
            model, hist = load_ckpt_c(eng, design, ck_in), []
            print(f"    ↺ {exp_id} seed {seed}: nạp {ck_in}")
        elif design.get("model") == "v4":'''
OLD_SAVE = "        if CFG.save_checkpoints and reuse is None:\n"
NEW_SAVE = "        if CFG.save_checkpoints and reuse is None and ck_in is None:\n"

CELL = r'''# ============================== 4(b) — BUNDLE DEMO: dữ liệu cho truy vấn tự do trên CPU (demo/, Hugging Face Space) ==============================
# Ghi CFG.out_dir/demo_bundle/<name>/: bundle.pt (mô hình G3 seed đầu + tensor đồ thị + từ vựng node/vị từ + đặc trưng
# gallery), items.json (video, caption mẫu), reference_outputs.json (kết quả CHÍNH demo_query cho các truy vấn mẫu, để
# kiểm tra bản port), thumbs/. Mã xử lý truy vấn (POS + lemma + parse) được demo/ sao chép nguyên văn từ notebook này.
import shutil as _sh


@torch.no_grad()
def export_msvd_demo_bundle(texts, part=None, run=None, name="msvd", thumb_px=256):
    run = run or EXPS[G_IDS["G3"]]["runs"][CFG.seeds[0]]
    part = part or EVALS["explore"]["part"]
    m = run["model"]; m.eval()
    D, g, inv = m.D, m.D.graph, m.D.graph.inv
    out = CFG.out_dir / "demo_bundle" / name
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    gal = np.asarray(part["gallery"])
    s_, d_, e_, w_, pi_ = g.edge_arrays()
    edge_text = []
    for k in range(D.E):
        j = int(D.gidx[k])
        p = ""
        if pi_[j] >= 0:
            row = g.pmix[pi_[j]]
            if row.nnz:
                c_ = int(row.indices[int(np.argmax(row.data))])
                p = PRED_VOCAB[c_][1]
        edge_text.append([inv.name[int(s_[j])], EDGE_TYPES[int(e_[j])] + (f"[{p}]" if p else ""), inv.name[int(d_[j])]])
    Acon = D.Acon[torch.as_tensor(gal, device=D.Acon.device)].cpu().numpy()
    bundle = dict(
        name=name, part=part["name"], clip=[CFG.clip_model, CFG.clip_weights], alpha=float(run["alpha"]),
        rerank_k=int(CFG.rerank_k), top_m=int(CFG.top_m_paths),
        ctor=dict(clip_dim=int(TXT_FEAT.shape[1]), d=int(CFG.v4_dim), T=int(m.T), rel=bool(m.rel), qcond=bool(m.qcond),
                  use_pred=bool(m.use_pred)),
        state={k_: v.detach().cpu().float() for k_, v in m.state_dict().items()},
        gallery=gal, vid_t=VID_T[torch.as_tensor(gal, device=VID_T.device)].cpu().numpy().astype(np.float32),
        vid_feat=np.asarray(VID_FEAT[gal], np.float32),
        D=dict(N=int(D.N), ns=int(D.ns), E=int(D.E), src=D.src_np.astype(np.int64), dst=D.dst_np.astype(np.int64),
               et=D.et.cpu().numpy().astype(np.int64), logw=D.logw.cpu().numpy().astype(np.float32),
               PE=D.PE.cpu().numpy().astype(np.float32), Mex=sp.csr_matrix(D.Mex, dtype=np.float32),
               Acon=sp.csr_matrix(Acon, dtype=np.float32)),
        edge_text=edge_text,
        inv=dict(n_vid=int(inv.n_vid), class_names=[inv.name[i] for i in range(inv.n_vid, inv.n_nodes)],
                 class_pos=[str(inv.pos[i]) for i in range(inv.n_vid, inv.n_nodes)],
                 key2id={k_: int(v) for k_, v in inv.key2id.items()},
                 concepts_by_lemma=[[list(k_), [int(x) for x in v]] for k_, v in inv.concepts_by_lemma.items()],
                 video_names=[inv.name[int(v)] for v in gal]),
        pred_vocab=[list(x) for x in PRED_VOCAB], pred_emb=np.asarray(PRED_EMB, np.float32),
        pred_families=list(PRED_FAMILIES), q_roles=list(Q_ROLES), rel_types=list(V4_REL_TYPES),
        nltk=dict(NLTK_STATUS) if "NLTK_STATUS" in globals() else {}, tagger=_TAGGER is not None)
    torch.save(bundle, out / "bundle.pt")
    items = []
    for v in gal:
        r = RECORDS[int(v)]
        items.append(dict(vid=int(v), id=r["video_id"]))
        p = thumb_path(int(v))
        if p is not None and Path(p).exists():
            try:
                from PIL import Image
                im = Image.open(p).convert("RGB"); im.thumbnail((thumb_px, thumb_px))
                im.save(out / "thumbs" / f"{r['video_id']}.jpg", quality=82)
            except Exception as ex:
                print("  thumbnail lỗi:", r["video_id"], ex)
    json.dump(items, open(out / "items.json", "w"), ensure_ascii=False)
    ref = [demo_query(t, run=run, part=part, show=False) for t in texts]
    for o, t in zip(ref, texts):                         # vector truy vấn đúng như demo_query dùng (kiểm tra bản port)
        o["q_vec"] = [float(x) for x in clip_encode_texts([t])[0].astype(TXT_FEAT.dtype)]
        o["txt_dtype"] = str(TXT_FEAT.dtype)
    json.dump(ref, open(out / "reference_outputs.json", "w"), ensure_ascii=False, indent=1)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2 ** 20
    print(f"Bundle demo '{name}' ({part['name']}): {len(gal)} video, {D.E} cạnh quan hệ, {len(ref)} truy vấn tham chiếu, "
          f"{len(list((out / 'thumbs').glob('*.jpg')))} thumbnail, {mb:.1f} MB → {out}")
    return out


_DEMO_TEXTS = list(CFG.demo_queries) + [
    "a man is riding a horse", "a woman is cooking in the kitchen", "a cat is playing with a toy",
    "a man is cutting an onion with a knife", "a boy is playing the piano", "people are dancing",
    "a man is driving a car on the road", "a girl is swimming in the pool", "a dog is eating food",
    "a man is slicing bread", "a person is pouring water into a glass", "two men are fighting",
    "a woman is riding a bicycle", "a monkey is climbing a tree", "a man is singing on stage"]
try:
    export_msvd_demo_bundle(_DEMO_TEXTS, part=EVALS["explore"]["part"], name="msvd")
except Exception as _ex:                                     # phần phụ — không làm hỏng phần còn lại của notebook
    import traceback; traceback.print_exc()
    print("⚠ Không xuất được bundle demo:", repr(_ex))
'''


def main():
    nb = json.load(open(NB))
    src = lambda c: "".join(c["source"])
    if any(MARK in src(c) for c in nb["cells"]):
        print("Đã patch rồi:", NB)
        return
    i_run = next(i for i, c in enumerate(nb["cells"]) if "def run_exp(" in src(c))
    s = src(nb["cells"][i_run])
    assert OLD_TRAIN in s and OLD_SAVE in s
    s = s.replace(OLD_TRAIN, NEW_TRAIN).replace(OLD_SAVE, NEW_SAVE)
    s = s.replace("def run_exp(", RESUME_HELPERS.lstrip("\n") + "def run_exp(", 1)
    nb["cells"][i_run]["source"] = s
    i_demo = next(i for i, c in enumerate(nb["cells"]) if 'save_json(DEMO_OUT, RESULTS / "demo_outputs.json")' in src(c))
    code = dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=CELL)
    nb["cells"].insert(i_demo + 1, code)
    json.dump(nb, open(NB, "w"), ensure_ascii=False, indent=1)
    print("Đã patch:", NB)


if __name__ == "__main__":
    main()
