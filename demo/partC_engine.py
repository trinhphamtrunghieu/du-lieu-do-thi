# -*- coding: utf-8 -*-
"""Truy vấn tự do trên bundle của phần C (MSVD) — port đúng demo_query của notebook phần C (mô hình G3 = QC-RGAT).

Bundle do export_msvd_demo_bundle (notebook phần C, mục 4(b)) ghi ra: bundle.pt, items.json, reference_outputs.json,
thumbs/. Mã xử lý truy vấn và lớp QCRGAT nằm trong demo/partC_lib.py (trích nguyên văn từ notebook).

Cơ chế (giống notebook):
    ứng viên = top-rerank_k của gallery theo cos CLIP (video = trung bình khung hình)
    s_graph  = ⟨normalize(qo(u)), W_v·x_v + Σ_c Acon[v, c]·H_q[c]⟩ với H_q = trạng thái node lớp sau attention
               phụ thuộc truy vấn trên cạnh quan hệ (vai trò từ, vị từ khớp)
    điểm     = α·z(cos) + (1−α)·z(s_graph)
    giải thích = các số hạng ĐỒ THỊ dương lớn nhất trong phân rã chính xác của s_graph (contains + đường đi quan hệ).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

import partC_lib as L


class _Offset:
    """Truy cập theo id node TOÀN CỤC (n_vid + chỉ số lớp) như INV.name / INV.pos của notebook."""

    def __init__(self, vals, base):
        self.vals, self.base = vals, base

    def __getitem__(self, i):
        return self.vals[int(i) - self.base]


class _Inv:
    def __init__(self, inv):
        self.n_vid = inv["n_vid"]
        self.key2id = inv["key2id"]
        self.concepts_by_lemma = defaultdict(list, {tuple(k): v for k, v in inv["concepts_by_lemma"]})
        self.name = _Offset(inv["class_names"], self.n_vid)
        self.pos = _Offset(inv["class_pos"], self.n_vid)


class _Data:
    """Thay cho RelDataV4: cùng tên thuộc tính mà QCRGAT dùng; video đánh chỉ số CỤC BỘ trong gallery."""

    def __init__(self, Db, use_pred):
        self.N, self.ns, self.E = Db["N"], Db["ns"], Db["E"]
        self.src_np, self.dst_np = np.asarray(Db["src"]), np.asarray(Db["dst"])
        self.src, self.dst = torch.from_numpy(self.src_np), torch.from_numpy(self.dst_np)
        self.et = torch.from_numpy(np.asarray(Db["et"]))
        self.logw = torch.from_numpy(np.asarray(Db["logw"], np.float32))
        self.PE = torch.from_numpy(np.asarray(Db["PE"], np.float32))
        self.Mex = sp.csr_matrix(Db["Mex"])
        self.Acon = torch.from_numpy(sp.csr_matrix(Db["Acon"]).toarray().astype(np.float32))
        self._qb = {}

    def query_batch(self, q_idx, use_pred=True):
        return self._qb[q_idx[0]]

    def make_query(self, key, text, tq, use_pred):
        """= RelDataV4.query_batch cho MỘT truy vấn văn bản tự do (V4_HITS / V4_ROLES / QPREDS như demo_query)."""
        hits = sorted({nid for nid, _, _ in L.parse_tokens_v2(L.query_tokens_v2(text), L.INV)})
        roles = L.query_roles_v4([text], L.INV)[0]
        qpreds = {rel: L.match_predicates(L.query_pred_tokens(text), rel) for rel in L.PRED_FAMILIES}
        role_src = torch.zeros(1, self.ns, dtype=torch.long)
        for h in hits:
            r = roles.get(h, L.Q_ROLES.index("verb" if L.INV.pos[h] == "v" else "other"))
            role_src[0, h - self.N] = r
        ex = np.zeros((1, self.E), np.float32)
        qp = np.zeros((1, L.PRED_EMB.shape[1]), np.float32)
        if use_pred and self.E:
            ids = [p for rel in L.PRED_FAMILIES for p in qpreds[rel]]
            if ids:
                ex[0] = np.minimum(np.asarray(self.Mex[ids].sum(0)).ravel(), 1.0)
                m = L.PRED_EMB[ids].mean(0)
                qp[0] = m / max(np.linalg.norm(m), 1e-8)
        self._qb[key] = dict(q=np.array([key]), role_src=role_src, ex=torch.from_numpy(ex),
                             soft=torch.from_numpy(qp) @ self.PE.T, tq=torch.from_numpy(np.asarray(tq, np.float32)[None]))
        return hits


class PartCBundle:
    def __init__(self, bundle_dir, encode_text=None):
        d = Path(bundle_dir)
        self.dir = d
        B = torch.load(d / "bundle.pt", map_location="cpu", weights_only=False)
        self.B = B
        L.INV = _Inv(B["inv"])
        L.PRED_VOCAB = [tuple(x) for x in B["pred_vocab"]]
        L.PRED_EMB = np.asarray(B["pred_emb"], np.float32)
        L.PRED_FAMILIES, L.Q_ROLES, L.V4_REL_TYPES = tuple(B["pred_families"]), tuple(B["q_roles"]), list(B["rel_types"])
        L._PRED_BY_FIRST.clear()
        L.CFG.top_m_paths = B["top_m"]
        L.VID_T = torch.from_numpy(np.asarray(B["vid_t"], np.float32))
        c = B["ctor"]
        self.D = _Data(B["D"], c["use_pred"])
        torch.manual_seed(0)
        self.model = L.QCRGAT(self.D, c["clip_dim"], d=c["d"], T=c["T"], rel=c["rel"], qcond=c["qcond"],
                              use_pred=c["use_pred"])
        self.model.load_state_dict(B["state"])
        self.model.eval()
        self.vid_feat = np.asarray(B["vid_feat"], np.float32)
        self.alpha, self.K, self.top_m = B["alpha"], B["rerank_k"], B["top_m"]
        self.items = json.load(open(d / "items.json"))
        self.video_names = B["inv"]["video_names"]
        self.edge_text = B["edge_text"]
        self.encode_text = encode_text
        thumbs = d / "thumbs"
        self.thumb = {it["id"]: str(thumbs / f"{it['id']}.jpg") for it in self.items if (thumbs / f"{it['id']}.jpg").exists()}
        self.meta = dict(name=B["name"], part=B["part"], clip=B["clip"], alpha=self.alpha, rerank_k=self.K)

    def term_text(self, t, v):
        vid = self.video_names[v]
        if t["kind"] == "contains":
            return f"{vid} –chứa→ {L.INV.name[self.D.N + t['node']]}"
        hops = [self.edge_text[int(k)] for k in t["edges"]]
        chain = hops[0][0] + "".join(f" –{r}→ {b}" for _, r, b in hops)
        return f"{chain} ←chứa– {vid}"

    @torch.no_grad()
    def search(self, text, k=5, q_vec=None):
        t = np.asarray(q_vec if q_vec is not None else self.encode_text(text), np.float32).ravel()
        key = len(self.D._qb)
        hits = self.D.make_query(key, text, t, self.model.use_pred)
        cos = self.vid_feat @ t
        top = np.argsort(-cos)[:self.K]
        cand, s_clip = top, cos[top]
        s = self.model.score(self.D._qb[key], torch.from_numpy(cand[None])).numpy()[0]
        zc, zs = L.znorm_per_row(s_clip[None])[0], L.znorm_per_row(s[None])[0]
        fin = self.alpha * zc + (1 - self.alpha) * zs
        order = np.argsort(-fin)[:k]
        res = []
        for r_, j in enumerate(order, 1):
            v = int(cand[j])
            ex = L.explain_full(self.model, key, v)
            res.append(dict(rank=r_, local=v, video_id=self.video_names[v], thumb=self.thumb.get(self.video_names[v]),
                            clip_cos=float(s_clip[j]), graph_score=float(s[j]), final=float(fin[j]), clip_rank=int(j) + 1,
                            part_clip=float(self.alpha * zc[j]), part_graph=float((1 - self.alpha) * zs[j]),
                            totals={k_: round(v_, 4) for k_, v_ in ex["totals"].items()},
                            explanation=[f"{tt['contrib']:+.4f} {self.term_text(tt, v)}" for tt in ex["support"]],
                            support=[dict(kind=tt["kind"], contrib=float(tt["contrib"]), text=self.term_text(tt, v),
                                          hops=[self.edge_text[int(e)] for e in tt["edges"]],
                                          node=L.INV.name[self.D.N + tt["node"]] if tt["node"] is not None else None)
                                     for tt in ex["support"]],
                            note="" if ex["support"] else "KHÔNG có đường đi giải thích hợp lệ (không số hạng đồ thị nào ủng hộ)"))
        self.D._qb.pop(key, None)
        return dict(query=text, gallery=self.meta["part"], alpha=self.alpha,
                    matched_concepts=[L.INV.name[h] for h in hits], gallery_size=len(self.items), results=res)
