# -*- coding: utf-8 -*-
"""Truy vấn tự do trên bundle của phần A (Action Genome) — port đúng demo_query của notebook phần A.

Bundle do export_demo_bundle (notebook phần A, mục 1.17) ghi ra:
    meta.json, arrays.npz (gallery, img_feat, h_img, path_w, idf, qproj.*), R_<mẫu>.npz (hàng = gallery),
    A_scene/A_related/A_isa.npz (node × node), A_triple.npz (hàng = gallery), reference_outputs.json, thumbs/.

Cơ chế (giống hệt notebook):
    ứng viên = top-rerank_k của gallery theo cos CLIP
    s_gnn  = ⟨normalize(q_proj(q)), h_ảnh⟩            (nhúng GNN tính sẵn trên đồ thị đầy đủ)
    s_path = Σ_mẫu w_mẫu · log1p(Σ_{khái niệm truy vấn} R_mẫu[ảnh, khái niệm])
    điểm   = α·z(cos) + (1−α)·(λ·z(s_gnn) + (1−λ)·z(s_path))     (z theo hàng, ddof = 0 như bản numpy)
    giải thích = các đường đi cụ thể ảnh → khái niệm truy vấn có đóng góp biên lớn nhất vào s_path.
"""
import json, math, os, re
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def znorm(x, eps=1e-6):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + eps)


def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class _Lemma:
    """WordNetLemmatizer như notebook; thiếu dữ liệu WordNet thì giữ nguyên từ (ghi rõ trong meta của kết quả)."""

    def __init__(self):
        self.ok, self.cache = False, {}
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
            self.lem = WordNetLemmatizer()
            self.lem.lemmatize("cups", "n")
            self.ok = True
        except Exception as ex:                               # pragma: no cover
            print("WordNet không dùng được, bỏ lemma:", repr(ex)[:200])

    def __call__(self, w, pos="n"):
        k = (w, pos)
        if k not in self.cache:
            try:
                self.cache[k] = self.lem.lemmatize(w, pos) if self.ok else w
            except Exception:
                self.cache[k] = w
        return self.cache[k]


class PartABundle:
    def __init__(self, bundle_dir, encode_text=None):
        d = Path(bundle_dir)
        self.dir = d
        self.meta = m = json.load(open(d / "meta.json"))
        z = np.load(d / "arrays.npz")
        self.gallery = z["gallery"]
        self.img_feat, self.h_img = z["img_feat"].astype(np.float32), z["h_img"].astype(np.float32)
        self.path_w, self.idf = z["path_w"].astype(np.float32), z["idf"].astype(np.float32)
        self.qp = {k[6:]: z[k].astype(np.float32) for k in z.files if k.startswith("qproj.")}
        self.patterns = m["patterns"]
        self.R = {p: sp.load_npz(d / f"R_{p}.npz").tocsr() for p in self.patterns}
        self.A_scene = sp.load_npz(d / "A_scene.npz").tocsr()
        self.A_related = sp.load_npz(d / "A_related.npz").tocsr()
        self.A_isa = sp.load_npz(d / "A_isa.npz").tocsr()
        self.A_triple = sp.load_npz(d / "A_triple.npz").tocsr() if (d / "A_triple.npz").exists() else None
        self.items = m["items"]
        self.local = {it["pos"]: j for j, it in enumerate(self.items)}       # vị trí ảnh → hàng trong bundle
        self.name2id = m["name2id"]
        self.id2name = {int(k): v for k, v in m["id2name"].items()}
        self.scene_pred = {(a, b): r for a, b, r in m["scene_pred"]}
        self.tri2id = {tuple(t): i for t, i in m["tri2id"]}
        self.stop = set(m["stop"])
        self.pdesc = m["pattern_desc"]
        self.pidx = {p: i for i, p in enumerate(self.patterns)}
        self.alpha, self.lam, self.K = m["alpha"], m["lam"], m["rerank_k"]
        pc = m["parse"]
        self.parse_kind = pc.get("kind", "default")
        if self.parse_kind == "ag":
            self.lemma = _Lemma()
            self.verb_alias, self.noun_alias = pc["verb_alias"], pc["noun_alias"]
            self.verbs, self.action_nodes = set(pc["verbs"]), set(pc["action_nodes"])
        self.encode_text = encode_text
        thumbs = d / "thumbs"
        self.thumb = {it["id"]: str(thumbs / f"{it['id']}.jpg") for it in self.items if (thumbs / f"{it['id']}.jpg").exists()}

    # ------------------------------------------------------------------ phân tích truy vấn
    def parse(self, text):
        if self.parse_kind == "ag":
            return self._parse_ag(text)
        toks = [t for t in _norm(text).split() if t not in self.stop and len(t) > 1]
        cands = [" ".join(toks[i:i + n]) for n in (2, 1) for i in range(len(toks) - n + 1)]
        hits, seen = [], set()
        for g in cands:
            nid = self.name2id.get(g)
            if nid is not None and nid not in seen:
                seen.add(nid); hits.append((nid, g, g))
        return hits

    def _parse_ag(self, text):
        """= ag_parse_query + ag_temporal_hits của notebook."""
        L = self.lemma
        toks = []
        for t in _norm(text).split():
            if t in self.stop or len(t) < 2:
                continue
            v = L(t, "v"); v = self.verb_alias.get(v, v)
            if v in self.verbs:
                toks.append(v); continue
            n = L(t, "n"); toks.append(self.noun_alias.get(n, n))
        cands = [" ".join(toks[i:i + n]) for n in (2, 1) for i in range(len(toks) - n + 1)]
        hits, seen = [], set()
        for g in cands:
            nid = self.name2id.get(g)
            if nid is not None and nid not in seen:
                seen.add(nid); hits.append((nid, g, g))
        if self.meta.get("n_tri"):
            big = {i for i in range(len(toks) - 1) if " ".join(toks[i:i + 2]) in self.action_nodes}
            order = sorted([(i, " ".join(toks[i:i + 2])) for i in big] +
                           [(i, toks[i]) for i in range(len(toks)) if i not in big and (i - 1) not in big])
            acts = [n for n in dict.fromkeys(g for _, g in order) if n in self.action_nodes]
            for a, b in zip(acts, acts[1:]):
                nid = self.tri2id.get((a, "before", b))
                if nid is not None and nid not in seen:
                    seen.add(nid); hits.append((nid, f"{a} before {b}", f"{a} … {b}"))
        return hits

    # ------------------------------------------------------------------ chấm điểm
    def q_proj(self, q):
        W0, b0, W2, b2 = self.qp["0.weight"], self.qp["0.bias"], self.qp["2.weight"], self.qp["2.bias"]
        x = q @ W0.T + b0
        x = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))     # GELU chính xác (nn.GELU mặc định)
        x = x @ W2.T + b2
        return x / max(np.linalg.norm(x), 1e-12)

    def path_feats(self, local_rows, cids):
        out = np.zeros((len(local_rows), len(self.patterns)), np.float32)
        if cids:
            cols = np.array(sorted(set(cids)))
            for p, name in enumerate(self.patterns):
                out[:, p] = np.asarray(self.R[name][local_rows][:, cols].sum(1)).ravel()
        return np.log1p(out)

    def search(self, text, k=5, q_vec=None):
        q = np.asarray(q_vec if q_vec is not None else self.encode_text(text), np.float32).ravel()
        hits = self.parse(text)
        cids = [n for n, _, _ in hits]
        sim = self.img_feat @ q
        top = np.argsort(-sim, kind="stable")[:self.K]
        s_clip = sim[top]
        s_gnn = self.h_img[top] @ self.q_proj(q)
        pm = self.path_feats(top, cids)
        s_path = pm @ self.path_w
        s = self.alpha * znorm(s_clip[None])[0] + (1 - self.alpha) * (
            self.lam * znorm(s_gnn[None])[0] + (1 - self.lam) * znorm(s_path[None])[0])
        order = np.argsort(-s, kind="stable")[:k]
        res = []
        for rank, j in enumerate(order, 1):
            row = int(top[j]); it = self.items[row]
            expl, full, det = self.explain(row, cids)
            res.append(dict(rank=rank, pos=it["pos"], id=it["id"], caption=it["caption"], objects=it["objects"],
                            thumb=self.thumb.get(it["id"]), clip_cos=float(s_clip[j]), clip_rank=int(j) + 1,
                            score=float(s[j]), s_gnn=float(s_gnn[j]), s_path=float(s_path[j]),
                            part_clip=float(self.alpha * znorm(s_clip[None])[0][j]),
                            part_graph=float(s[j] - self.alpha * znorm(s_clip[None])[0][j]),
                            explanation=expl, n_paths=det["n_paths"], fidelity=det))
        return dict(query=text, concepts=[[int(n), g] for n, g, _ in hits], alpha=self.alpha, lam=self.lam,
                    gallery_size=len(self.items), results=res)

    # ------------------------------------------------------------------ giải thích (= enumerate_paths, explain_and_score)
    @staticmethod
    def _nbrs(M, n):
        sl = slice(M.indptr[n], M.indptr[n + 1])
        return zip(M.indices[sl].tolist(), M.data[sl].tolist())

    def enumerate_paths(self, row, concept_ids, cap=100000):
        target = set(concept_ids)
        objs = self.items[row]["img_objects"]
        own = set(objs)
        out = []

        def add(pattern, nodes, rel=None, ew=1.0):
            c = nodes[-1]
            if pattern != "P0_contains" and self.meta["drop_redundant_indirect"] and c in own:
                return
            w = float(self.idf[c]) * float(ew)
            if w > 0:
                out.append(dict(pattern=pattern, nodes=nodes, rel=rel, w=w))

        for o in objs:
            if o in target:
                add("P0_contains", [o])
            for o2, w1 in self._nbrs(self.A_scene, o):
                rel = self.scene_pred.get((o, o2), "quan hệ")
                if o2 in target:
                    add("P1_scene", [o, o2], rel, w1)
                for c, w2 in self._nbrs(self.A_isa, o2):
                    if c in target:
                        add("P4_scene_isa", [o, o2, c], rel, w1 * w2)
            for c, w1 in self._nbrs(self.A_isa, o):
                if c in target:
                    add("P2_isa", [o, c], None, w1)
            for o2, w1 in self._nbrs(self.A_related, o):
                if o2 in target:
                    add("P3_related", [o, o2], None, w1)
                for c, w2 in self._nbrs(self.A_isa, o2):
                    if c in target:
                        add("P5_related_isa", [o, o2, c], None, w1 * w2)
            if len(out) > cap:
                break
        if self.A_triple is not None and self.A_triple.nnz:
            for t, v in self._nbrs(self.A_triple, row):
                if int(t) in target:
                    add("P6_triple", [int(t)], None, v)
        return out[:cap]

    def _counts(self, paths):
        v = np.zeros(len(self.patterns), np.float64)
        for p in paths:
            v[self.pidx[p["pattern"]]] += p["w"]
        return np.log1p(v).astype(np.float32)

    def path_text(self, p):
        n = [self.id2name.get(x, f"#{x}") for x in p["nodes"]]
        d = dict(c=n[-1], o=n[0], rel=p.get("rel") or "quan_hệ")
        if len(n) >= 3:
            d["o2"] = n[1]
        return self.pdesc[p["pattern"]].format(**d)

    def explain(self, row, concept_ids, top_m=None):
        top_m = top_m or self.meta["top_m_paths"]
        paths = self.enumerate_paths(row, concept_ids)
        score = lambda ps: float((self._counts(ps) * self.path_w).sum())
        full = score(paths)
        contrib = [full - score(paths[:i] + paths[i + 1:]) for i in range(len(paths))]
        order = np.argsort(-np.array(contrib)) if contrib else np.array([], int)
        m = min(top_m, len(paths) - 1) if len(paths) >= 2 else len(paths)
        expl_idx = list(order[:m])
        expl = [dict(text=self.path_text(paths[i]), pattern=paths[i]["pattern"], contribution=float(contrib[i]),
                     nodes=[self.id2name.get(x, f"#{x}") for x in paths[i]["nodes"]], rel=paths[i].get("rel"))
                for i in expl_idx]
        keep = [paths[i] for i in expl_idx]
        drop = [p for i, p in enumerate(paths) if i not in set(expl_idx)]
        rng = np.random.RandomState(0)
        rnd = []
        for _ in range(10):
            ridx = set(rng.choice(len(paths), size=min(len(expl_idx), len(paths)), replace=False).tolist()) if paths else set()
            rnd.append(score([p for i, p in enumerate(paths) if i not in ridx]))
        detail = dict(n_paths=len(paths), score_full=full, score_without_explanation=score(drop),
                      score_only_explanation=score(keep),
                      score_without_random=float(np.mean(rnd)) if (rnd and len(paths) >= 2) else float("nan"),
                      sparsity=1 - len(expl_idx) / max(len(paths), 1))
        return expl, full, detail
