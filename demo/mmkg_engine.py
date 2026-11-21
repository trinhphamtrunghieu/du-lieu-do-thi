# -*- coding: utf-8 -*-
"""Bộ máy truy vấn cho demo: văn bản → top-k ảnh + đường đi giải thích trên MMKG Ẩm thực Việt Nam.

Giữ đúng cơ chế xếp hạng của `mmkg/Project3_MMKG_VN.ipynb` (mục 5 và 9):

    s(q, i) = α·z(cos(q, i)) + (1−α)·z( Σ_d P(d|q) · [λ·p_GNN(d|i) + (1−λ)·p_kNN(d|i)] )

- P(d|q): liên kết thực thể trong truy vấn với node KG, điểm đường đi 1 bước / 2 bước (trọng số w2),
  trộn với độ tương đồng CLIP giữa truy vấn và mô tả node món (η), softmax nhiệt độ T.
- p(d|i): ảnh → món bằng GraphSAGE quan hệ (R-SAGE) trên MMKG và bằng kNN trên ảnh train.
- α, λ, η, T, w2 chọn trên bộ truy vấn VALIDATION của seed đã chọn, không nhìn test.

Giải thích lấy thẳng từ các số hạng của điểm đồ thị: món trung gian d* có đóng góp P(d|q)·p(d|i) lớn nhất,
đường đi KG nối thực thể trong truy vấn với d*, và ảnh train gần nhất có cạnh `mô_tả` → d*.
`fidelity()` xoá đúng các cạnh đó rồi chấm lại để kiểm tra giải thích có thật sự tác động lên điểm.
"""
import os, sys, re, json, math, time, random, itertools, unicodedata, hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {"mclip_b32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),          # nhẹ, chạy được trên CPU
          "mclip_h14": ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k")}  # mô hình chính của báo cáo (cần GPU)

GRID = dict(alpha=[round(a, 2) for a in np.linspace(0, 1, 11)], lam=[0, 0.25, 0.5, 0.75, 1.0],
            eta=[0, 0.25, 0.5, 0.75, 1.0], T=[0.03, 0.1, 0.3], w2=[0.5, 1.0])

ALIASES_VI = {"TP. Hồ Chí Minh": ["sài gòn", "hồ chí minh", "tp hcm"], "Bắc Bộ": ["miền bắc"],
              "Trung Bộ": ["miền trung"], "Nam Bộ": ["miền nam"]}


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)


def softmax(X, T):
    X = X / T
    X = X - X.max(1, keepdims=True)
    E = np.exp(X)
    return E / E.sum(1, keepdims=True)


def norm_text(s, strip):
    s = unicodedata.normalize("NFC", s.lower())
    if strip:
        s = unicodedata.normalize("NFD", s.replace("đ", "d"))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^\w]+", " ", s).strip() + " "


class RSAGE(nn.Module):
    """GraphSAGE quan hệ: mỗi loại cạnh (kể cả chiều ngược) có ma trận riêng, gộp trung bình."""

    def __init__(self, d_in, d_h, n_rel, layers=2, drop=0.2):
        super().__init__()
        self.inp = nn.Linear(d_in, d_h)
        self.selfs = nn.ModuleList([nn.Linear(d_h, d_h) for _ in range(layers)])
        self.rels = nn.ModuleList([nn.ModuleList([nn.Linear(d_h, d_h, bias=False) for _ in range(n_rel)])
                                   for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(layers)])
        self.drop = drop

    def forward(self, x, edges):
        h = self.inp(x)
        for l in range(len(self.selfs)):
            m = self.selfs[l](h)
            for r, (src, dst) in enumerate(edges):
                if len(src) == 0:
                    continue
                msg = self.rels[l][r](h[src])
                agg = torch.zeros_like(h).index_add_(0, dst, msg)
                deg = torch.zeros(h.shape[0], 1).index_add_(0, dst, torch.ones(len(dst), 1)).clamp(min=1)
                m = m + agg / deg
            h = h + F.dropout(F.gelu(self.norms[l](m)), self.drop, self.training)
        return F.normalize(h, dim=-1)


def find_repo_root(start=None):
    p = os.path.abspath(start or os.getcwd())
    for _ in range(6):
        if os.path.exists(os.path.join(p, "mmkg", "kg_data.py")):
            return p
        p = os.path.dirname(p)
    raise FileNotFoundError("Không tìm thấy thư mục repo (cần mmkg/kg_data.py). Đặt biến môi trường P3_REPO.")


class MMKGSearch:
    """Dựng KG, trích đặc trưng CLIP, huấn luyện GNN, chọn tham số trên val; sau đó gọi `search()`."""

    def __init__(self, repo=None, model_key=None, seed=0, cache_dir=None, device=None, smoke=None, verbose=True):
        self.repo = repo or os.environ.get("P3_REPO") or find_repo_root(os.path.dirname(os.path.abspath(__file__)))
        self.mmkg_dir = os.path.join(self.repo, "mmkg")
        self.model_key = model_key or os.environ.get("P3_DEMO_MODEL", "mclip_b32")
        self.seed = seed
        self.cache_dir = cache_dir or os.path.join(self.repo, "demo", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = device or os.environ.get("P3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.smoke = (os.environ.get("P3_DEMO_SMOKE", "0") == "1") if smoke is None else smoke
        self.verbose = verbose
        self._clip = None
        t0 = time.time()
        self._build_kg()
        self._load_split_and_queries()
        self._features()
        self._train_graph_parts()
        self._select_hparams()
        self.log(f"Sẵn sàng sau {time.time() - t0:.0f}s | mô hình {MODELS[self.model_key][0]} | "
                 f"tham số chọn trên val: {self.hp} (val MRR {self.val_mrr:.3f})")

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ KG
    def _build_kg(self):
        sys.path.insert(0, self.mmkg_dir)
        import kg_data
        self.kg = kg_data
        ents, kg_triples, self.slug = kg_data.build_kg()
        self.ents, self.kg_triples = ents, kg_triples
        self.ENT = {e["id"]: e for e in ents}
        self.EIDX = {e["id"]: i for i, e in enumerate(ents)}
        self.DISHES = [e["id"] for e in ents if e["kind"] == "món"]
        self.DIDX = {d: i for i, d in enumerate(self.DISHES)}
        imgs = pd.read_csv(os.path.join(self.mmkg_dir, "images.csv"))
        imgs["path"] = imgs["file"].map(lambda f: os.path.join(self.mmkg_dir, f))
        missing = (~imgs["path"].map(os.path.exists)).sum()
        assert missing == 0, f"thiếu {missing} file ảnh trong mmkg/images"
        self.imgs = imgs
        self.IMG_IDS = list(imgs.image_id)
        self.IIDX = {i: k for k, i in enumerate(self.IMG_IDS)}
        self.IMG_DISH = np.array([self.DIDX[d] for d in imgs.dish_id])
        R = kg_data.REL
        self.REL = R
        self.ATTR_RELS = [R[k] for k in ("type", "origin", "method", "ingredient")]
        self.LINK_RELS = [R["group"], R["region"]]
        self.R1, self.R2, self.PATH2 = self.reach_mats(kg_triples)
        tab = []
        for strip in (False, True):
            t = []
            for e in ents:
                for n in [e["name"]] + ALIASES_VI.get(e["name"], []):
                    t.append((norm_text(n, strip), e["id"]))
            t.sort(key=lambda x: -len(x[0]))
            tab.append(t)
        self.ALIAS = {False: tab[0], True: tab[1]}
        self.log(f"KG: {len(ents)} thực thể, {len(kg_triples)} triple tri thức, {len(imgs)} ảnh")

    def reach_mats(self, triples):
        R1 = np.zeros((len(self.DISHES), len(self.ents)), np.float32)
        R2 = np.zeros_like(R1)
        up = defaultdict(list)
        for h, r, t in triples:
            if r in self.LINK_RELS:
                up[h].append((r, t))
        path2 = defaultdict(list)
        for h, r, t in triples:
            if h in self.DIDX and r in self.ATTR_RELS:
                R1[self.DIDX[h], self.EIDX[t]] = 1
                for r2, t2 in up.get(t, []):
                    R2[self.DIDX[h], self.EIDX[t2]] = 1
                    path2[(h, t2)].append((r, t, r2))
        return R1, R2, path2

    def link_entities(self, text):
        nfd = unicodedata.normalize("NFD", text)
        has_diac = any(unicodedata.category(c) == "Mn" for c in nfd) or "đ" in text.lower()
        s = norm_text(text, not has_diac)
        found = []
        for a, e in self.ALIAS[not has_diac]:
            if a in s:
                found.append(e)
                s = s.replace(a, " # ")
        return found

    def dish_desc(self, d):
        by = defaultdict(list)
        for h, r, t in self.kg_triples:
            if h == d:
                by[r].append(self.ENT[t]["name"])
        R = self.REL
        s = f"{self.ENT[d]['name']}, {', '.join(by[R['type']])}"
        if by[R["origin"]]: s += f", xuất xứ {', '.join(by[R['origin']])}"
        if by[R["method"]]: s += f", {' và '.join(by[R['method']])}"
        if by[R["ingredient"]]: s += f", gồm {', '.join(by[R['ingredient']])}"
        return s

    # ------------------------------------------------------------------ split + truy vấn val
    def _load_split_and_queries(self):
        """Dùng đúng split và bộ truy vấn đã lưu từ lần chạy Kaggle (results/mmkg)."""
        rdir = os.path.join(self.repo, "results", "mmkg")
        splits = json.load(open(os.path.join(rdir, "splits.json")))
        self.split = splits[str(self.seed)]
        qs = json.load(open(os.path.join(rdir, "queries.json")))
        self.queries = qs
        self.val_queries = [q for q in qs if q["use"] in ("val", "both")]
        self.test_queries = [q for q in qs if q["use"] in ("test", "both")]
        self.gal = {p: [self.IIDX[i] for i in self.IMG_IDS if self.split[i] == p] for p in ("train", "val", "test")}
        self.gal["all"] = list(range(len(self.IMG_IDS)))
        self.log("Split seed", self.seed, {p: len(v) for p, v in self.gal.items()})

    # ------------------------------------------------------------------ CLIP
    def _load_clip(self):
        if self._clip is None:
            import open_clip
            arch, pt = MODELS[self.model_key]
            self.log(f"Nạp CLIP {arch}/{pt} trên {self.device} …")
            m, _, prep = open_clip.create_model_and_transforms(arch, pretrained=pt, device=self.device)
            self._clip = (m.eval(), prep, open_clip.get_tokenizer(arch))
        return self._clip

    def _tokenize(self, tok, texts):
        try:
            return tok(texts)
        except AttributeError:   # open_clip 3.2.0 + transformers mới
            return tok.tokenizer(list(texts), return_tensors="pt", max_length=tok.context_length,
                                 padding="max_length", truncation=True).input_ids

    def _fake(self, s, dim=64):
        v = np.random.default_rng(int(hashlib.md5(s.encode()).hexdigest()[:8], 16)).normal(size=dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts):
        if self.smoke:
            return np.stack([self._fake(t) for t in texts])
        m, _, tok = self._load_clip()
        out = []
        for b in range(0, len(texts), 64):
            t = self._tokenize(tok, texts[b:b + 64]).to(self.device)
            out.append(F.normalize(m.encode_text(t).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def encode_images(self, paths):
        if self.smoke:
            P = {d: self._fake("dish" + d) for d in self.DISHES}
            out = []
            for k, p in enumerate(paths):
                v = 2 * P[self.DISHES[self.IMG_DISH[k]]] + self._fake(p)
                out.append(v / np.linalg.norm(v))
            return np.stack(out).astype(np.float32)
        from PIL import Image
        m, prep, _ = self._load_clip()
        out = []
        for b in range(0, len(paths), 32):
            x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[b:b + 32]]).to(self.device)
            out.append(F.normalize(m.encode_image(x).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def _features(self):
        texts = sorted({e["name"] for e in self.ents} | {self.dish_desc(d) for d in self.DISHES}
                       | {q["vi"] for q in self.queries})
        tag = "smoke" if self.smoke else self.model_key
        cache = os.path.join(self.cache_dir, f"feats_{tag}.npz")
        key = hashlib.md5(("\n".join(texts) + "\n".join(self.imgs.sha256)).encode()).hexdigest()
        self.feat_key = key
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            if str(z["key"]) == key:
                self.I, self.Ttab = z["img"], dict(zip(z["texts"], z["txt"]))
                self.log("Đặc trưng CLIP lấy từ cache:", cache)
                return
        t0 = time.time()
        I = self.encode_images(list(self.imgs.path))
        T = self.encode_texts(texts)
        np.savez(cache, img=I, txt=T, texts=np.array(texts, dtype=object), key=key)
        self.I, self.Ttab = I, dict(zip(texts, T))
        self.log(f"Trích đặc trưng CLIP: {len(I)} ảnh, {len(T)} câu, {time.time() - t0:.0f}s → {cache}")

    def text_vec(self, text):
        if text not in self.Ttab:
            self.Ttab[text] = self.encode_texts([text])[0]
        return self.Ttab[text]

    # ------------------------------------------------------------------ GNN + kNN (ảnh → món)
    def _build_graph(self, ent_feat, train_ids, k_img=5):
        n_img = len(self.IMG_IDS)
        rel_names = sorted({r for _, r, _ in self.kg_triples})
        E = []
        for r in rel_names:
            src = [n_img + self.EIDX[h] for h, rr, t in self.kg_triples if rr == r]
            dst = [n_img + self.EIDX[t] for h, rr, t in self.kg_triples if rr == r]
            E.append((src, dst)); E.append((dst, src))
        tr = np.array([self.IIDX[i] for i in train_ids])
        S = self.I @ self.I[tr].T
        for k in range(n_img):
            S[k, np.nonzero(tr == k)[0]] = -9
        nb = np.argsort(-S, 1)[:, :k_img]
        E.append((list(tr[nb].ravel()), list(np.repeat(np.arange(n_img), k_img))))
        return np.concatenate([self.I, ent_feat]), E

    def _train_gnn(self, epochs=300, mask_frac=0.4, state=None):
        seed = self.seed
        torch.manual_seed(seed); np.random.seed(seed)
        labels = self.IMG_DISH
        train_ids = [i for i in self.IMG_IDS if self.split[i] == "train"]
        val_ids = [i for i in self.IMG_IDS if self.split[i] == "val"]
        ent_feat = np.stack([self.Ttab[e["name"]] for e in self.ents])
        X, E = self._build_graph(ent_feat, train_ids)
        X = torch.tensor(X, dtype=torch.float32)
        n_img = len(self.IMG_IDS)
        dish_nodes = torch.tensor([n_img + self.EIDX[d] for d in self.DISHES])
        lab = torch.tensor(labels)

        def with_links(ids):
            idx = [self.IIDX[i] for i in ids]
            d = [n_img + self.EIDX[self.DISHES[labels[k]]] for k in idx]
            return E + [(idx, d), (d, idx)]

        to_t = lambda E_: [(torch.tensor(s, dtype=torch.long), torch.tensor(d, dtype=torch.long)) for s, d in E_]
        model = RSAGE(X.shape[1], 256, len(E) + 2)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        tau = 0.07
        full_E = to_t(with_links(train_ids))
        val_idx = torch.tensor([self.IIDX[i] for i in val_ids], dtype=torch.long)
        if state is not None:            # GNN đã huấn luyện, lấy từ cache: không chạy optimizer
            model.load_state_dict(state["model"])
            model.eval()
            return model, X, full_E, dish_nodes, tau, state["val_acc"]
        best, best_state, bad = -1, None, 0
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(len(train_ids))
            hide = [train_ids[k] for k in perm[: int(mask_frac * len(train_ids))]]
            keep = [train_ids[k] for k in perm[int(mask_frac * len(train_ids)):]]
            h = model(X, to_t(with_links(keep)))
            tgt = torch.tensor([self.IIDX[i] for i in hide])
            loss = F.cross_entropy(h[tgt] @ h[dish_nodes].T / tau, lab[tgt])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 9 and len(val_ids):
                model.eval()
                with torch.no_grad():
                    h = model(X, full_E)
                    acc = (h[val_idx] @ h[dish_nodes].T).argmax(1).eq(lab[val_idx]).float().mean().item()
                if acc > best:
                    best, best_state, bad = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 8:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, X, full_E, dish_nodes, tau, best

    def gnn_probs(self, E_t):
        with torch.no_grad():
            h = self.gnn(self.X, E_t)
            return F.softmax(h[: len(self.IMG_IDS)] @ h[self.dish_nodes].T / self.tau, dim=1).numpy()

    def knn_probs(self, train_idx, k=10, temp=0.05):
        tr = np.array(sorted(train_idx))
        S = self.I @ self.I[tr].T
        P = np.zeros((len(self.IMG_IDS), len(self.DISHES)), np.float32)
        for n in range(len(self.IMG_IDS)):
            s = S[n].copy()
            s[tr == n] = -9
            nb = np.argsort(-s)[:k]
            w = np.exp((s[nb] - s[nb].max()) / temp)
            np.add.at(P[n], self.IMG_DISH[tr[nb]], w)
            P[n] /= P[n].sum()
        return P

    def _train_graph_parts(self):
        t0 = time.time()
        tag = "smoke" if self.smoke else self.model_key
        cache = os.path.join(self.cache_dir, f"gnn_{tag}_seed{self.seed}.pt")
        import inspect
        gnn_key = hashlib.md5((self.feat_key + json.dumps(self.split, sort_keys=True)
                               + inspect.getsource(MMKGSearch._train_gnn) + inspect.getsource(RSAGE)).encode()).hexdigest()
        state = None
        if os.path.exists(cache):
            st = torch.load(cache, map_location="cpu", weights_only=False)
            if st.get("key") == gnn_key:
                state = st
                self.log("GNN lấy từ cache:", cache)
        self.gnn, self.X, self.E_t, self.dish_nodes, self.tau, acc = self._train_gnn(state=state)
        if state is None:
            torch.save(dict(model=self.gnn.state_dict(), val_acc=acc, key=gnn_key), cache)
        self.P_gnn = self.gnn_probs(self.E_t)
        self.P_knn = self.knn_probs(self.gal["train"])
        self.desc_emb = np.stack([self.Ttab[self.dish_desc(d)] for d in self.DISHES])
        self.log(f"GNN (R-SAGE) sẵn sàng sau {time.time() - t0:.0f}s; độ chính xác ảnh→món trên val {acc:.3f}")

    # ------------------------------------------------------------------ chấm điểm
    def query_dish(self, text, qv, hp, R1=None, R2=None):
        """P(d|q) và các thành phần: điểm ràng buộc KG, điểm mô tả node, thực thể nhận ra."""
        R1 = self.R1 if R1 is None else R1
        R2 = self.R2 if R2 is None else R2
        es = self.link_entities(text)
        Skg = np.zeros((1, len(self.DISHES)), np.float32)
        cons = [e for e in es if e not in self.DIDX]
        if cons:
            Skg[0] = np.stack([np.maximum(R1[:, self.EIDX[e]], hp["w2"] * R2[:, self.EIDX[e]]) for e in cons]).mean(0)
        for e in es:
            if e in self.DIDX:
                Skg[0, self.DIDX[e]] = max(Skg[0, self.DIDX[e]], 1.0)
        Stxt = qv[None] @ self.desc_emb.T
        Q = softmax(hp["eta"] * Skg + (1 - hp["eta"]) * zrow(Stxt) * 0.1, hp["T"])
        return Q, Skg, es

    def score(self, text, gal, hp, qv=None, R1=None, R2=None, P_gnn=None, P_knn=None):
        qv = self.text_vec(text) if qv is None else qv
        Q, Skg, es = self.query_dish(text, qv, hp, R1, R2)
        P_gnn = self.P_gnn if P_gnn is None else P_gnn
        P_knn = self.P_knn if P_knn is None else P_knn
        Pimg = hp["lam"] * P_gnn[gal] + (1 - hp["lam"]) * P_knn[gal]
        S_clip = qv[None] @ self.I[gal].T
        S_graph = Q @ Pimg.T
        S = hp["alpha"] * zrow(S_clip) + (1 - hp["alpha"]) * zrow(S_graph)
        return dict(S=S[0], S_clip=S_clip[0], S_graph=S_graph[0], zc=zrow(S_clip)[0], zg=zrow(S_graph)[0],
                    Q=Q[0], Skg=Skg[0], Pimg=Pimg, entities=es, qv=qv)

    def _select_hparams(self):
        gal = self.gal["val"]
        qs = [q for q in self.val_queries
              if any(self.DISHES[self.IMG_DISH[k]] in q["gold"] for k in gal)]
        G = np.array([[self.DISHES[self.IMG_DISH[k]] in q["gold"] for k in gal] for q in qs])
        QV = np.stack([self.text_vec(q["vi"]) for q in qs])
        S_clip = zrow(QV @ self.I[gal].T)
        Stxt = zrow(QV @ self.desc_emb.T) * 0.1
        Skg = {}
        for w2 in GRID["w2"]:
            hp = dict(w2=w2, eta=1.0, T=1.0)
            Skg[w2] = np.stack([self.query_dish(q["vi"], QV[n], hp)[1][0] for n, q in enumerate(qs)])
        best = None
        for w2, eta, Tq, lam in itertools.product(GRID["w2"], GRID["eta"], GRID["T"], GRID["lam"]):
            Q = softmax(eta * Skg[w2] + (1 - eta) * Stxt, Tq)
            Zg = zrow(Q @ (lam * self.P_gnn[gal] + (1 - lam) * self.P_knn[gal]).T)
            for a in GRID["alpha"]:
                S = a * S_clip + (1 - a) * Zg
                bestg = np.where(G, S, -np.inf).max(1, keepdims=True)
                m = float((1.0 / (1 + (S > bestg).sum(1))).mean())
                if best is None or m > best[0] + 1e-9:
                    best = (m, dict(alpha=a, lam=lam, eta=eta, T=Tq, w2=w2))
        self.val_mrr = best[0]
        self.hp = {k: float(v) for k, v in best[1].items()}

    # ------------------------------------------------------------------ giải thích
    def explain_one(self, out, gpos, gal_ids, n_evidence=3):
        """Giải thích của ảnh ở vị trí gpos: món d*, đường đi KG, ảnh bằng chứng — lấy từ số hạng của điểm đồ thị."""
        contrib = out["Q"] * out["Pimg"][gpos]
        d_star = self.DISHES[int(contrib.argmax())]
        paths = []
        for e in out["entities"]:
            if e == d_star:
                paths.append([("TRUY_VẤN", "nhắc_tên", d_star)])
            elif e in self.DIDX:
                continue
            elif self.R1[self.DIDX[d_star], self.EIDX[e]]:
                rel = [rr for h, rr, t in self.kg_triples if h == d_star and t == e][0]
                paths.append([(d_star, rel, e)])
            elif (d_star, e) in self.PATH2:
                r1, mid, r2 = self.PATH2[(d_star, e)][0]
                paths.append([(d_star, r1, mid), (mid, r2, e)])
        unmatched = [e for e in out["entities"] if e not in self.DIDX and
                     not any(p[-1][2] == e for p in paths)]
        k = gal_ids[gpos]
        tr = np.array(self.gal["train"])
        order = tr[np.argsort(-(self.I[tr] @ self.I[k]))]
        evidence = [int(j) for j in order if self.IMG_DISH[j] == self.DIDX[d_star] and j != k][:n_evidence]
        top_d = np.argsort(-contrib)[:3]
        return dict(d_star=d_star, share=float(contrib.max() / (contrib.sum() + 1e-12)), paths=paths,
                    unmatched=unmatched, evidence=evidence,
                    top_dishes=[(self.DISHES[j], float(contrib[j] / (contrib.sum() + 1e-12))) for j in top_d])

    def search(self, query, k=5, gallery="test"):
        """Trả về dict: thực thể nhận ra, P(d|q) top món, và top-k ảnh, mỗi ảnh kèm điểm + giải thích."""
        gal = self.gal[gallery]
        out = self.score(query, gal, self.hp)
        order = np.argsort(-out["S"], kind="stable")[:k]
        res = []
        for rank, gp in enumerate(order, 1):
            kimg = gal[gp]
            ex = self.explain_one(out, gp, gal)
            res.append(dict(rank=rank, gpos=int(gp), img=int(kimg), image_id=self.IMG_IDS[kimg],
                            path=self.imgs.path[kimg], dish=self.DISHES[self.IMG_DISH[kimg]],
                            dish_name=self.ENT[self.DISHES[self.IMG_DISH[kimg]]]["name"],
                            split=self.split[self.IMG_IDS[kimg]],
                            score=float(out["S"][gp]), clip_cos=float(out["S_clip"][gp]),
                            part_clip=float(self.hp["alpha"] * out["zc"][gp]),
                            part_graph=float((1 - self.hp["alpha"]) * out["zg"][gp]),
                            author=self.imgs.author[kimg], license=self.imgs.license[kimg],
                            source=self.imgs.source_url[kimg], **ex))
        qd = np.argsort(-out["Q"])[:5]
        return dict(query=query, gallery=gallery, hp=self.hp, entities=out["entities"],
                    query_dishes=[(self.DISHES[j], float(out["Q"][j])) for j in qd],
                    results=res, _out=out, _gal=gal)

    # ------------------------------------------------------------------ fidelity
    def _rescore(self, query, gal, gpos, drop_triples, drop_links, qv):
        trip = [t for t in self.kg_triples if t not in drop_triples]
        R1, R2, _ = self.reach_mats(trip)
        dl = set(drop_links)
        P_knn = self.knn_probs([j for j in self.gal["train"] if j not in dl])
        n_img = len(self.IMG_IDS)
        E2 = []
        for (s, d) in self.E_t:
            s_, d_ = s.numpy(), d.numpy()
            keep = ~(np.isin(s_, list(dl)) & (d_ >= n_img)) & ~(np.isin(d_, list(dl)) & (s_ >= n_img))
            E2.append((s[torch.tensor(keep)], d[torch.tensor(keep)]))
        P_gnn = self.gnn_probs(E2)
        o = self.score(query, gal, self.hp, qv=qv, R1=R1, R2=R2, P_gnn=P_gnn, P_knn=P_knn)
        S = o["S"]
        return float(S[gpos]), int((S > S[gpos]).sum() + 1)

    def fidelity(self, result, item, n_random=5, seed=0):
        """Xoá các cạnh trong giải thích của `item` (triple KG + cạnh mô_tả của ảnh bằng chứng) rồi chấm lại.
        So với xoá cùng số cạnh ngẫu nhiên. Không huấn luyện lại GNN, nên đây là cận dưới."""
        S0 = result["_out"]["S"]; gal = result["_gal"]; gp = item["gpos"]; qv = result["_out"]["qv"]
        tri = {t for p in item["paths"] for t in p if t[0] != "TRUY_VẤN"}
        links = list(item["evidence"])
        s_e, r_e = self._rescore(result["query"], gal, gp, tri, links, qv)
        rng = random.Random(seed)
        pool = [t for t in self.kg_triples if t[1] in self.ATTR_RELS + self.LINK_RELS]
        rnd = []
        for _ in range(n_random):
            rt = set(rng.sample(pool, len(tri)))
            rl = rng.sample(self.gal["train"], len(links))
            rnd.append(self._rescore(result["query"], gal, gp, rt, rl, qv))
        r0 = int((S0 > S0[gp]).sum() + 1)
        return dict(n_edges=len(tri) + len(links), score_before=float(S0[gp]), rank_before=r0,
                    score_after_expl=s_e, rank_after_expl=r_e,
                    score_after_rand=float(np.mean([x[0] for x in rnd])),
                    rank_after_rand=float(np.mean([x[1] for x in rnd])))

    # ------------------------------------------------------------------ hiển thị
    def name(self, eid):
        return self.ENT[eid]["name"] if eid in self.ENT else ("truy vấn" if eid == "TRUY_VẤN" else eid)

    def path_str(self, path):
        s = self.name(path[0][0])
        for h, r, t in path:
            s += f" —{r}→ {self.name(t)}"
        return s

    def print_result(self, R):
        print(f"Truy vấn: {R['query']!r}   (gallery: {R['gallery']}, {len(R['_gal'])} ảnh)")
        print("Thực thể nhận ra trong KG:", [f"{self.name(e)} [{self.ENT[e]['kind']}]" for e in R["entities"]] or "—")
        print("P(món | truy vấn):", ", ".join(f"{self.name(d)} {p:.2f}" for d, p in R["query_dishes"][:3]))
        for it in R["results"]:
            ok = "✓" if it["dish"] == it["d_star"] else "✗"
            print(f"\n#{it['rank']} {it['image_id']}  [{it['dish_name']}]  điểm {it['score']:+.2f} "
                  f"= CLIP {it['part_clip']:+.2f} + đồ thị {it['part_graph']:+.2f}")
            print(f"   qua món «{self.name(it['d_star'])}» {ok} ({it['share']:.0%} điểm đồ thị)")
            for p in it["paths"]:
                print("   ", self.path_str(p))
            for e in it["unmatched"]:
                print(f"    (ràng buộc «{self.name(e)}» KHÔNG thỏa với món này)")
            if it["evidence"]:
                print("    ảnh bằng chứng —mô_tả→", self.name(it["d_star"]), ":",
                      ", ".join(self.IMG_IDS[j] for j in it["evidence"]))

    def explanation_graph(self, item, query=None):
        """Đồ thị con giải thích (networkx.DiGraph): truy vấn, thực thể khớp, món d*, ảnh kết quả, ảnh bằng chứng."""
        import networkx as nx
        G = nx.DiGraph()
        q = "TRUY_VẤN"
        import textwrap
        G.add_node(q, label=textwrap.fill(f"«{query}»", 16) if query else "truy vấn", kind="query")
        res_node = "IMG:" + item["image_id"]
        G.add_node(res_node, label=f"ảnh #{item['rank']}\n({item['dish_name']})", kind="result")
        d = item["d_star"]
        G.add_node(d, label=self.name(d), kind="món")
        G.add_edge(res_node, d, label=f"p(món|ảnh)", kind="pred")
        for p in item["paths"]:
            for h, r, t in p:
                if h == "TRUY_VẤN":
                    G.add_edge(q, t, label="nhắc tên", kind="mention")
                    continue
                for n in (h, t):
                    if n not in G:
                        G.add_node(n, label=self.name(n), kind=self.ENT[n]["kind"])
                G.add_edge(h, t, label=r, kind="kg")
            last = p[-1][2]
            if p[0][0] != "TRUY_VẤN":
                G.add_edge(q, last, label="nhắc", kind="mention")
        for e in item["unmatched"]:
            G.add_node(e, label=self.name(e), kind="unmatched")
            G.add_edge(q, e, label="nhắc (không thỏa)", kind="mention")
        for j in item["evidence"]:
            n = "EV:" + self.IMG_IDS[j]
            G.add_node(n, label=self.IMG_IDS[j].replace("img_", ""), kind="evidence")
            G.add_edge(n, d, label="mô_tả", kind="kg")
        return G
