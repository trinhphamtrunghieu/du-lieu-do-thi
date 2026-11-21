# -*- coding: utf-8 -*-
"""SINH TỰ ĐỘNG bởi notebooks/tools/gen_partC_lib.py — KHÔNG sửa tay.
Mã trích nguyên văn từ notebooks/components/partC_project03-level4b-v2-graphfix.ipynb (xử lý truy vấn: POS, lemma,
parse; mô hình QC-RGAT). Dữ liệu đồ thị (INV, PRED_VOCAB, PRED_EMB, VID_T, …) do demo/partC_engine.py gán từ bundle."""
import re, math, json, os, time, random
from collections import defaultdict, Counter
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger

for _pkg in ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
    try:
        nltk.download(_pkg, quiet=True, raise_on_error=False)
    except Exception:
        pass

DEVICE = "cpu"
INV = PRED_VOCAB = PRED_EMB = VID_T = None
PRED_FAMILIES, V4_REL_TYPES, Q_ROLES = (), [], ()
class _Cfg:
    top_m_paths = 5
CFG = _Cfg()

# --- notebook cell 20 ---
_LEM, _LEM_CACHE = WordNetLemmatizer(), {}

# --- notebook cell 20 ---
def _verb_lemma(w):
    """WordNetLemmatizer trả về ứng viên NGẮN NHẤT ('riding' → 'rid', 'shining' → 'shin'); ở đây chọn ứng viên
    có tần suất WordNet cao nhất ('riding' → 'ride')."""
    wn = _wordnet()
    cands = wn._morphy(w, "v") if wn is not None and hasattr(wn, "_morphy") else []
    if len(cands) > 1:
        cnt = lambda c: sum(l.count() for s in wn.synsets(c, pos="v") for l in s.lemmas() if l.name().lower() == c)
        return max(sorted(cands), key=lambda c: (cnt(c), len(c)))      # sorted: không phụ thuộc PYTHONHASHSEED
    return sorted(cands)[0] if cands else _LEM.lemmatize(w, "v")

# --- notebook cell 20 ---
def lemma(w, pos="n"):
    k = (w, pos)
    r = _LEM_CACHE.get(k)
    if r is None:
        try:
            if pos == "v":
                r = _verb_lemma(w)
            else:                                   # như NLTK (ứng viên ngắn nhất) nhưng hoà thì theo thứ tự chữ
                wn_ = _wordnet()
                cands = wn_._morphy(w, pos) if wn_ is not None and hasattr(wn_, "_morphy") else []
                r = min(sorted(cands), key=len) if cands else _LEM.lemmatize(w, pos)
        except LookupError:
            r = w[:-1] if (pos == "n" and w.endswith("s") and len(w) > 3) else w
        _LEM_CACHE[k] = r
    return r

# --- notebook cell 20 ---
try:
    from nltk.tag import PerceptronTagger
    _TAGGER = PerceptronTagger()                    # tạo MỘT lần (pos_tag() tạo lại mỗi lời gọi → rất chậm)
    _TAGGER.tag(["a", "man", "is", "playing"])
except Exception as e:
    _TAGGER = None
    print("⚠ Không nạp được POS tagger NLTK → phân loại từ theo WordNet (kém chính xác hơn):", type(e).__name__)

# --- notebook cell 20 ---
VISUAL_LEX = {"noun.person", "noun.animal", "noun.artifact", "noun.object", "noun.food",
              "noun.plant", "noun.substance", "noun.body", "noun.location", "noun.group",
              "noun.phenomenon"}

# --- notebook cell 20 ---
SPORT_ROOTS = {"sport.n.01", "game.n.01"}

# --- notebook cell 20 ---
_WN = None

# --- notebook cell 20 ---
def _wordnet():
    """Nạp WordNet một lần; không được thì trả None (pipeline rơi về FALLBACK_ISA)."""
    global _WN
    if _WN is None:
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("dog"); _WN = wn
        except Exception:
            _WN = False
    return _WN or None

# --- notebook cell 20 ---
def _main_noun_synset(w):
    wn = _wordnet()
    if wn is None:
        return None
    syns = wn.synsets(w.replace(" ", "_"), pos="n")
    return syns[0] if syns else None

# --- notebook cell 20 ---
_NOUN_CACHE, _VERB_CACHE = {}, {}

# --- notebook cell 20 ---
def is_visual_noun(w):
    if w in _NOUN_CACHE:
        return _NOUN_CACHE[w]
    wn = _wordnet()
    if wn is None:
        _NOUN_CACHE[w] = True
        return True
    s, ok = _main_noun_synset(w), False
    if s is not None:
        lex = s.lexname()
        if lex in VISUAL_LEX:
            ok = True
        elif lex == "noun.act":
            anc = {h.name() for p in s.hypernym_paths() for h in p[:-1]}
            ok = bool(anc & SPORT_ROOTS)
        if ok:
            key = w.replace(" ", "_")
            cnt = lambda pos: sum(l.count() for x in wn.synsets(key, pos=pos)
                                  for l in x.lemmas() if l.name().lower() == key)
            ok = cnt("n") >= cnt("v")
    _NOUN_CACHE[w] = ok
    return ok

# --- notebook cell 20 ---
def is_action_verb(v):
    if v not in _VERB_CACHE:
        wn = _wordnet()
        _VERB_CACHE[v] = True if wn is None else bool(wn.synsets(v, pos="v"))
    return _VERB_CACHE[v]

# --- notebook cell 21 ---
STOP = set("a an the of on in at to with and or is are was were this that these those "
           "there here it its his her their some any for from by as be being been very "
           "photo image picture showing shows scene view near next video clip".split())

# --- notebook cell 21 ---
NOT_OBJECTS = set("front background foreground top bottom side end corner air group area region place space "
                  "picture photo image one other something someone somebody anything thing stuff view middle edge "
                  "part way lot pair couple bunch number while each both several many few kind type video clip "
                  "footage scene time minute second".split())

# --- notebook cell 21 ---
_SKIP_P = set("is are was were be been being has have had while as that who which and or".split())

# --- notebook cell 21 ---
AUX_VERBS = set("be have do can will would shall should may might must seem appear become begin start try "
                "keep continue let get go come".split())

# --- notebook cell 21 ---
def _fallback_pos(toks):
    wn, out = _wordnet(), []
    for t in toks:
        if wn is None:
            out.append((t, "NN" if len(t) > 2 and t not in STOP else "DT"))
        elif t.endswith("ing") and wn.synsets(lemma(t, "v"), pos="v"):
            out.append((t, "VBG"))
        elif wn.synsets(t, pos="n"):
            out.append((t, "NN"))
        elif wn.synsets(lemma(t, "v"), pos="v"):
            out.append((t, "VB"))
        else:
            out.append((t, "DT"))
    return out

# --- notebook cell 22 ---
_PARSE2_CACHE = {}

# --- notebook cell 22 ---
def ckey(lemma_, pos):
    """Khoá node lớp: 'dog#n', 'play#v'."""
    return f"{lemma_}#{pos}"

# --- notebook cell 22 ---
def parse_caption_v2(cap):
    """Như parse_caption (v1) nhưng (1) khoá theo (lemma, POS), (2) mỗi quan hệ giữ LOẠI và VỊ TỪ.
    Trả về (danh từ, {động từ: dạng V-ing}, {(nguồn, loại, đích, vị từ)})."""
    if cap in _PARSE2_CACHE:
        return _PARSE2_CACHE[cap]
    toks = re.findall(r"[a-z]+", str(cap).lower())
    tags = (_TAGGER.tag(toks) if _TAGGER is not None else _fallback_pos(toks)) if toks else []
    nouns, verbs, triples = set(), {}, set()
    subj, last_n, preds, pred_has_verb, cur_v = None, None, [], False, None
    for w, t in tags:
        if t in ("NN", "NNS") and len(w) > 2 and w not in NOT_OBJECTS and w not in STOP:
            n = lemma(w, "n")
            if n in NOT_OBJECTS or not is_visual_noun(n):
                continue
            nouns.add(n)
            if last_n is not None and 1 <= len(preds) <= 2 and last_n != n:
                triples.add((ckey(last_n, "n"), "svo" if pred_has_verb else "prep", ckey(n, "n"), " ".join(preds)))
            if cur_v is not None and n != subj:
                triples.add((ckey(n, "n"), "object_of", ckey(cur_v, "v"), cur_v))
                cur_v = None
            subj = subj or n
            last_n, preds, pred_has_verb = n, [], False
        elif t.startswith("VB"):
            v = lemma(w, "v")
            if v in AUX_VERBS or len(v) < 3 or not is_action_verb(v):
                continue
            if t == "VBG":
                verbs[v] = w
            else:
                verbs.setdefault(v, None)
            if subj is not None:
                triples.add((ckey(subj, "n"), "agent_of", ckey(v, "v"), v))
            if last_n is not None:
                preds.append(v); pred_has_verb = True
            cur_v = v
        elif t in ("IN", "TO", "RP") and w not in _SKIP_P and last_n is not None:
            preds.append(w)
    res = (frozenset(nouns), dict(verbs), frozenset(triples))
    _PARSE2_CACHE[cap] = res
    return res

# --- notebook cell 30 ---
def predicate_tokens(p):
    return tuple(p.split())

# --- notebook cell 32 ---
def query_tokens_v2(text):
    """[(POS dự đoán, lemma danh từ, lemma động từ)] — POS tagger trên CHÍNH câu truy vấn (input của hệ thống)."""
    toks = re.findall(r"[a-z]+", str(text).lower())
    tags = (_TAGGER.tag(toks) if _TAGGER is not None else _fallback_pos(toks)) if toks else []
    out = []
    for w, t in tags:
        if w in STOP or len(w) < 2:
            continue
        p = "v" if t.startswith("VB") else ("n" if t.startswith("NN") else "x")
        out.append((p, lemma(w, "n"), lemma(w, "v")))
    return out

# --- notebook cell 32 ---
def parse_tokens_v2(toks, inv):
    """Khớp token với node: lớp đúng POS trước (danh từ → 'x#n', động từ → 'x#v'), không có thì POS còn lại,
    không có lớp thì khái niệm taxonomy cùng lemma + POS. Trả về [(node_id, tên node, token)]."""
    hits, seen = [], set()
    for p, ln, lv in toks:
        order = [("v", lv), ("n", ln)] if p == "v" else [("n", ln), ("v", lv)]
        found = None
        for pos, lm in order:
            i = inv.key2id.get(ckey(lm, pos))
            if i is not None:
                found = [i]; break
        if found is None:
            for pos, lm in order:
                if inv.concepts_by_lemma.get((lm, pos)):
                    found = inv.concepts_by_lemma[(lm, pos)]; break
        for i in found or []:
            if i not in seen:
                seen.add(i); hits.append((i, inv.name[i], lv if p == "v" else ln))
    return hits

# --- notebook cell 32 ---
def query_pred_tokens(text):
    """Tập token của truy vấn để khớp VỊ TỪ: từ gốc + lemma động từ (giới từ giữ nguyên; KHÔNG bỏ stop-word)."""
    w = re.findall(r"[a-z]+", str(text).lower())
    return frozenset(w) | frozenset(lemma(x, "v") for x in w)

# --- notebook cell 32 ---
_PRED_BY_FIRST = {}

# --- notebook cell 32 ---
def match_predicates(qset, rel):
    """pred_id của họ rel có MỌI token nằm trong truy vấn ('ride on' khớp 'a man is riding on a horse')."""
    if rel not in _PRED_BY_FIRST:
        idx = defaultdict(list)
        for i, (r, p) in enumerate(PRED_VOCAB):
            if r == rel:
                t = predicate_tokens(p)
                idx[t[0]].append((i, t))
        _PRED_BY_FIRST[rel] = idx
    idx = _PRED_BY_FIRST[rel]
    return sorted({i for tok in qset for i, t in idx.get(tok, ()) if all(x in qset for x in t)})

# --- notebook cell 34 ---
def query_roles_v4(texts, inv):
    """Vai của node lớp trong CHÍNH truy vấn (parser caption áp lên văn bản truy vấn lúc suy luận — mục đích query_input)."""
    out = []
    for text in texts:
        _, _, trip = parse_caption_v2(text)
        role = {}
        for s_, rel, o_, _p in trip:
            a, b = inv.key2id.get(s_), inv.key2id.get(o_)
            if a is None or b is None:
                continue
            if rel == "agent_of":
                role.setdefault(a, "subj"); role.setdefault(b, "verb")
            elif rel == "object_of":
                role.setdefault(a, "obj"); role.setdefault(b, "verb")
            elif rel == "svo":
                role.setdefault(a, "subj")
                role[b] = role[b] if role.get(b) not in (None, "other") else "obj"
            elif rel == "prep":
                role.setdefault(b, "pobj")
        out.append({k: Q_ROLES.index(r) for k, r in role.items()})
    return out

# --- notebook cell 34 ---
class QCRGAT(nn.Module):
    """rel=False ⇒ G1 (chỉ contains); use_pred=False ⇒ G2 (không vị từ); qcond=False ⇒ G6 (attention tĩnh, không theo truy vấn)."""

    SCALARS = ("beta", "kappa", "rgate", "g_ex", "g_soft")      # cổng vô hướng: lr × CFG.scalar_lr_mult (như cổng của v3)

    def __init__(self, data, clip_dim, d=64, T=1, rel=True, qcond=True, use_pred=True):
        super().__init__()
        assert T in (1, 2), "phân rã đường đi hiện cài cho T = 1 hoặc 2"
        self.D, self.T, self.rel, self.qcond, self.use_pred = data, T, rel, qcond, use_pred
        nt = len(V4_REL_TYPES)
        self.vid = nn.Linear(clip_dim, d)
        self.q = nn.Sequential(nn.Linear(clip_dim, d), nn.GELU(), nn.Linear(d, d))
        self.qo = nn.Linear(d, d)
        self.cls = nn.Embedding(data.ns, d)
        nn.init.normal_(self.cls.weight, std=0.02)
        self.z = nn.Parameter(1 + 0.1 * torch.randn(T, nt, d))
        self.beta = nn.Parameter(torch.full((nt,), -1.0))
        self.kappa = nn.Parameter(torch.zeros(nt))
        self.qgate = nn.Linear(d, nt)
        self.rgate = nn.Parameter(torch.zeros(len(Q_ROLES), nt))
        self.g_ex = nn.Parameter(torch.ones(nt))         # prior: cạnh có vị từ KHỚP truy vấn được ưu tiên (học tiếp từ dữ liệu)
        self.g_soft = nn.Parameter(torch.zeros(nt))
        self.clip_logit = nn.Parameter(torch.tensor(0.5413))

    def attention(self, qb, u, D=None):
        """a_q(cạnh) ∈ (0, 1), [B, E]. Chỉ phụ thuộc truy vấn + thuộc tính tĩnh của cạnh (KHÔNG phụ thuộc trạng thái node)."""
        D = D or self.D
        a = (self.beta[D.et] + self.kappa[D.et] * D.logw)[None].expand(u.shape[0], -1)
        if self.qcond:
            a = a + self.qgate(u)[:, D.et] + self.rgate[qb["role_src"][:, D.src], D.et]
            if self.use_pred:
                a = a + self.g_ex[D.et][None] * qb["ex"] + self.g_soft[D.et][None] * qb["soft"]
        return torch.sigmoid(a)

    def class_states(self, qb, u, D=None):
        D = D or self.D
        E = self.cls.weight[None].expand(u.shape[0], -1, -1)
        if not self.rel or D.E == 0:
            return E, None
        a = self.attention(qb, u, D)
        H = E
        for t in range(self.T):
            H = E.index_add(1, D.dst, a[:, :, None] * self.z[t][D.et][None] * H[:, D.src, :])
        return H, a

    def score(self, qb, cand, D=None):
        """cand: [B, K] video id (tensor) ⇒ [B, K]."""
        D = D or self.D
        u = self.q(qb["tq"])
        H, _ = self.class_states(qb, u, D)
        V = self.vid(VID_T[cand]) + torch.einsum("bkc,bcd->bkd", D.Acon[cand], H)
        return torch.einsum("bd,bkd->bk", F.normalize(self.qo(u), dim=-1), V)

    @torch.no_grad()
    def decompose(self, q, vid, D=None):
        """Phân rã CHÍNH XÁC s(q, vid) = thị giác + Σ nhãn + Σ đường đi quan hệ. Mỗi đường đi: list chỉ số cạnh (trong D)
        theo chiều đi từ node nguồn tới node nhãn c của video, kèm đóng góp."""
        D = D or self.D
        qb = D.query_batch([q], self.use_pred)
        u = self.q(qb["tq"])
        uo = F.normalize(self.qo(u), dim=-1)[0]
        E = self.cls.weight
        vt = int(vid)
        terms = [dict(kind="visual", node=None, edges=(), contrib=float(uo @ self.vid(VID_T[vt])))]
        acon = D.Acon[vt]
        tags = torch.nonzero(acon > 0).flatten().tolist()
        for c in tags:
            terms.append(dict(kind="contains", node=c, edges=(), contrib=float(acon[c] * (uo @ E[c]))))
        if self.rel and D.E:
            a = self.attention(qb, u, D)[0]
            inc = defaultdict(list)
            for k in range(D.E):
                inc[int(D.dst_np[k])].append(k)
            for c in tags:
                for e1 in inc.get(c, []):                     # đường đi 1 bước: u –e1→ c   (dùng z của lớp cuối)
                    zz = self.z[self.T - 1][D.et[e1]]
                    terms.append(dict(kind="relation", node=c, edges=(e1,),
                                      contrib=float(acon[c] * a[e1] * (uo @ (zz * E[D.src_np[e1]])))))
                    if self.T == 2:                           # đường đi 2 bước: w –e0→ u –e1→ c
                        for e0 in inc.get(int(D.src_np[e1]), []):
                            zz2 = self.z[1][D.et[e1]] * self.z[0][D.et[e0]]
                            terms.append(dict(kind="relation", node=c, edges=(e0, e1),
                                              contrib=float(acon[c] * a[e1] * a[e0] * (uo @ (zz2 * E[D.src_np[e0]])))))
        return terms

# --- notebook cell 36 ---
def znorm_per_row(x, eps=1e-6):
    if isinstance(x, torch.Tensor):
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + eps)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + eps)

# --- notebook cell 46 ---
@torch.no_grad()
def explain_full(model, q, vid, top_m=None):
    """Giải thích ĐẦY ĐỦ "vì sao video được xếp hạng như vậy" = top-m số hạng ĐỒ THỊ ủng hộ (contains + đường đi quan hệ).
    Số hạng thị giác (W_v x_v) không phải đồ thị nên được báo riêng. Σ mọi số hạng = điểm đồ thị của mô hình."""
    top_m = top_m or CFG.top_m_paths
    terms = model.decompose(q, vid)
    graph_terms = [t for t in terms if t["kind"] != "visual"]
    sup = sorted([t for t in graph_terms if t["contrib"] > 0], key=lambda t: -t["contrib"])[:top_m]
    opp = sorted([t for t in graph_terms if t["contrib"] < 0], key=lambda t: t["contrib"])[:top_m]
    tot = {k: sum(t["contrib"] for t in terms if t["kind"] == k) for k in ("visual", "contains", "relation")}
    return dict(terms=terms, support=sup, oppose=opp, totals=tot, score=sum(t["contrib"] for t in terms),
                n_graph_terms=len(graph_terms), pos_graph=sum(t["contrib"] for t in graph_terms if t["contrib"] > 0))
