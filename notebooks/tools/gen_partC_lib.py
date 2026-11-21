# -*- coding: utf-8 -*-
"""Sinh demo/partC_lib.py: trích NGUYÊN VĂN mã xử lý truy vấn + mô hình QC-RGAT từ notebook phần C.

Lấy các định nghĩa gốc (hàm/lớp/hằng ở mức module của các cell), rồi tự thêm mọi tên mà chúng dùng tới (phân tích AST,
lặp tới bao đóng). Tên là DỮ LIỆU của đồ thị (INV, PRED_VOCAB, VID_T…) không trích mà để demo/partC_engine.py gán từ
bundle xuất ra trên Kaggle. Chạy lại script này mỗi khi notebook phần C đổi.
"""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks/components/partC_project03-level4b-v2-graphfix.ipynb"
OUT = ROOT / "demo/partC_lib.py"

ROOTS = ["query_tokens_v2", "parse_tokens_v2", "query_pred_tokens", "match_predicates", "query_roles_v4",
         "parse_caption_v2", "ckey", "QCRGAT", "znorm_per_row", "explain_full"]
# dữ liệu gán từ bundle (không trích)
DATA = {"INV", "PRED_VOCAB", "PRED_EMB", "PRED_FAMILIES", "VID_T", "V4_REL_TYPES", "Q_ROLES", "DEVICE", "CFG",
        "TXT_FEAT", "V4_HITS", "V4_ROLES", "QPREDS", "RECORDS"}
SKIP = {"print", "len", "range", "set", "dict", "list", "tuple", "sorted", "min", "max", "sum", "any", "all", "zip",
        "enumerate", "float", "int", "str", "bool", "isinstance", "hasattr", "getattr", "super", "map", "filter",
        "frozenset", "abs", "round", "type", "object", "Exception", "LookupError", "ValueError", "KeyError",
        "AttributeError", "IndexError", "TypeError", "RuntimeError", "globals", "open", "iter", "next", "reversed",
        "self", "True", "False", "None", "staticmethod", "property", "classmethod", "repr", "hash", "id", "vars",
        "__name__", "slice", "divmod", "pow", "ord", "chr", "callable", "NotImplementedError", "StopIteration",
        "re", "np", "torch", "nn", "F", "math", "defaultdict", "Counter", "nltk", "sp", "json", "os", "time", "random"}


def main():
    nb = json.load(open(NB))
    defs = {}                                   # tên → (thứ tự cell, đoạn mã)
    for ci, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("%%"):
            continue
        try:
            t = ast.parse(s)
        except SyntaxError:
            continue
        lines = s.split("\n")
        for node in t.body:
            seg = "\n".join(lines[node.lineno - 1 - len(getattr(node, "decorator_list", [])):node.end_lineno])
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defs[node.name] = (ci, node.lineno, seg, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                tg = node.targets if isinstance(node, ast.Assign) else [node.target]
                for x in tg:
                    for n in ast.walk(x):
                        if isinstance(n, ast.Name) and n.id not in defs:   # lấy định nghĩa ĐẦU TIÊN của hằng
                            defs[n.id] = (ci, node.lineno, seg, node)
            elif isinstance(node, ast.Try) and any(isinstance(b, ast.Assign) and any(
                    isinstance(x, ast.Name) and x.id == "_TAGGER" for x in b.targets) for b in node.body):
                defs["_TAGGER"] = (ci, node.lineno, seg, node)

    def _locals(fn):
        loc = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
        glob = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Global):
                glob |= set(n.names)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                loc.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.Lambda)) and n is not fn:
                loc |= {a.arg for a in ast.walk(n.args) if isinstance(a, ast.arg)}
                if isinstance(n, ast.FunctionDef):
                    loc.add(n.name)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                loc.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                loc |= {(a.asname or a.name).split(".")[0] for a in n.names}
        return loc - glob

    def free_names(node):
        """Tên đọc mà không phải biến cục bộ / tham số của hàm (hay phương thức) chứa nó."""
        out = set()
        fns = [n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.Lambda))]
        if not fns:                                   # câu gán / try ở mức module
            return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        top = [f for f in fns if isinstance(f, ast.FunctionDef)] or fns
        for f in top:
            loc = _locals(f) if isinstance(f, ast.FunctionDef) else set()
            out |= {n.id for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - loc
        if isinstance(node, ast.ClassDef):            # thân lớp ngoài các phương thức (thuộc tính lớp, lớp cha)
            out |= {n.id for b in node.bases for n in ast.walk(b) if isinstance(n, ast.Name)}
            for b in node.body:
                if not isinstance(b, ast.FunctionDef):
                    out |= {n.id for n in ast.walk(b) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        return out

    need, todo = {}, list(ROOTS)
    while todo:
        k = todo.pop()
        if k in need or k in DATA or k in SKIP:
            continue
        if k not in defs:
            continue
        need[k] = defs[k]
        todo += [n for n in free_names(defs[k][3]) if n not in need]
    missing = sorted(set().union(*(free_names(v[3]) for v in need.values())) - set(need) - DATA - SKIP - set(defs))
    segs = sorted({(v[0], v[1], v[2]) for v in need.values()})
    head = '''# -*- coding: utf-8 -*-
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

'''
    body = "\n\n".join(f"# --- notebook cell {ci} ---\n{seg}" for ci, _, seg in segs)
    OUT.write_text(head + body + "\n")
    print(f"Ghi {OUT}: {len(need)} định nghĩa từ {len({s[0] for s in segs})} cell; tên tự do chưa rõ: {missing[:40]}")


if __name__ == "__main__":
    main()
