# -*- coding: utf-8 -*-
"""Dựng 2 kernel Kaggle xuất bundle demo + checkpoint (chạy lại được, không sửa notebook gốc):

  notebooks/kaggle_export/ag/    ← notebook phần A, chỉ Action Genome: cấu hình 'none', seed 0, CLIP ViT-L/14
                                   (cache CLIP lấy từ output kernel hieutrinh123/du-lieu-do-thi), bỏ các cell chỉ
                                   chạy thí nghiệm Flickr8k. Output: /kaggle/working/output/action_genome/
                                   {checkpoints, demo_bundle/action_genome, results}.
  notebooks/kaggle_export/msvd/  ← notebook phần C nguyên vẹn (5 seed như bản chính thức); nạp checkpoint + cache đặc trưng + thumbnail
                                   từ output kernel lamvo30ai/project03-level4b-v2-graphfix (không huấn luyện lại).
                                   Output: /kaggle/working/project03_level4b*/{checkpoints, demo_bundle/msvd, results}.

Chạy:  python notebooks/tools/build_export_kernels.py   rồi   kaggle kernels push -p notebooks/kaggle_export/<ag|msvd>
"""
import json, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "hieutrinh123"


def code(src):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src)


def md(src):
    return dict(cell_type="markdown", metadata={}, source=src)


def build_ag():
    nb = json.load(open(ROOT / "notebooks/components/partA_du-lieu-do-thi.ipynb"))
    src = lambda c: "".join(c["source"])
    cells = nb["cells"]
    i_fl = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "records, hf_parts = load_flickr8k()" in src(c))
    i_ag = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and
                'AG_CACHE_DIR = Path("/kaggle/working/output/action_genome/cache")' in src(c))
    i_main = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "def load_action_genome" in src(c))
    out, n_skip = [], 0
    for i, c in enumerate(cells):
        s = src(c)
        if c["cell_type"] == "code":
            if i_fl <= i < i_ag:                         # khối chạy thí nghiệm Flickr8k
                if s.startswith("# Chọn CFG.learn_clip_w trên VALIDATION"):   # giữ định nghĩa, bỏ lời gọi trên Flickr
                    c = dict(c, source=s.replace("\nclipw_flickr = pick_learn_clip_w(records, split, img_feat, txt_feat)", ""))
                else:
                    n_skip += 1
                    continue
            elif i > i_main and not s.lstrip().startswith("%%script true"):   # kiểm định cặp AG sau cell chính
                n_skip += 1
                continue
        out.append(c)
        if i == i_ag:
            out.append(code(
                "# ---- KERNEL XUẤT BUNDLE DEMO (notebooks/tools/build_export_kernels.py) ----\n"
                "# Backbone ViT-L/14 (cùng CLIP với phần C ⇒ Space chỉ nạp một mô hình); cache clip_ag_ViT-L-14 lấy từ output\n"
                "# kernel hieutrinh123/du-lieu-do-thi (đã Add Input). Chỉ cấu hình 'none', seed 0: đủ cho demo + checkpoint.\n"
                'CFG.clip_model, CFG.clip_weights, CFG.clip_bs = "ViT-L-14", "laion2b_s32b_b82k", 128\n'
                'CFG.ablations = ("none",)\n'
                "CFG.seeds = [0]\n"
                'dataset_name = "action_genome"       # bình thường đặt ở cell Flickr8k (đã bỏ); encode_free_text cần\n'
                'print("Kernel xuất bundle AG:", CFG.clip_model, CFG.clip_weights, CFG.ablations, CFG.seeds)\n'))
    out.insert(0, md("# Project 3 — kernel xuất bundle demo + checkpoint: Action Genome\n\n"
                     "Sinh tự động từ `notebooks/components/partA_du-lieu-do-thi.ipynb` bởi "
                     "`notebooks/tools/build_export_kernels.py`; bỏ các cell chỉ chạy thí nghiệm Flickr8k. "
                     "Kết quả: `/kaggle/working/output/action_genome/{checkpoints,demo_bundle,results}`."))
    nb["cells"] = out
    d = ROOT / "notebooks/kaggle_export/ag"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(d / "p3_export_ag.ipynb", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(id=f"{OWNER}/p3-export-ag", title="p3-export-ag", code_file="p3_export_ag.ipynb", language="python",
                   kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
                   machine_shape="NvidiaTeslaT4", dataset_sources=[f"{OWNER}/action-genome"], competition_sources=[],
                   kernel_sources=[f"{OWNER}/du-lieu-do-thi"], model_sources=[]),
              open(d / "kernel-metadata.json", "w"), indent=2)
    print(f"AG: {len(out)} cell (bỏ {n_skip} cell Flickr8k / kiểm định cặp) → {d}")


def build_msvd():
    nb = json.load(open(ROOT / "notebooks/components/partC_project03-level4b-v2-graphfix.ipynb"))
    pre = code(
        "# ---- KERNEL XUẤT BUNDLE DEMO (notebooks/tools/build_export_kernels.py) ----\n"
        "# Giữ nguyên cấu hình chính thức (5 seed, các stage) để quyết định thiết kế trùng lần chạy gốc; nạp checkpoint\n"
        "# (find_ckpt_c) + cache đặc trưng + thumbnail từ output kernel\n"
        "# lamvo30ai/project03-level4b-v2-graphfix (đã Add Input) ⇒ không huấn luyện lại, không mã hoá lại video.\n"
        "import os, glob, shutil\n"
        "from pathlib import Path\n"
        "_caches = sorted({str(Path(p).parent) for p in glob.glob('/kaggle/input/**/cache/frames_msvd_*.npz', recursive=True)})\n"
        "os.environ.setdefault('MSVD_L4_CACHE_DIRS', ':'.join(_caches))\n"
        "print('cache đặc trưng từ input:', _caches)\n")
    post_thumbs = code(
        "# ---- KERNEL XUẤT BUNDLE DEMO: chép thumbnail của lần chạy trước (cần cho demo + bundle) ----\n"
        "_n = 0\n"
        "for _p in glob.glob('/kaggle/input/**/thumbs/*.jpg', recursive=True):\n"
        "    _dst = CFG.out_dir / 'thumbs' / Path(_p).name\n"
        "    if not _dst.exists():\n"
        "        shutil.copy2(_p, _dst); _n += 1\n"
        "print('thumbnail chép từ input:', _n)\n")
    cells = nb["cells"]
    src = lambda c: "".join(c["source"])
    i_cfg = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "save_checkpoints = _env(" in src(c))
    cells.insert(i_cfg + 1, post_thumbs)
    cells.insert(0, pre)
    cells.insert(0, md("# Project 3 — kernel xuất bundle demo + checkpoint: MSVD (phần C)\n\n"
                       "Sinh tự động từ `notebooks/components/partC_project03-level4b-v2-graphfix.ipynb` bởi "
                       "`notebooks/tools/build_export_kernels.py`. Kết quả: `.../demo_bundle/msvd`, `.../checkpoints`."))
    d = ROOT / "notebooks/kaggle_export/msvd"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(d / "p3_export_msvd.ipynb", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(id=f"{OWNER}/p3-export-msvd", title="p3-export-msvd", code_file="p3_export_msvd.ipynb",
                   language="python", kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False,
                   enable_internet=True, machine_shape="NvidiaTeslaT4", dataset_sources=[], competition_sources=[],
                   kernel_sources=["lamvo30ai/project03-level4b-v2-graphfix"], model_sources=[]),
              open(d / "kernel-metadata.json", "w"), indent=2)
    print(f"MSVD: {len(cells)} cell → {d}")


if __name__ == "__main__":
    build_ag()
    build_msvd()
