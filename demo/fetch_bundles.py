# -*- coding: utf-8 -*-
"""Tải bundle demo Action Genome / MSVD từ output kernel Kaggle về demo/bundles/ (cần Kaggle API key).

    python demo/fetch_bundles.py            # cả hai
    python demo/fetch_bundles.py ag         # chỉ Action Genome
Kernel: hieutrinh123/p3-export-ag, hieutrinh123/p3-export-msvd (dựng bởi notebooks/tools/build_export_kernels.py).
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUSH = ROOT / "notebooks/tools/kaggle_push.py"
SRC = {"ag": ("hieutrinh123/p3-export-ag", "output/action_genome/demo_bundle/action_genome", "action_genome"),
       "msvd": ("hieutrinh123/p3-export-msvd", "project03_level4b/demo_bundle/msvd", "msvd")}

for key in (sys.argv[1:] or list(SRC)):
    kernel, prefix, name = SRC[key]
    subprocess.run([sys.executable, str(PUSH), "get", kernel, prefix, str(ROOT / "demo/bundles" / name)], check=True)
