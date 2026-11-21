# -*- coding: utf-8 -*-
"""Sinh script kernel Kaggle cho MMKG (đã dùng cho hieutrinh123/project3-mmkg-vn, version 4).

Repo GitHub là private nên kernel không clone được: script nhúng kg_data.py, images.csv và notebook
(gzip + base64); 316 ảnh lấy từ dataset Kaggle hieutrinh123/project3-mmkg-vn-images (bản sao
mmkg/images/, kiểm tra sha256). Kernel chạy notebook bằng papermill và lưu bản đã chạy (giữ output).

    python mmkg/kaggle/build_kernel.py > kernel_script.py
    # rồi push với GPU T4, Internet bật, nguồn dữ liệu hieutrinh123/project3-mmkg-vn-images
"""
import base64, gzip, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
MMKG = os.path.dirname(HERE)

TEMPLATE = r'''# Wrapper Kaggle cho MMKG Ẩm thực Việt Nam (Project 3, cấp 3).
import base64, csv, glob, gzip, hashlib, json, os, shutil, subprocess, zipfile

EMBED = "__EMBED__"

def sh(c):
    print("$", c, flush=True)
    subprocess.run(c, shell=True, check=True)

W = "/kaggle/working"; D = "/kaggle/tmp/mmkg"; OUT = W + "/mmkg_out"
os.makedirs(OUT, exist_ok=True); os.makedirs(D, exist_ok=True)
for rel, text in json.loads(gzip.decompress(base64.b64decode(EMBED))).items():
    open(os.path.join(D, rel), "w", encoding="utf-8").write(text)
hit = glob.glob("/kaggle/input/**/pho_bo_01.jpg", recursive=True)
if hit:
    shutil.copytree(hit[0].split("/images/")[0] + "/images", D + "/images", dirs_exist_ok=True)
else:
    zipfile.ZipFile(glob.glob("/kaggle/input/**/mmkg_images.zip", recursive=True)[0]).extractall(D)
rows = list(csv.DictReader(open(os.path.join(D, "images.csv"), encoding="utf-8")))
ok = sum(os.path.exists(os.path.join(D, r["file"])) and
         hashlib.sha256(open(os.path.join(D, r["file"]), "rb").read()).hexdigest() == r["sha256"] for r in rows)
print(f"Ảnh khớp sha256 với repo: {ok}/{len(rows)}", flush=True)
assert ok == len(rows)
sh("pip install -q open_clip_torch==3.2.0 papermill")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
os.environ["MMKG_DIR"] = D; os.environ["MMKG_OUT"] = OUT
try:
    sh(f"cd {D} && papermill Project3_MMKG_VN.ipynb {OUT}/Project3_MMKG_VN_executed.ipynb -k python3 --log-output")
finally:
    sh(f"ls -la {OUT}")
'''

if __name__ == "__main__":
    files = {f: open(os.path.join(MMKG, f), encoding="utf-8").read() for f in ("kg_data.py", "images.csv")}
    nb = json.load(open(os.path.join(MMKG, "Project3_MMKG_VN.ipynb"), encoding="utf-8"))
    for c in nb["cells"]:                      # bỏ output cũ để script < 1 MB
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    files["Project3_MMKG_VN.ipynb"] = json.dumps(nb, ensure_ascii=False)
    print(TEMPLATE.replace("__EMBED__", base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()))
