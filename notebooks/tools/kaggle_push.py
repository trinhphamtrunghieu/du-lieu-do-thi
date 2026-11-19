# -*- coding: utf-8 -*-
"""Đẩy / theo dõi / tải output kernel Kaggle qua REST API (thay cho CLI `kaggle`, cùng định dạng request).

    python notebooks/tools/kaggle_push.py push   notebooks/kaggle_export/ag
    python notebooks/tools/kaggle_push.py status hieutrinh123/p3-export-ag
    python notebooks/tools/kaggle_push.py list   hieutrinh123/p3-export-ag [lọc]
    python notebooks/tools/kaggle_push.py get    hieutrinh123/p3-export-ag <tiền tố đường dẫn> <thư mục đích>

Xác thực: biến môi trường KAGGLE_USERNAME / KAGGLE_KEY (Basic auth), hoặc proxy chèn sẵn credential.
"""
import json, os, sys, urllib.parse
from pathlib import Path

import requests

API = "https://www.kaggle.com/api/v1"
CA = os.environ.get("REQUESTS_CA_BUNDLE") or ("/root/.ccr/ca-bundle.crt" if Path("/root/.ccr/ca-bundle.crt").exists() else True)
AUTH = (os.environ.get("KAGGLE_USERNAME", "proxy-injected"), os.environ.get("KAGGLE_KEY", "proxy-injected"))


def _req(method, path, **kw):
    r = requests.request(method, API + path, auth=AUTH, verify=CA, timeout=300, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path} → HTTP {r.status_code}: {r.text[:500]}")
    return r


def push(folder):
    folder = Path(folder)
    meta = json.load(open(folder / "kernel-metadata.json"))
    nb = json.load(open(folder / meta["code_file"]))
    for c in nb["cells"]:                              # như CLI: bỏ output, source là một chuỗi
        if c["cell_type"] == "code":
            c["outputs"] = []
        if isinstance(c.get("source"), list):
            c["source"] = "".join(c["source"])
    body = dict(slug=meta["id"], newTitle=meta["title"], text=json.dumps(nb), language=meta.get("language", "python"),
                kernelType=meta.get("kernel_type", "notebook"), isPrivate=meta.get("is_private", True),
                enableGpu=meta.get("enable_gpu", False), enableTpu=meta.get("enable_tpu", False),
                enableInternet=meta.get("enable_internet", True), datasetDataSources=meta.get("dataset_sources", []),
                competitionDataSources=meta.get("competition_sources", []), kernelDataSources=meta.get("kernel_sources", []),
                modelDataSources=meta.get("model_sources", []), categoryIds=meta.get("keywords", []),
                machineShape=meta.get("machine_shape"))
    r = _req("POST", "/kernels/push", json=body).json()
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if r.get("error"):
        raise SystemExit(1)


def status(kernel):
    u, s = kernel.split("/")
    print(json.dumps(_req("GET", f"/kernels/status?userName={u}&kernelSlug={s}").json(), ensure_ascii=False))


def _files(kernel):
    u, s = kernel.split("/")
    tok, files, log = "", [], None
    while True:
        q = f"/kernels/output?userName={u}&kernelSlug={s}&pageSize=200" + (f"&pageToken={urllib.parse.quote(tok)}" if tok else "")
        d = _req("GET", q).json()
        files += d.get("files", [])
        log = log or d.get("log")
        tok = d.get("nextPageToken") or ""
        if not tok:
            return files, log


def list_files(kernel, filt=""):
    files, _ = _files(kernel)
    for f in files:
        if filt in f["fileName"]:
            print(f["fileName"])
    print(len(files), "file")


def get(kernel, prefix, dest):
    files, log = _files(kernel)
    dest = Path(dest)
    n = 0
    for f in files:
        name = f["fileName"]
        if not name.startswith(prefix):
            continue
        out = dest / name[len(prefix):].lstrip("/")
        out.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f["url"], verify=CA, timeout=600)
        r.raise_for_status()
        out.write_bytes(r.content)
        n += 1
    if prefix == "__log__" and log:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "kernel_log.json").write_text(log)
    print(f"Tải {n} file → {dest}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    dict(push=push, status=status, list=list_files, get=get)[cmd](*args)
