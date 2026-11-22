"""Tạo lại (và kiểm tra) split cố định của phần A: Flickr8k và Action Genome.

Hai file JSON trong thư mục này là bản gốc `split.json` lấy từ output Kaggle
`hieutrinh123/du-lieu-do-thi` (cùng sha1 với log notebook gộp `hieutrinh123/project3-merged`).
Script này sinh lại đúng các danh sách đó từ nguồn dữ liệu và so sha1, rồi ghi thêm CSV
ánh xạ `image_id` (chỉ số nội bộ của notebook) sang định danh gốc của dataset.

Cách dùng:
    python splits/make_splits.py                      # chỉ Flickr8k (không cần tải dữ liệu)
    python splits/make_splits.py --ag-records PATH    # thêm Action Genome; PATH = ag_records.json
                                                      # (dataset Kaggle hieutrinh123/action-genome)
    python splits/make_splits.py --check              # chỉ kiểm tra, không ghi file

Quy tắc (giống hệt `make_split` / `ag_make_split` trong notebooks/components/partA_du-lieu-do-thi.ipynb):
- Flickr8k: split chính thức của HF `jxie/flickr8k` (train 6000 / validation 1000 / test 1000).
  `image_id` đánh số liên tục theo thứ tự train → validation → test, mỗi split giữ thứ tự hàng của HF.
  Chia theo ẢNH: 5 caption của một ảnh luôn nằm cùng split.
- Action Genome: train/test chính thức của AG (theo Charades); val = 15% video của train gốc,
  chọn bằng random.Random(20240101).shuffle trên danh sách video_id đã sắp xếp, lấy phần đầu.
  `image_id` = vị trí bản ghi trong ag_records.json (đã sắp theo video_id). Chia theo VIDEO.
- sha1 = 12 ký tự đầu của sha1(json.dumps({"train","val","test"}, sort_keys=True)).
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLICKR_HF_SPLITS = (("train", "train", 6000), ("validation", "val", 1000), ("test", "test", 1000))
AG_VAL_SEED = 20240101
AG_VAL_RATIO = 0.15


def sha1_of(split):
    payload = json.dumps({k: split[k] for k in ("train", "val", "test")}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def check_disjoint(split, name):
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(split[a]) & set(split[b]), f"{name}: {a}/{b} chồng lấn"


def make_flickr8k():
    split, rows, iid = {"train": [], "val": [], "test": []}, [], 0
    for hf_name, tag, n in FLICKR_HF_SPLITS:
        for hf_row in range(n):
            split[tag].append(iid)
            rows.append(dict(image_id=iid, split=tag, hf_split=hf_name, hf_row=hf_row))
            iid += 1
    split["seed"] = "official-hf-split"
    split["sha1"] = sha1_of(split)
    return split, rows


def make_action_genome(ag_records_path):
    recs = json.load(open(ag_records_path))["records"]
    vids = [r["video_id"] for r in recs]
    assert vids == sorted(vids) and len(set(vids)) == len(vids), "ag_records.json phải sắp theo video_id, không trùng"
    # train gốc của AG = train ∪ val trong ag_records.json; tách val lại bằng seed cố định
    pool = sorted(r["video_id"] for r in recs if r["split"] in ("train", "val"))
    rng = random.Random(AG_VAL_SEED)
    rng.shuffle(pool)
    val_vids = set(pool[:int(len(pool) * AG_VAL_RATIO)])
    split, rows = {"train": [], "val": [], "test": []}, []
    for i, r in enumerate(recs):
        tag = "test" if r["split"] == "test" else ("val" if r["video_id"] in val_vids else "train")
        assert tag == r["split"], f"{r['video_id']}: tính ra {tag}, ag_records ghi {r['split']}"
        split[tag].append(i)
        rows.append(dict(image_id=i, split=tag, video_id=r["video_id"], ag_official_split="test" if tag == "test" else "train"))
    split["seed"] = "ag-official-train-test + val seed 20240101"
    split["sha1"] = sha1_of(split)
    return split, rows


def verify_and_write(name, split, rows, write):
    check_disjoint(split, name)
    ref_path = HERE / f"partA_{name}_split.json"
    ref = json.load(open(ref_path))
    same = all(ref[k] == split[k] for k in ("train", "val", "test")) and ref["sha1"] == split["sha1"]
    counts = " ".join(f"{k}={len(split[k])}" for k in ("train", "val", "test"))
    print(f"{name}: sha1={split['sha1']} {counts} | khớp {ref_path.name}: {same}")
    assert same, f"{name}: split sinh lại KHÁC file đã lưu"
    if write:
        out = HERE / f"partA_{name}_split.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  → {out.relative_to(HERE.parent)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-records", type=Path, help="đường dẫn ag_records.json")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi CSV")
    a = ap.parse_args()
    verify_and_write("flickr8k", *make_flickr8k(), write=not a.check)
    if a.ag_records:
        verify_and_write("action_genome", *make_action_genome(a.ag_records), write=not a.check)
    else:
        ref = json.load(open(HERE / "partA_action_genome_split.json"))
        check_disjoint(ref, "action_genome")
        assert sha1_of(ref) == ref["sha1"]
        print(f"action_genome: bỏ qua sinh lại (thiếu --ag-records); file đã lưu sha1={ref['sha1']} hợp lệ")


if __name__ == "__main__":
    main()
