# Project 3 — Truy vấn đa phương thức có giải thích bằng đồ thị

- Đề bài: `docs/Project03_Multimodal_XAI.pdf`.
- **Kết quả: `RESULTS.md`.**
- Đối chiếu với đề bài: `REQUIREMENTS_CHECK.md`.

## Các phần

| Phần | Nội dung | Notebook | Kết quả |
| --- | --- | --- | --- |
| A | Flickr8k + Action Genome: CLIP + GNN + điểm đường đi + nhánh khớp truy vấn↔node (mô hình cuối, cấp 1–2, 4a, 4b) | `notebooks/components/partA_du-lieu-do-thi.ipynb` | `results/partA_vitl14_*`, `results/partA_vith14_*` |
| B | Visual Genome, COCO, ARO, SugarCrepe, COCO-spatial, video MSVD v5 | `notebooks/components/partB_image_video/` | `results/partB_image_video/` |
| C | Level 4(b) v2-graphfix: video MSVD + MSR-VTT | `notebooks/components/partC_project03-level4b-v2-graphfix.ipynb` | `results/partC_level4b/` |
| MMKG | Cấp 3: MMKG Ẩm thực Việt Nam tự xây (148 thực thể, 750 triple, 316 ảnh) | `mmkg/Project3_MMKG_VN.ipynb` (đã chạy, giữ output) | `results/mmkg/` |

`notebooks/Project3_Merged.ipynb` gộp A + C + B để chạy một lần. MMKG là notebook riêng.

## Cấu trúc

```
README.md  RESULTS.md  REQUIREMENTS_CHECK.md  requirements.txt
docs/                      đề bài
notebooks/
  Project3_Merged.ipynb    notebook gộp A + C + B (+ kernel-metadata.json)
  kaggle/                  bản Kaggle < 1 MB của notebook gộp (payload trong dataset hieutrinh123/project3-merged-assets)
  components/              notebook gốc của từng phần
mmkg/                      dữ liệu MMKG: kg_data.py, images/, images.csv, DATA_CARD.md, notebook, mã thu thập ảnh
results/                   toàn bộ output thật từ Kaggle (xem results/README.md)
```

## Chạy lại

**Notebook gộp (A + C + B) trên Kaggle**
1. Upload `notebooks/kaggle/Project3_Merged_kaggle.ipynb`, hoặc chạy `kaggle kernels push -p notebooks/kaggle`.
2. Bật **GPU T4** và **Internet**.
3. Add Input các dataset: `hieutrinh123/project3-merged-assets`, `hieutrinh123/action-genome`, `dannywu375/visualgenome`, `mathurinache/visual-genome`, `clkmuhammed/microsoft-coco-2017-common-objects-in-context`.
4. Add Input các notebook nguồn (cache đặc trưng): `hieutrinh123/du-lieu-do-thi`, `lamvo30ai/project03-level4b-v2-graphfix`.
5. Run All. Mất khoảng 7–8 giờ: phần A ~4,5 giờ, B + C ~3 giờ.

Biến môi trường:
- `P3_RUN_L4B_V2` = `auto` / `1` / `0`: điều khiển phần C.
- `P3_SMOKE=1`: chạy thử phần B bằng dữ liệu giả.

Đổi phần A sang ViT-H/14: sửa `CFG.clip_model, CFG.clip_weights, CFG.clip_bs` thành `"ViT-H-14", "laion2b_s32b_b79k", 64`.

**MMKG trên Kaggle**
1. Repo private nên kernel không clone được. Sinh script bằng `python mmkg/kaggle/build_kernel.py > kernel_script.py`; script nhúng `kg_data.py`, `images.csv` và notebook.
2. Push script với **GPU T4**, **Internet** bật và dataset `hieutrinh123/project3-mmkg-vn-images`.
3. Chạy khoảng 25 phút. Kết quả nằm ở `/kaggle/working/mmkg_out/`.

**Máy riêng:** `pip install -r requirements.txt`, rồi sửa các đường dẫn `/kaggle/input/...`. MMKG chạy được trực tiếp: `cd mmkg && jupyter nbconvert --execute Project3_MMKG_VN.ipynb`. Đặt `MMKG_SMOKE=1` để chạy thử bằng đặc trưng giả.
