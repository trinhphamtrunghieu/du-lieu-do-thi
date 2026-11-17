# Project 3 — Truy vấn đa phương thức có giải thích bằng đồ thị

Đề bài: `Project03_Multimodal_XAI.pdf`. Kết quả: `RESULTS.md`. Đối chiếu với đề bài: `REQUIREMENTS_CHECK.md`.

## Cấu trúc

| Đường dẫn | Nội dung |
| --- | --- |
| `merged/Project3_Merged.ipynb` | Notebook gộp, chạy một lần. Gồm phần A (Flickr8k + Action Genome), phần C (Level 4(b) v2-graphfix, MSVD + MSR-VTT) và phần B (Visual Genome, COCO, ARO, SugarCrepe, COCO-spatial, video MSVD v5) |
| `merged/kaggle/` | Bản Kaggle của notebook gộp (< 1 MB; payload lớn nằm trong dataset `hieutrinh123/project3-merged-assets`), bản chỉ chạy phần B + C, `kernel-metadata.json` |
| `du-lieu-do-thi.ipynb` | Notebook gốc của phần A |
| `image_video_pipeline/` | Notebook gốc của phần B và hướng dẫn chạy riêng |
| `level4b_v2_graphfix/` | Notebook gốc của phần C và kết quả lần chạy gốc |
| `results/` | Output thật từ Kaggle: `project3-merged-v1/` (phần A), `project3-merged-v2/` (phần B + C), `vit-h14/` (phần A với CLIP ViT-H/14) |
| `mmkg/` | Cấp 3: MMKG Ẩm thực Việt Nam tự xây (148 thực thể, 750 triple, 316 ảnh), `DATA_CARD.md`, notebook `Project3_MMKG_VN.ipynb` đã chạy (giữ output), kết quả trong `mmkg/results/` |

## Chạy lại trên Kaggle

1. Tạo notebook mới, upload `merged/kaggle/Project3_Merged_kaggle.ipynb`. Hoặc dùng `kaggle kernels push -p merged/kaggle` với `kernel-metadata.json` có sẵn.
2. Bật **GPU T4** và **Internet**.
3. **Add Input** các dataset:
   - `hieutrinh123/project3-merged-assets`
   - `hieutrinh123/action-genome`
   - `dannywu375/visualgenome`
   - `mathurinache/visual-genome`
   - `clkmuhammed/microsoft-coco-2017-common-objects-in-context`
4. **Add Input** các notebook nguồn (cache đặc trưng):
   - `hieutrinh123/du-lieu-do-thi`
   - `lamvo30ai/project03-level4b-v2-graphfix`
5. Run All. Thời gian khoảng 7–8 giờ: phần A ~4,5 giờ, phần B + C ~3 giờ. Nếu không gắn output của v2-graphfix, phần C chạy lại đầy đủ và tổng lên khoảng 10 giờ.

Biến môi trường:
- `P3_RUN_L4B_V2`: `auto` (mặc định) / `1` / `0`, điều khiển phần C.
- `P3_SMOKE=1`: chạy thử phần B bằng dữ liệu giả, không phải kết quả.

Đổi backbone phần A sang ViT-H/14: sửa hai dòng `CFG.clip_model, CFG.clip_weights, CFG.clip_bs = ...` thành `"ViT-H-14", "laion2b_s32b_b79k", 64`.

Chạy trên máy riêng: `pip install -r requirements.txt`, rồi sửa các đường dẫn `/kaggle/input/...` cho phù hợp. Notebook được viết cho Kaggle.

## Kết quả nằm ở đâu sau khi chạy

- Phần A: `/kaggle/working/results/` (Flickr8k), `/kaggle/working/output/action_genome/results/`
- Phần B: `/kaggle/working/p3_out/` (`RESULTS.md`, `summary.json`, `splits/`, `figures/`, `video/`)
- Phần C: `/kaggle/working/p3_out/video_v2/`

## Chạy lại MMKG (cấp 3)

Repo private nên kernel Kaggle không clone được.
1. Sinh script kernel: `python mmkg/kaggle/build_kernel.py > kernel_script.py`. Script nhúng `kg_data.py`, `images.csv` và notebook.
2. Push script với **GPU T4**, **Internet** bật, và dataset `hieutrinh123/project3-mmkg-vn-images` (316 ảnh, kiểm tra sha256).
3. Khoảng 25 phút. Notebook đã chạy nằm ở `/kaggle/working/mmkg_out/`.

Chạy máy riêng: `cd mmkg && jupyter nbconvert --execute Project3_MMKG_VN.ipynb`. Cần GPU, `open_clip_torch==3.2.0`, Internet để tải trọng số CLIP. Đặt `MMKG_SMOKE=1` để chạy thử bằng đặc trưng giả; kết quả lúc đó không có ý nghĩa.
