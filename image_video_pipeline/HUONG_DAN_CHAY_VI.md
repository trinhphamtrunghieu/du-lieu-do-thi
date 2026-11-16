# Notebook final: ảnh + video (không gồm MMKG và bộ 20 truy vấn)

**File:** `Project3_Final_Image_Video.ipynb`. Notebook tự chứa. Mã giống hệt bản đang chạy trên Kaggle: `lamvo30ai/project3-full-pipeline`, version 8.

## Nội dung

| Mục | Phần | Yêu cầu đề cương |
|---|---|---|
| 1–4 | Thiết lập, dữ liệu nhúng, lõi đồ thị, dữ liệu + luật chống rò rỉ | Mục 4, 6 |
| **4v** | **Video MSVD (text-to-video):** chạy nguyên vẹn notebook Level4B v5 (CLIP + PE-Core + đồ thị bằng chứng + Qwen2.5-VL) trong tiến trình riêng | Cấp 4 (b) |
| 5 | CLIP ViT-L/14, RelTR (đóng băng) | — |
| 6 / 6b / 6c | Văn bản → ảnh: bộ chấm đồ thị, GraphSAGE, SCENIR (ICML 2025) | Cấp 1, cấp 2 |
| 7 / 7b | ARO, SugarCrepe; cạnh hình học trên COCO-spatial | Cấp 2, cấp 4 (a) |
| 8 | XAI: fidelity, lật ảnh, validity, sparsity | Cấp 3 |
| 10 / 10b | Demo `search()`; 5 ca thành công + 5 ca thất bại | Cấp 1, mục 8 |
| 11a / 11 | Độ nhạy α, so sánh cách tạo cạnh, file split; bảng tổng hợp + đối chiếu đề cương | — |

## Chạy trên Kaggle

1. Tạo notebook mới. **Upload** `Project3_Final_Image_Video.ipynb`, hoặc dùng `kaggle kernels push -p .` với `kernel-metadata.json` trong thư mục này.
2. **Accelerator: GPU T4.** **Internet: On.**
3. **Datasets:**
   - `dannywu375/visualgenome`
   - `mathurinache/visual-genome`
   - `clkmuhammed/microsoft-coco-2017-common-objects-in-context`
4. **Notebook nguồn** (cache đặc trưng video): `lamvo30ai/project03-level4b-v2-graphfix`.
5. Run All. Thời gian khoảng **3 giờ**: video khoảng 2–2.5 giờ, ảnh khoảng 30 phút.

Kết quả nằm trong `/kaggle/working/p3_out/`: `RESULTS.md`, `summary.json`, `figures/`, `splits/`, `video/`.

## Lưu ý

- Nếu nhánh video lỗi, notebook ghi lại lỗi và phần ảnh vẫn chạy tiếp.
- Test MSVD đã được xem nhiều lần qua các phiên bản trước, nên số test video chỉ mang tính thăm dò.
- Chạy thử không cần GPU: đặt biến môi trường `P3_SMOKE=1`. Chế độ này dùng dữ liệu giả, không phải kết quả.
