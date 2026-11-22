# Đối chiếu với đề bài Project 3 (`docs/Project03_Multimodal_XAI.pdf`)

Cập nhật 17/11/2026, sau khi notebook gộp `notebooks/Project3_Merged.ipynb` chạy xong trên Kaggle (`hieutrinh123/project3-merged`). Phần A chạy ở version 1, phần C + B ở version 2. MMKG cấp 3 (`mmkg/`) chạy ở Kaggle `hieutrinh123/project3-mmkg-vn` version 4.

Ký hiệu:
- **Đạt:** có code và output thật trong repo.
- **Một phần:** có, nhưng chưa đủ như đề yêu cầu.
- **Thiếu:** chưa có.

Phần A = `du-lieu-do-thi` (Flickr8k + Action Genome); phần B = pipeline ảnh/video (Visual Genome, COCO, ARO, SugarCrepe, COCO-spatial, MSVD v5); phần C = Level 4(b) v2-graphfix (MSVD + MSR-VTT).

## Tóm tắt

| Cấp | Điểm trần | Tình trạng |
| --- | --- | --- |
| 1 | 7,0 | Đạt đủ |
| 2 | 8,5 | Đạt đủ |
| 3 | 10 | Đạt: XAI định lượng và MMKG tự xây tiếng Việt (`mmkg/`) |
| 4 (mở rộng) | +1,0–2,0 | Đạt hướng (a) và (b) |
| Sản phẩm nộp | — | **Thiếu** báo cáo 12–18 trang, slide, bảng phân công, video demo dự phòng |

## Cấp 1 — Nhiệm vụ 1 và 2

| Yêu cầu | Tình trạng | Bằng chứng |
| --- | --- | --- |
| Baseline CLIP đóng băng trên ≥ 1 tập công khai | Đạt | CLIP ViT-L/14 đóng băng ở mọi phần. Flickr8k MRR 0,800; Action Genome 0,171; VG/COCO 0,718. Xem `RESULTS.md` |
| Đồ thị từ dữ liệu có sẵn (scene graph VG, nhãn) | Đạt | Phần B: scene graph GT Visual Genome và RelTR. Phần A: nhãn Action Genome + Charades, caption Flickr8k |
| ≥ 1 thuật toán đã học (GCN/GAT/GraphSAGE) | Đạt | GraphSAGE ở phần A (GNN chính) và phần B (TN1b); TypedSAGE / QC-RGAT ở phần C |
| Xếp hạng lại α·cos + (1−α)·điểm đồ thị; so CLIP với CLIP + Graph trên cùng split | Đạt | `results/partA_vitl14_*/paired_graph_vs_clip.csv`; phần B TN1; `results/partC_level4b/graph_confirmatory_tests.csv` |
| Ablation bỏ cạnh / cạnh ngẫu nhiên | Đạt | Phần A: `no_relations`, `no_class_edges`, xáo node, xáo nhãn hành động. Phần B: G-rand, SAGE-rand, SAGE-0, "đồ thị của mục khác". Phần C: H2–H11 |
| Hiển thị quan hệ/đường đi khi xếp hạng | Đạt | Demo `demo_query()` ở phần A, `search()` ở phần B; `fidelity_per_query.csv` |
| Phân tích 5 thành công + 5 thất bại | Đạt (có dữ liệu) | `results/partA_vitl14_*/case_analysis.json` (10 ca mỗi tập); `results/partB_image_video/cases_text_to_image.json`. Phần phân tích bằng lời vẫn cần viết vào báo cáo |

## Cấp 2 — Nhiệm vụ 2

| Yêu cầu | Tình trạng | Bằng chứng |
| --- | --- | --- |
| ≥ 2 tập công khai | Đạt | Flickr8k, Action Genome, Visual Genome + COCO, ARO, SugarCrepe, COCO-spatial, MSVD, MSR-VTT |
| 1 phương pháp mới từ 2024 trở lại, cùng pipeline/split | Đạt | SCENIR (ICML 2025), TN1c phần B: MRR 0,7191 so với CLIP 0,7184 (không dương). Phần A có cell SCENIR nhưng đang tắt |
| Bộ ≥ 20 truy vấn có nhóm gián tiếp, báo cáo tách nhóm | Đạt | `results/partA_vitl14_*/eval_queries_20.json` (10 trực tiếp + 10 gián tiếp, chỉ lấy từ test); kết quả tách nhóm direct/indirect trong `match_vs_clip.csv`, `paired_graph_vs_clip.csv`. Phần C có bộ 20 truy vấn gián tiếp đăng ký trước (`eval_queries_level4b.json`) |
| Bộ 20 truy vấn chỉ dùng để đánh giá | Đạt | Truy vấn lấy từ test, không dùng để chọn tham số (ghi trong file JSON) |
| Ablation xáo trộn / đồ thị ngẫu nhiên | Đạt | Như cấp 1 |
| So sánh các cách tạo cạnh (mở rộng) | Đạt | Phần B: bảng "So sánh các cách tạo cạnh" trong `results/partB_image_video/REPORT_generated.md`; phần A: none / no_class_edges / no_relations / no_actions |
| Phân tích độ nhạy α (mở rộng) | Đạt | `alpha_sensitivity.csv` (phần A); `figures/alpha_sensitivity.png` trong output Kaggle (phần B, chưa lưu ảnh vào repo) |

## Cấp 3 — Nhiệm vụ 3 (xAI) + dữ liệu tự xây

| Yêu cầu | Tình trạng | Bằng chứng |
| --- | --- | --- |
| Fidelity định lượng ≥ 10 truy vấn | Đạt | Phần A: 20 truy vấn mỗi tập (`fidelity_per_query.csv`); bỏ giải thích làm điểm tụt 2,69 (Flickr8k) và 2,40 (AG), so với 1,12 và 0,42 khi bỏ ngẫu nhiên. Phần B: 622 mục ARO, xoá bằng chứng đổi quyết định 12,3% so với 0% ngẫu nhiên. Video: 56,7% so với 25% |
| Validity | Đạt | Phần A: validity 1,0 (đường đi có thật trong đồ thị). Phần B: khớp GT IoU ≥ 0,5: 54,2% cả hai, 92,1% ít nhất một |
| Sparsity | Đạt | Phần A: 0,50 (Flickr8k), 0,80 (AG); phần B: `xai_items.csv` |
| Giải thích đến từ chính cơ chế xếp hạng | Đạt | Phần A: điểm đường đi là một số hạng của điểm cuối (λ < 1 bắt buộc); phần C: giải thích = số hạng của điểm, Σ = điểm |
| Tự xây MMKG nhỏ (ưu tiên tiếng Việt), 50–150 thực thể, 300–1.000 triple, 200–500 ảnh, kèm data card | Đạt | `mmkg/`: MMKG Ẩm thực Việt Nam, 148 thực thể, 750 triple (434 tri thức + 316 ảnh–món), 316 ảnh Wikimedia Commons có giấy phép. Data card `mmkg/DATA_CARD.md` mô tả nguồn, cách tạo triple, cách liên kết ảnh–thực thể, điều kiện sử dụng |
| Chạy phương pháp tốt nhất cấp 1–2 trên MMKG, phân tích khả năng chuyển giao | Đạt | `mmkg/Project3_MMKG_VN.ipynb` (có output, Kaggle `hieutrinh123/project3-mmkg-vn` v4). MRR test 0,517 → 0,796; gián tiếp 0,327 → 0,783. Có ablation bỏ thuộc tính / KG ngẫu nhiên / xáo liên kết ảnh và xAI (fidelity 1,91 so với 0,01 ngẫu nhiên). Có phân tích chuyển giao (CLIP đa ngữ / tiếng Anh, có dấu / không dấu / tiếng Anh, so với Flickr8k và AG) trong `RESULTS.md`, mục "Cấp 3" |

## Cấp 4 — Mở rộng tự chọn

| Hướng | Tình trạng | Bằng chứng |
| --- | --- | --- |
| (a) Thuật toán: kết hợp điểm thích nghi / cơ chế xếp hạng lại, có giả thuyết + ablation + ≥ 2 tập | Đạt | Phần A: nhánh khớp truy vấn↔node + kết hợp thích nghi, thử trên Flickr8k và Action Genome. AG: 0,429 → 0,650, có đầy đủ ablation và kiểm định cặp. Phần B: cạnh hình học trên ARO và COCO-spatial |
| (b) Video (text-to-video) | Đạt | Action Genome (0,171 → 0,650); MSVD v5 (0,593 → 0,707); MSR-VTT xác nhận (0,496 → 0,506) |
| (c) Khảo sát người dùng | Không làm | — |

## Yêu cầu chia dữ liệu và chống rò rỉ (mục 6)

| Yêu cầu | Tình trạng | Bằng chứng |
| --- | --- | --- |
| Test không dùng để huấn luyện / chọn tham số / xây cạnh thống kê | Đạt | Cạnh mức lớp và IDF chỉ tính trên train; nhánh khớp học trên train; α, λ, β… chọn trên val |
| Cạnh thống kê xây từ tập huấn luyện | Đạt | `MultimodalGraph` (phần A) chỉ đếm trên train; phần B lọc rò rỉ benchmark |
| α và mọi siêu tham số chọn trên validation | Đạt | `match_log.csv`, `final_model_val_selection.csv`; phần B chọn α trên TUNE |
| Cùng một tệp chia; nộp split hoặc seed + mã tạo split | Đạt | Danh mục: `splits/README.md`. Phần A: `splits/partA_flickr8k_split.json`, `splits/partA_action_genome_split.json` (bản gốc từ output Kaggle) + mã tạo lại `splits/make_splits.py`, có kiểm tra sha1. Phần B: `results/partB_image_video/splits/`. Phần C: `split_manifest.json`. MMKG: `results/mmkg/splits.json` |
| ≥ 3 seed, báo cáo trung bình ± độ lệch chuẩn | Đạt | Mọi thí nghiệm chính: seed 0, 1, 2 |

## Đánh giá (mục 7)

| Yêu cầu | Tình trạng |
| --- | --- |
| R@1, R@5, R@10, MRR cho từng chiều | Đạt (cả nDCG@10 ở phần A) |
| Chiều ảnh → văn bản (mở rộng) | Đạt (ARO, SugarCrepe, COCO-spatial ở phần B) |
| Tách nhóm trực tiếp / gián tiếp | Đạt |
| So sánh giải thích giữa các mô hình (mở rộng) | Một phần: phần A so fidelity giữa cấu hình; chưa có bảng so sánh riêng |

## Sản phẩm cần nộp (mục 8)

| Sản phẩm | Tình trạng | Ghi chú |
| --- | --- | --- |
| Mã nguồn tiền xử lý, trích đặc trưng, tạo đồ thị, huấn luyện, đánh giá | Đạt | `notebooks/Project3_Merged.ipynb` và các notebook thành phần |
| File requirements/environment + hướng dẫn chạy lại | Đạt (vừa thêm) | `requirements.txt`, `README.md`; phiên bản thật trong `results/partC_level4b/env_versions.json` |
| Tập chia cố định / mã tạo split, mô tả chống rò rỉ | Đạt | Xem mục 6 ở trên |
| Báo cáo kỹ thuật 12–18 trang, slide, bảng phân công | **Thiếu** | `RESULTS.md` là nguồn số liệu; cần viết báo cáo và slide |
| Notebook nhập truy vấn → top-k + giải thích | Đạt | `demo/app.py` (Gradio, 3 tab: MMKG ẩm thực VN, Action Genome, MSVD; Space https://huggingface.co/spaces/trinhphamtrunghieu/du-lieu-do-thi), `demo/Demo_Truy_Van.ipynb` (ipywidgets, MMKG). Engine AG/MSVD port từ notebook phần A/C, khớp kết quả tham chiếu của notebook (26 + 20 truy vấn, sai số điểm < 1e-6); ngoài ra `demo_query()` (phần A, C), `search()` (phần B, MMKG) |
| Phân tích chi tiết 5 thành công + 5 thất bại | Một phần | Có dữ liệu (`case_analysis.json`); cần viết phân tích bằng lời |
| Bộ ≥ 20 truy vấn kèm mô tả cách xây | Đạt | `eval_queries_20.json` có trường `construction` |
| MMKG tự xây + data card (cấp 3) | Đạt | `mmkg/` và `mmkg/DATA_CARD.md` |
| Demo trực tiếp + video dự phòng ≤ 5 phút | Thiếu video | Notebook chạy được; chưa quay video |
| Khai báo dùng mã nguồn mở / công cụ AI trong báo cáo | Cần ghi trong báo cáo | — |

## Kết quả cần lưu ý khi trình bày

- Trên Visual Genome + COCO (phần B, TN1), cả GraphSAGE và SCENIR đều không hơn CLIP. Điều này nhất quán với Flickr8k: khi CLIP đã mạnh và đồ thị không thêm thông tin mới, mức tăng rất nhỏ.
- Đồ thị có hiệu quả rõ ở những nơi CLIP yếu và đồ thị mang thông tin mới:
  - Action Genome: 0,171 → 0,650.
  - ARO: 50% → 66%.
  - COCO-spatial: 47% → 70% (chỉ đồ thị).
- Video MSVD: phần lớn mức tăng đến từ encoder PE-Core; riêng cấu trúc đồ thị chỉ thêm +0,009 MRR. Test MSVD đã xem 8 lần nên chỉ là thăm dò.
- MMKG ẩm thực: truy vấn gián tiếp sinh theo mẫu dùng đúng từ vựng của KG, nên phương pháp đồ thị có lợi thế. Bộ 24 câu viết tay dùng từ ngoài KG vẫn tăng (+0,154 MRR), nhưng ít hơn. Với truy vấn trực tiếp, mức tăng đến từ nhãn món của ảnh train, không phải từ thuộc tính KG.
