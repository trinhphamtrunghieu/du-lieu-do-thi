# Kết quả: đồ thị trong truy vấn đa phương thức (du-lieu-do-thi)

Cập nhật 16/11/2026. Số liệu là kết quả trên tập **TEST** của notebook Kaggle `hieutrinh123/du-lieu-do-thi`, trung bình 3 seed; mọi tham số (α, λ, ridge, γ, β, trọng số node bộ ba) chọn trên tập **val**. Khoảng trong ngoặc vuông là khoảng tin cậy 95% (bootstrap cặp theo truy vấn, 10.000 lần).

## Kết luận chính

- **Action Genome:** đồ thị cảnh (có nhãn hành động/bối cảnh Charades) + GNN + nhánh khớp truy vấn↔node đưa MRR từ **0,171 (CLIP) lên 0,650**, R@1 từ 10,4% lên 53,8%, R@10 từ 31,3% lên 85,9%.
- **Flickr8k:** đồ thị chỉ tăng khoảng **+0,03 MRR** (0,800 → 0,833 với ViT-L/14).
- Xáo node giữa các mẫu làm mất mức tăng trên cả hai dataset ⇒ mức tăng đến từ **nội dung đúng** của đồ thị, không phải do thêm tham số.
- Điều kiện: gallery có **chú thích của con người** (caption 1–4 cho Flickr8k; nhãn Action Genome + Charades). Đồ thị sinh tự động từ ảnh (tagger, linear probe, BLIP) chỉ giúp tối đa +0,007 trên Flickr8k ViT-L.

## So sánh với các bản trước (TEST)

| Bộ dữ liệu / chỉ số | CLIP | v2 | v4 | v5 | v7 | v8 | v9 (cuối) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flickr8k MRR (ViT-L/14) | 0.7996 | 0.8025 | — | 0.8062 | 0.8319 | 0.8318 | **0.8325 ± 0.0015** (không GNN: 0.8339) |
| Flickr8k MRR (ViT-B/32) | 0.7227 | — | 0.7292 | 0.7382 | 0.7907 | 0.7954 | **0.7947 ± 0.0013** |
| Flickr8k quan hệ khó MRR (246) | 0.878 | — | — | — | 0.861 | 0.865 | **0.865** |
| Action Genome MRR | 0.171 | — | 0.371 | — | 0.417 | 0.526 | **0.6501 ± 0.0007** (không GNN: 0.574) |
| AG R@1 | 0.104 | — | — | — | 0.324 | 0.384 | **0.538** |
| AG R@10 | 0.313 | — | — | — | 0.600 | — | **0.859** |
| AG quan hệ khó MRR | 0.682 / 0.626 | — | 0.864 | — | 0.923 | 0.941 | **0.964** |
| AG phản thực tế ΔRR | +0.052 / +0.045 | — | +0.188 | — | +0.207 | +0.166 | **+0.098** |

Cột v2–v9 là version notebook trên Kaggle; v9 là CLIP + Graph + Match, các cột trước là CLIP + Graph (v8 là bản Match đầu tiên). "—" = version đó không chạy hoặc không lưu chỉ số này. Bộ quan hệ khó và bộ phản thực tế của AG được dựng lại từ v7 (548 → 685 truy vấn; 545 → 683), nên CLIP có hai giá trị: v4 / từ v7. ΔRR càng cao càng nhạy với động từ; v9 đổi một phần độ nhạy này lấy độ chính xác. v3 và v6 không ra kết quả (lỗi / bị thay); v10–v11 là bản ViT-H đang chạy.

## Kết quả cuối (version 9, CLIP ViT-L/14)

**Action Genome** (text → video, 1.810 truy vấn test)

| Mô hình | MRR | R@1 | R@10 | Δ MRR so với CLIP |
| --- | --- | --- | --- | --- |
| CLIP | 0,171 | 10,4% | 31,3% | — |
| CLIP + Graph | 0,429 | 34,1% | 60,2% | +0,258 [+0,241; +0,274] |
| CLIP + Match (không GNN) | 0,574 | 44,5% | 81,2% | +0,403 [+0,382; +0,424] |
| **CLIP + Graph + Match** | **0,650** | **53,8%** | **85,9%** | **+0,479 [+0,458; +0,499]** |

**Flickr8k** (text → ảnh, 1.000 truy vấn test; node từ caption người viết 1–4, truy vấn là caption 0)

| Mô hình | ViT-L/14 MRR | ViT-L/14 R@1 | ViT-B/32 MRR | ViT-B/32 R@1 |
| --- | --- | --- | --- | --- |
| CLIP | 0,800 | 70,6% | 0,723 | 61,2% |
| CLIP + Graph | 0,829 | 74,4% | 0,787 | 68,7% |
| CLIP + Match (không GNN) | 0,834 | 75,4% | 0,778 | 66,9% |
| **CLIP + Graph + Match** | **0,833** | **74,9%** | **0,795** | **69,6%** |

ViT-L: CLIP + Graph + Match hơn CLIP +0,033 [+0,018; +0,048], nhưng không hơn CLIP + Graph (+0,003, n.s.). ViT-B: hơn CLIP +0,072 [+0,054; +0,090].

## Thành phần nào đóng góp (Action Genome)

So mô hình cuối (MRR 0,650) với từng cấu hình bị bớt một thành phần, trên cùng truy vấn test:

| Cấu hình so sánh | MRR | Mô hình cuối hơn (95% CI) | Kết luận |
| --- | --- | --- | --- |
| Xáo nhãn hành động giữa các video | 0,437 | +0,213 [+0,196; +0,229] | Mức tăng đến từ nội dung đúng của node |
| Bỏ node hành động / bối cảnh Charades | 0,509 | +0,141 [+0,128; +0,155] | Nhãn hành động là nguồn thông tin lớn nhất |
| Bỏ GNN + đường đi (CLIP + Match) | 0,574 | +0,076 [+0,066; +0,087] | GNN vẫn có đóng góp khi đã có nhánh khớp |
| Bỏ cạnh quan hệ (bộ ba) | 0,606 | +0,045 [+0,035; +0,054] | Quan hệ có ích, qua GNN |
| Bỏ cạnh mức lớp | 0,620 | +0,030 [+0,022; +0,038] | Cạnh mức lớp có ích nhỏ |
| Bỏ nhánh khớp (CLIP + Graph) | 0,429 | +0,221 [+0,202; +0,241] | Nhánh khớp là cải tiến lớn nhất của v8–v9 |

**Đổi động từ trong truy vấn** (683 truy vấn, video đúng không còn khớp): độ tụt ΔRR của CLIP 0,045; CLIP + Graph 0,195; CLIP + Graph + Match 0,098. Nhánh khớp tăng độ chính xác nhưng kém nhạy với động từ hơn GNN.

**Flickr8k:** xáo node giữa các ảnh làm mất toàn bộ mức tăng (0,799 = CLIP). Cạnh quan hệ và cạnh mức lớp không có tác dụng đáng kể. Trên bộ quan hệ khó, đồ thị vẫn thua CLIP (0,865 so với 0,878).

## Vì sao điểm tuyệt đối trên Action Genome thấp

CLIP trên AG chỉ đạt R@1 10,4% / R@10 31,3% (Flickr8k: 70,6% / 96,9%). Lý do 1 có số liệu; 2–5 là đặc điểm dữ liệu, chưa đo trực tiếp.

1. CLIP yếu trên video Charades: quay trong nhà, tối, mờ; mỗi video biểu diễn bằng vài frame gộp lại.
2. Các video rất giống nhau: cùng phòng ở, cùng bộ đồ vật và hành động thường ngày.
3. Mô tả Charades chung chung, do crowdsourcing viết; nhiều video cùng khớp một câu.
4. Chỉ một video được tính là đúng, nên MRR / R@1 đánh giá thấp chất lượng thực tế.
5. Từ vựng node nhỏ (~35 vật, 25 quan hệ, 157 hành động); chỉ xếp hạng lại top-100 trong 1.810 ứng viên.

## Các thay đổi theo version

| Version Kaggle | Thay đổi chính |
| --- | --- |
| 11 | Như v9, backbone CLIP ViT-H/14 (đang chạy) |
| 9 | Tách khối node bộ ba trong nhánh khớp, trọng số chọn trên val |
| 8 | Nhánh khớp truy vấn↔node (ridge TF-IDF + CLIP → vector node), bổ sung ứng viên và cộng vào điểm cuối |
| 7 | AG: node hành động, bối cảnh, bộ ba thời gian từ Charades; Flickr8k: node từ caption người viết 1–4 |
| 5 | Flickr8k: node từ caption BLIP sinh tự động |
| 4 | AG: đối chứng composite_swap, truy vấn đổi động từ |
| 2 | Nhãn linear probe, bộ ba cấp ảnh, bộ quan hệ khó, sửa fidelity |
| 1 | Bản gốc |

## Hạn chế và việc tiếp theo

- Mọi mức tăng cần gallery có chú thích của con người.
- Flickr8k: đồ thị vẫn thua CLIP ở bộ quan hệ khó; 16 truy vấn gián tiếp bị tụt.
- Nhánh khớp coi node như túi từ ⇒ kém nhạy với động từ hơn GNN.
- Phần 4 (MSVD → MSR-VTT) chưa chạy lại với các cải tiến mới (kết quả cũ: CLIP 0,505 → 0,514).
- Đang chạy: bản ViT-H/14 (Kaggle version 11) để so cùng backbone.

## Nguồn

Output của notebook Kaggle `hieutrinh123/du-lieu-do-thi` (version 4, 7, 8, 9): `results/` (Flickr8k) và `output/action_genome/results/` (Action Genome) — `main_results_with_controls.csv`, `match_vs_clip.csv`, `match_paired.csv`, `relational_hard_negatives_match*.csv`, `counterfactual_verb_swap_match*.csv`, `paired_graph_vs_clip.csv`.
