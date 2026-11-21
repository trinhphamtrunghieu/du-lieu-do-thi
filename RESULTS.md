# Kết quả: đồ thị trong truy vấn đa phương thức (du-lieu-do-thi)

Cập nhật 17/11/2026 (thêm mục "Cấp 3 — MMKG Ẩm thực Việt Nam"). Số liệu là kết quả trên tập **TEST** của notebook Kaggle `hieutrinh123/du-lieu-do-thi`, trung bình 3 seed; mọi tham số (α, λ, ridge, γ, β, trọng số node bộ ba) chọn trên tập **val**. Khoảng trong ngoặc vuông là khoảng tin cậy 95% (bootstrap cặp theo truy vấn, 10.000 lần).

## Kết luận chính

- **Action Genome:** đồ thị cảnh (có nhãn hành động/bối cảnh Charades) + GNN + nhánh khớp truy vấn↔node đưa MRR từ **0,171 (CLIP) lên 0,650**, R@1 từ 10,4% lên 53,8%, R@10 từ 31,3% lên 85,9%.
- **Flickr8k:** đồ thị chỉ tăng khoảng **+0,03 MRR** (0,800 → 0,833 với ViT-L/14).
- **MMKG Ẩm thực Việt Nam tự xây (cấp 3):** CLIP đa ngữ 0,517 → **0,796** MRR. Truy vấn gián tiếp tăng từ 0,327 lên 0,783; bỏ thuộc tính KG thì mức tăng này mất hết. Chi tiết ở mục "Cấp 3" bên dưới.
- Xáo node giữa các mẫu làm mất mức tăng trên cả hai dataset ⇒ mức tăng đến từ **nội dung đúng** của đồ thị, không phải do thêm tham số.
- Điều kiện: gallery có **chú thích của con người** (caption 1–4 cho Flickr8k; nhãn Action Genome + Charades). Đồ thị sinh tự động từ ảnh (tagger, linear probe, BLIP) chỉ giúp tối đa +0,007 trên Flickr8k ViT-L.

## Phương pháp (mô hình cuối, version 9)

1. **Tìm ứng viên bằng CLIP.** Mã hoá truy vấn và ảnh/video (video = trung bình các frame, L2) bằng CLIP ViT-L/14 (`laion2b_s32b_b82k`), lấy top-100 ứng viên.
2. **Dựng đồ thị cảnh cho mỗi ảnh/video.** Node: đối tượng, quan hệ người–vật dạng bộ ba ⟨người, cầm, cốc⟩, khái niệm WordNet; cạnh mức lớp (scene / related / isa) chỉ tính trên tập train.
   - **Action Genome:** bổ sung node hành động ("open door"), node bối cảnh và bộ ba thời gian ⟨a trước b⟩ từ nhãn Charades.
   - **Flickr8k:** node lấy từ caption người viết 1–4; truy vấn là caption 0, không dùng làm node (kịch bản gallery đã được chú thích).
3. **GNN + đường đi trên đồ thị.** GraphSAGE 2 lớp (256 chiều), huấn luyện InfoNCE trên truy vấn train, CLIP đóng băng. Thêm điểm đường đi từ khái niệm trong truy vấn đến ảnh/video theo 7 mẫu đường đi (P0–P6); các đường đi này cũng dùng để sinh giải thích.
4. **Nhánh khớp truy vấn ↔ node đồ thị** (version 8–9). Học ridge dạng đối ngẫu, chỉ trên tập train, từ đặc trưng truy vấn [TF-IDF 1–2 gram ⊕ CLIP-text] sang vector node (IDF + L2) của ảnh/video đúng. Điểm khớp s_match(q, v) = ŷ_q · y_v. Dùng để:
   - kéo thêm ứng viên vào top-100: xếp hạng bước đầu bằng z(CLIP) + γ·z(s_match);
   - cộng vào điểm cuối với trọng số β.
   Khối node bộ ba có trọng số riêng w_tri ∈ {0; 0,25; 0,5; 1}, chọn trên val bằng match-only; val chọn w_tri = 0 ở mọi lần chạy (quan hệ vẫn được dùng qua GNN).
5. **Kết hợp điểm.** Điểm cuối = (1 − β)·[α·z(CLIP) + (1 − α)·(λ·z(GNN) + (1 − λ)·z(đường đi))] + β·z(s_match), z = chuẩn hoá theo từng truy vấn. α, λ, λ_ridge, γ, β chọn trên val.
6. **Đánh giá.** Tập test, 3 seed; chỉ số R@1 (độ chính xác top-1), R@10, MRR; kiểm định bootstrap cặp theo truy vấn (10.000 lần). Đối chứng: xáo node / nhãn hành động giữa các mẫu, bỏ node hành động, bỏ GNN (CLIP + Match), bỏ cạnh quan hệ, bỏ cạnh mức lớp; bộ truy vấn quan hệ khó và truy vấn đổi động từ.

**Cách đọc chỉ số:** R@1 là độ chính xác top-1 (tỉ lệ truy vấn có đáp án đúng ở hạng 1); R@10 là top-10 accuracy; MRR là trung bình 1/hạng của đáp án đúng, không phải tỉ lệ phần trăm.

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

## Cùng mô hình với backbone CLIP ViT-H/14 (version 11)

Code giữ nguyên như version 9, chỉ đổi backbone sang `ViT-H-14` (`laion2b_s32b_b79k`). CSV đầy đủ nằm trong `results/partA_vith14_flickr8k/` và `results/partA_vith14_action_genome/`.

| Tập | Mô hình | R@1 | R@10 | MRR | Δ MRR so với CLIP (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Action Genome | CLIP | 12,2% | 35,5% | 0,198 | — |
| | CLIP + Graph | 36,0% | 65,2% | 0,458 | +0,260 [+0,244; +0,277] |
| | CLIP + Match (không GNN) | 45,6% | 82,0% | 0,583 | +0,385 [+0,364; +0,406] |
| | **CLIP + Graph + Match** | **54,6%** | **86,8%** | **0,661 ± 0,002** | **+0,463 [+0,443; +0,483]** |
| Flickr8k | CLIP | 73,1% | 97,0% | 0,820 | — |
| | CLIP + Graph | 76,4% | 97,2% | 0,842 | +0,022 [+0,010; +0,035] |
| | **CLIP + Graph + Match** | **76,3%** | **97,4%** | **0,843 ± 0,009** | **+0,023 [+0,010; +0,037]** |

- So với ViT-L/14: CLIP mạnh hơn (Action Genome 0,171 → 0,198; Flickr8k 0,800 → 0,820). Mô hình cuối cũng cao hơn nhẹ: Action Genome 0,650 → 0,661, Flickr8k 0,833 → 0,843.
- Mức tăng nhờ đồ thị gần như giữ nguyên trên Action Genome (+0,46 đến +0,48). Trên Flickr8k, mức tăng giảm từ +0,033 xuống +0,023, vì CLIP đã mạnh hơn nên đồ thị còn ít chỗ để bù.
- Đối chứng: xáo nhãn hành động hạ Action Genome còn 0,453 (−0,208); xáo node Flickr8k còn 0,823, ngang CLIP. Bộ quan hệ khó của Action Genome: 0,965 so với CLIP 0,673. ΔRR khi đổi động từ: 0,100 (CLIP 0,037).

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
| 11 | Như v9, backbone CLIP ViT-H/14 (Action Genome MRR 0,661; Flickr8k 0,843) |
| 9 | Tách khối node bộ ba trong nhánh khớp, trọng số chọn trên val |
| 8 | Nhánh khớp truy vấn↔node (ridge TF-IDF + CLIP → vector node), bổ sung ứng viên và cộng vào điểm cuối |
| 7 | AG: node hành động, bối cảnh, bộ ba thời gian từ Charades; Flickr8k: node từ caption người viết 1–4 |
| 5 | Flickr8k: node từ caption BLIP sinh tự động |
| 4 | AG: đối chứng composite_swap, truy vấn đổi động từ |
| 2 | Nhãn linear probe, bộ ba cấp ảnh, bộ quan hệ khó, sửa fidelity |
| 1 | Bản gốc |

## Phần B — Visual Genome / ARO / SugarCrepe / MSVD (`notebooks/components/partB_image_video/`)

Notebook `notebooks/components/partB_image_video/Project3_Final_Image_Video.ipynb` là một pipeline riêng, bản Kaggle gốc `lamvo30ai/project3-full-pipeline` (version 8). Nó dùng **scene graph Visual Genome (GT) và scene graph RelTR** thay cho đồ thị của notebook `du-lieu-do-thi`. Nội dung:

- **TN1:** văn bản → ảnh trên VG + COCO, CLIP top-100 → điểm đường đi, cùng GraphSAGE và SCENIR.
- **TN2:** ảnh → caption trên ARO VG-Relation và SugarCrepe.
- **TN2b:** cạnh hình học trên COCO-spatial.
- **XAI:** fidelity, lật ảnh, validity, sparsity.
- **Video MSVD:** CLIP + PE-Core + đồ thị bằng chứng + Qwen2.5-VL.

Số dưới đây là **kết quả chạy thật** của notebook gộp, Kaggle `hieutrinh123/project3-merged` version 2 (phần C + B; CLIP ViT-L/14, 3 seed). Toàn bộ output nằm trong `results/partB_image_video/`. Số tái lập khớp hằng `REF` của nhóm trong phạm vi ±0,002 MRR.

**TN1 — văn bản → ảnh, full-gallery, Visual Genome + COCO (MRR, trung bình ± std 3 seed):**

| CLIP | Đường đi G-full | G-obj (chỉ đối tượng) | G-rand (đồ thị ảnh khác) | GraphSAGE | SCENIR (ICML 2025) |
| --- | --- | --- | --- | --- | --- |
| 0,7184 ± 0,022 | 0,7214 ± 0,021 | 0,7221 ± 0,021 | 0,7175 ± 0,023 | 0,7175 ± 0,022 | 0,7191 ± 0,021 |

Đồ thị chỉ tăng khoảng +0,003 MRR. Theo luật đăng ký trước của notebook, cả GraphSAGE và SCENIR đều "không dương": không hơn CLIP, cũng không hơn bản cạnh ngẫu nhiên hay bỏ cạnh. Đây là cùng xu hướng với Flickr8k: khi CLIP đã mạnh và đồ thị không thêm thông tin mới, mức tăng rất nhỏ.

**TN2 — ảnh → caption đúng (2 lựa chọn), độ chính xác (%):**

| Nhóm | CLIP | Chỉ đồ thị | CLIP + đồ thị | Đồ thị của mục khác | Bỏ vị trí |
| --- | --- | --- | --- | --- | --- |
| ARO – toàn bộ | 50,1 | 65,8 | **66,1** | 53,3 | 50,0 |
| ARO – trái/phải | 49,7 | 66,3 | **66,7** | 50,9 | 50,0 |
| ARO – vị từ | 50,5 | 66,3 | 65,2 | 60,3 | — |
| SugarCrepe – thay quan hệ | 69,9 | 54,4 | **70,3** | 50,6 | — |
| SugarCrepe – đổi chỗ đối tượng | 63,5 | 49,7 | 61,3 | 46,0 | — |
| COCO-spatial – toàn bộ (cạnh hình học) | 46,8 | **69,8** | 63,9 | 51,3 | 50,0 |

- Trên ARO, CLIP gần như đoán ngẫu nhiên; đồ thị RelTR đưa lên khoảng 66% (+16 điểm).
- Đối chứng đồ thị của mục khác và bỏ vị trí đều tụt về khoảng 50%, nên mức tăng đến từ đúng quan hệ.
- COCO-spatial (132 mục chấm được): lật ảnh trái–phải làm đồ thị đảo quyết định ở 87,9% mục, CLIP chỉ 8,3%.

**XAI (ARO, 622 mục):**
- Lật ảnh làm đồ thị đảo quyết định ở 87,9% mục, CLIP 4,4%.
- Xoá bằng chứng làm đổi quyết định ở 12,3% mục, xoá ngẫu nhiên 0%.
- Validity (khớp GT, IoU ≥ 0,5): cả hai đối tượng 54,2%, ít nhất một 92,1%.

**Video MSVD — notebook Level 4(b) v5 (CLIP + PE-Core + đồ thị bằng chứng + Qwen2.5-VL; 27.763 truy vấn, 670 video):**

| Mô hình | R@1 | R@10 | MRR |
| --- | --- | --- | --- |
| CLIP | 47,5% | 82,0% | 0,5931 |
| CLIP + đồ thị | 49,0% | 84,2% | 0,6109 |
| CLIP + PE-Core, không đồ thị | 59,2% | 88,4% | 0,6982 |
| **CLIP + PE-Core + đồ thị bằng chứng (M4)** | **60,2%** | **88,8%** | **0,7069** |
| M4 với đồ thị ngẫu nhiên / xáo | 59,2% | 88,3% | 0,6979 |

- M4 hơn CLIP +0,114 [+0,101; +0,127]. Phần lớn đến từ encoder PE-Core; riêng cấu trúc đồ thị chỉ thêm +0,009 [+0,004; +0,014] so với bản không đồ thị, và +0,009 so với đồ thị xáo.
- Test MSVD đã được xem 8 lần qua các phiên bản, nên các số này chỉ mang tính thăm dò.
- Fidelity video (60 truy vấn): xoá 3 bằng chứng hàng đầu làm tụt hạng ở 56,7% truy vấn, xoá 3 bằng chứng ngẫu nhiên chỉ 25%.

**Notebook gộp:** `notebooks/Project3_Merged.ipynb` chạy cả ba phần trong một lần, metadata Kaggle ở `notebooks/kernel-metadata.json`.
- Phần A: `du-lieu-do-thi` (Flickr8k + Action Genome, ViT-L/14).
- Phần B: notebook ảnh + video ở trên.
- Phần C: Level 4(b) v2-graphfix, nằm ở mục 4u, xem mục dưới.

Thời gian khoảng 7–8 giờ trên T4 nếu output của phần C đã được Add Input. Nếu chưa, phần C phải chạy lại đầy đủ và tổng lên khoảng 10 giờ.

## Phần C — Video MSVD / MSR-VTT, Level 4(b) v2-graphfix (`notebooks/components/partC_project03-level4b-v2-graphfix.ipynb`)

Notebook `lamvo30ai/project03-level4b-v2-graphfix` (version 1), text → video. Nó dựng đồ thị v3 có vị từ (agent / object / svo / prep), mô hình TypedSAGE và QC-RGAT, tagger học từ caption train và cạnh ConceptNet. Nó cũng tạo cache đặc trưng frame CLIP ViT-L/14 mà notebook video v5 (mục 4v) dùng lại.

Số dưới đây đọc từ `results/partC_level4b/graph_experiment_results.csv` và `graph_confirmatory_tests.csv`, là output thật của lần chạy trên Kaggle (khoảng 2,4 giờ).

| Tập | n truy vấn / video | Mô hình | R@1 | R@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| MSR-VTT fresh1k (xác nhận) | 20.000 / 1.000 | CLIP | 38,3% | 71,9% | 0,4960 |
| | | QC-RGAT, đồ thị đầy đủ (G3) | 39,1% | 73,2% | **0,5059** |
| | | Chỉ cạnh contains (G1) | 39,3% | 73,4% | 0,5076 |
| | | + ConceptNet (G9) | 39,1% | 73,3% | 0,5062 |
| MSVD test (thăm dò, lần nhìn test thứ 6) | 27.763 / 670 | CLIP | 47,5% | 82,0% | 0,5931 |
| | | QC-RGAT, đồ thị đầy đủ (G3) | 48,7% | 84,0% | **0,6082** |
| | | Chỉ cạnh contains (G1) | 48,9% | 84,1% | 0,6097 |

**Đọc kết quả:**
- Trên MSR-VTT (tập xác nhận), đồ thị hơn CLIP +0,0099 MRR [+0,0065; +0,0133], có ý nghĩa sau hiệu chỉnh Holm. Trên MSVD test, đồ thị hơn CLIP +0,015 [+0,009; +0,021].
- Các cạnh quan hệ không giúp: bản chỉ dùng cạnh contains còn nhỉnh hơn bản đầy đủ (−0,0017 trên MSR-VTT). Đúng vị từ hay vị từ bị xáo, topology thật hay đã đi dây lại, có hay không có ConceptNet đều không khác biệt có ý nghĩa.
- Nhóm truy vấn gián tiếp (4.089 caption): đồ thị hơn CLIP +0,0117 MRR [+0,0075; +0,0158].
- XAI: xoá phần giải thích làm tụt hạng ở 50% truy vấn, so với 0% khi xoá ngẫu nhiên.

Kết luận đầy đủ: `results/partC_level4b/level4b_conclusions.md`.

**Cách chạy riêng phần B:** xem `notebooks/components/partB_image_video/HUONG_DAN_CHAY_VI.md`. Cần GPU T4, Internet, 3 dataset VG / COCO và notebook nguồn `lamvo30ai/project03-level4b-v2-graphfix` cho cache video; tổng khoảng 3 giờ.

## Cấp 3 — MMKG Ẩm thực Việt Nam tự xây (`mmkg/`)

Notebook `mmkg/Project3_MMKG_VN.ipynb` (đã chạy, giữ output), Kaggle `hieutrinh123/project3-mmkg-vn` version 4, GPU T4, 17/11/2026. Data card: `mmkg/DATA_CARD.md`. Output: `results/mmkg/`.

**Dữ liệu.**
- 148 thực thể: 45 món, 54 nguyên liệu, 10 nhóm nguyên liệu, 12 loại món, 10 cách chế biến, 13 tỉnh, 4 vùng.
- 750 triple: 434 triple tri thức + 316 triple ảnh → món.
- 316 ảnh Wikimedia Commons, mỗi ảnh ghi rõ giấy phép (CC0 / CC BY / CC BY-SA / PD) và tác giả.
- Tri thức do nhóm biên soạn với hỗ trợ AI. Ảnh được lọc tự động rồi duyệt bằng mắt: 410 ứng viên → 316 ảnh.

**Phương pháp.** Giữ cấu trúc mô hình cuối của phần A, điều chỉnh cho MMKG:

s = α·z(CLIP) + (1−α)·z(Σ_d P(d|q)·[λ·p_GNN(d|ảnh) + (1−λ)·p_kNN(d|ảnh)])

- P(d|q): liên kết thực thể trong truy vấn với node KG; điểm đường đi 1–2 bước (món → tỉnh → vùng, món → nguyên liệu → nhóm); cộng khớp với mô tả node.
- p_GNN: R-GraphSAGE trên đồ thị ảnh + KG, học dự đoán cạnh ảnh–món. Cạnh `mô_tả` của ảnh val/test không có trong đồ thị.
- CLIP đa ngữ `xlm-roberta-large-ViT-H-14`, đóng băng.
- Mọi tham số chọn trên val; 3 seed chia ảnh theo món, nhóm ảnh gần trùng luôn cùng một tập.

**Kết quả trên test** (MRR, trung bình 3 seed):

| Cấu hình | Trực tiếp (42) | Gián tiếp (42) | Diễn đạt tự nhiên (24) | Tất cả (108) | R@1 tất cả |
| --- | --- | --- | --- | --- | --- |
| CLIP đa ngữ ViT-H/14 | 0,634 | 0,327 | 0,646 | 0,517 | 38,9% |
| **CLIP + KG** | **0,807** | **0,783** | **0,800** | **0,796 ± 0,024** | **71,6%** |
| Bỏ thuộc tính KG | 0,801 | 0,312 | 0,620 | 0,571 | 46,0% |
| KG ngẫu nhiên (xáo đuôi triple) | 0,793 | 0,334 | 0,610 | 0,574 | 46,3% |
| Xáo liên kết ảnh–món | 0,637 | 0,329 | 0,657 | 0,521 | 39,2% |

Bootstrap cặp, CLIP + KG − CLIP:
- Tất cả: +0,279 [+0,208; +0,351].
- Gián tiếp: +0,456 [+0,339; +0,575].
- Trực tiếp: +0,173 [+0,073; +0,276].
- Diễn đạt tự nhiên: +0,154 [+0,037; +0,284].

Đọc kết quả:
- **Truy vấn gián tiếp:** mức tăng đến từ thuộc tính trong KG. Bỏ thuộc tính hoặc xáo KG làm mất toàn bộ mức tăng.
- **Truy vấn trực tiếp:** mức tăng đến từ liên kết ảnh–món của ảnh train. Thuộc tính KG không giúp gì (+0,006, không có ý nghĩa). So sánh này không hoàn toàn cân bằng: CLIP là zero-shot, còn nhánh đồ thị dùng nhãn món của ảnh train.
- **Diễn đạt tự nhiên** (câu viết tay, dùng từ ngoài KG như "heo", "miền Tây"): vẫn tăng +0,154. Mức tăng này nhỏ hơn và khoảng tin cậy rộng hơn so với truy vấn sinh theo mẫu.

**Thành phần** (seed 0, giữ tham số chọn trên val, tắt từng phần, MRR tất cả):

| Cấu hình | MRR |
| --- | --- |
| Tham số chọn trên val (α = 0,5, λ = 0,5) | 0,829 |
| Chỉ đồ thị (α = 0) | 0,804 |
| Chỉ CLIP | 0,525 |
| Ảnh→món chỉ kNN (λ = 0) | 0,766 |
| Ảnh→món chỉ GNN (λ = 1) | 0,842 |
| Truy vấn→món chỉ mô tả node, không dùng ràng buộc KG | 0,666 |

GNN tốt hơn kNN 0,08 MRR khi dùng một mình. Val chọn λ = 0,5 nên trên test thấp hơn GNN thuần một chút.

**xAI** (108 truy vấn test, seed 0; giải thích = món trung gian + đường đi KG + ≤ 3 ảnh train làm bằng chứng):

| Chỉ số | Giá trị |
| --- | --- |
| Fidelity: độ tụt điểm khi xoá cạnh trong giải thích | **1,91** (xoá cùng số cạnh ngẫu nhiên: 0,01) |
| Fidelity: tỉ lệ ảnh top-1 bị tụt hạng | **72%** (ngẫu nhiên: 6,5%) |
| Validity: món trong giải thích đúng là món của ảnh top-1 | 79,6% |
| Validity: triple trong giải thích có thật trong KG | 100% |
| Sparsity: số cạnh trong giải thích | 5,0 (món trung gian chiếm 97% điểm đồ thị) |

Fidelity là cận dưới vì mô hình không huấn luyện lại và mô tả văn bản của node giữ nguyên.

Khi ảnh top-1 sai, giải thích cho thấy lý do. Ví dụ ảnh Bánh ít được kéo lên vì mô hình đoán nó là Bánh bột lọc.

**Khả năng chuyển giao** (MRR tất cả, CLIP → CLIP + KG):

| Thiết lập | CLIP | + KG | Δ |
| --- | --- | --- | --- |
| CLIP đa ngữ ViT-H/14, tiếng Việt có dấu | 0,517 | 0,796 | +0,279 |
| CLIP đa ngữ ViT-H/14, tiếng Việt không dấu | 0,299 | 0,767 | +0,468 |
| CLIP đa ngữ ViT-H/14, truy vấn tiếng Anh (dịch từ KG) | 0,465 | 0,770 | +0,305 |
| CLIP đa ngữ ViT-B/32, tiếng Việt | 0,366 | 0,674 | +0,308 |
| CLIP tiếng Anh ViT-L/14 (backbone phần A), truy vấn tiếng Việt | 0,135 | 0,635 | +0,500 |
| CLIP tiếng Anh ViT-L/14, truy vấn tiếng Anh | 0,413 | 0,718 | +0,305 |

Phần chuyển giao được:
- Cách trộn z-score + α chọn trên val, GNN trên đồ thị ảnh–node và điểm đường đi giữ nguyên từ phần A.
- Giống Action Genome (0,171 → 0,650), đồ thị có ích nhất khi CLIP thiếu thông tin. Ở đây là tiếng Việt, mất dấu, hoặc thuộc tính không nhìn thấy trong ảnh (xuất xứ, nguyên liệu).
- Với CLIP tiếng Anh, KG gần như bù được khoảng cách ngôn ngữ (0,135 → 0,635).

Phần không chuyển được:
- Nhánh ridge truy vấn → node của phần A không dùng được vì MMKG không có caption để học. Liên kết thực thể và mô tả node thay vai trò này.
- Cần tự viết bảng bí danh cho từ đồng nghĩa.

**Hạn chế của MMKG:**
- Truy vấn gián tiếp sinh theo mẫu nên dùng đúng từ vựng của KG. Liên kết thực thể đúng 88/88, là điều kiện thuận lợi.
- Tri thức và việc duyệt ảnh chưa có người kiểm tra độc lập.
- Mỗi món chỉ có 1–10 ảnh. Test có 42 món, nên khoảng tin cậy rộng.

## Hạn chế và việc tiếp theo

- Mọi mức tăng cần gallery có chú thích của con người.
- Flickr8k: đồ thị vẫn thua CLIP ở bộ quan hệ khó; 16 truy vấn gián tiếp bị tụt.
- Nhánh khớp coi node như túi từ ⇒ kém nhạy với động từ hơn GNN.
- Phần 4 (MSVD → MSR-VTT) chưa chạy lại với các cải tiến mới (kết quả cũ: CLIP 0,505 → 0,514).

## Nguồn

Mọi output thật nằm trong `results/`; bảng tra cứu ở `results/README.md`.

- Phần A, CLIP ViT-L/14: `results/partA_vitl14_flickr8k/`, `results/partA_vitl14_action_genome/`. Đây là lần chạy lại mô hình version 9 trong notebook gộp (`hieutrinh123/project3-merged` version 1): Action Genome 0,642, Flickr8k 0,839. Số version 9 trong các bảng trên (0,650 / 0,833) lấy từ kernel `hieutrinh123/du-lieu-do-thi`; CSV của các version cũ không lưu trong repo.
- Phần A, CLIP ViT-H/14: `results/partA_vith14_*/` (kernel `hieutrinh123/du-lieu-do-thi` version 11).
- Phần B: `results/partB_image_video/` (`hieutrinh123/project3-merged` version 2).
- Phần C: `results/partC_level4b/`.
- MMKG: `results/mmkg/` (`hieutrinh123/project3-mmkg-vn` version 4).
- Log Kaggle: `results/logs/`.
