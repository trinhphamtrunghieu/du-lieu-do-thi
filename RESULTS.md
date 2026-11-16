# Kết quả: đồ thị trong truy vấn đa phương thức (du-lieu-do-thi)

Cập nhật 16/11/2026. Số liệu là kết quả trên tập **TEST** của notebook Kaggle `hieutrinh123/du-lieu-do-thi`, trung bình 3 seed; mọi tham số (α, λ, ridge, γ, β, trọng số node bộ ba) chọn trên tập **val**. Khoảng trong ngoặc vuông là khoảng tin cậy 95% (bootstrap cặp theo truy vấn, 10.000 lần).

## Kết luận chính

- **Action Genome:** đồ thị cảnh (có nhãn hành động/bối cảnh Charades) + GNN + nhánh khớp truy vấn↔node đưa MRR từ **0,171 (CLIP) lên 0,650**, R@1 từ 10,4% lên 53,8%, R@10 từ 31,3% lên 85,9%.
- **Flickr8k:** đồ thị chỉ tăng khoảng **+0,03 MRR** (0,800 → 0,833 với ViT-L/14).
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

Code giữ nguyên như version 9, chỉ đổi backbone sang `ViT-H-14` (`laion2b_s32b_b79k`). CSV đầy đủ nằm trong `results/vit-h14/`.

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

## Nhánh Visual Genome / ARO / SugarCrepe / MSVD (notebook `image_video_pipeline/`)

Notebook `image_video_pipeline/Project3_Final_Image_Video.ipynb` là một pipeline riêng, bản Kaggle gốc `lamvo30ai/project3-full-pipeline` (version 8). Nó dùng **scene graph Visual Genome (GT) và scene graph RelTR** thay cho đồ thị của notebook `du-lieu-do-thi`. Nội dung:

- **TN1:** văn bản → ảnh trên VG + COCO, CLIP top-100 → điểm đường đi, cùng GraphSAGE và SCENIR.
- **TN2:** ảnh → caption trên ARO VG-Relation và SugarCrepe.
- **TN2b:** cạnh hình học trên COCO-spatial.
- **XAI:** fidelity, lật ảnh, validity, sparsity.
- **Video MSVD:** CLIP + PE-Core + đồ thị bằng chứng + Qwen2.5-VL.

**Lưu ý về số liệu:** file nộp không kèm output. Các số dưới đây là hằng `REF` ghi trong notebook (cell 2), là "số của các lần chạy riêng trước, để đối chiếu tái lập". Chưa có bản chạy lại trong repo này. Số của TN1b (GraphSAGE), TN1c (SCENIR), TN2b (COCO-spatial) và video MSVD chỉ có sau khi chạy notebook (`p3_out/RESULTS.md`).

**TN1 — văn bản → ảnh, full-gallery, Visual Genome (MRR):**

| CLIP | G-full | G-obj (chỉ đối tượng) | G-rand (cạnh ngẫu nhiên) | G-shuf (xáo) |
| --- | --- | --- | --- | --- |
| 0,7182 | 0,7212 | 0,7219 | 0,7174 | 0,7200 |

Đồ thị chỉ tăng khoảng +0,003 MRR. Bản chỉ dùng đối tượng ngang bản đầy đủ, và cạnh ngẫu nhiên / xáo gần bằng CLIP. Kết quả này cùng xu hướng với Flickr8k ở trên: khi CLIP đã mạnh và đồ thị không thêm thông tin mới, mức tăng rất nhỏ.

**TN2 — ảnh → caption đúng, độ chính xác (%):**

| Nhóm | CLIP | Chỉ đồ thị | CLIP + đồ thị |
| --- | --- | --- | --- |
| ARO – toàn bộ | 50,1 | 65,8 | **66,0** |
| ARO – quan hệ không gian | 49,7 | 66,3 | **66,7** |
| ARO – vị từ khác | 50,7 | 66,3 | 65,2 |
| SugarCrepe – thay quan hệ | 69,9 | 54,4 | **70,3** |
| SugarCrepe – đổi chỗ đối tượng | 63,5 | 49,7 | 61,3 |

Trên ARO, CLIP gần như đoán ngẫu nhiên (~50% với 2 lựa chọn), còn đồ thị RelTR tăng lên khoảng 66% (+16 điểm). Trên SugarCrepe, đồ thị một mình kém CLIP, và kết hợp chỉ giúp nhẹ ở nhóm thay quan hệ.

**XAI (ARO, toàn bộ):**
- Độ chính xác của đồ thị trên tập XAI: 85,7%.
- Lật ảnh trái–phải làm đồ thị đảo quyết định ở 88,1% mục.
- Xoá bằng chứng làm đổi quyết định ở 11,1% mục.
- Validity (cả hai đối tượng khớp GT, IoU ≥ 0,5): 54,9%.

**Notebook gộp:** `merged/Project3_Merged.ipynb` chạy cả ba phần trong một lần, metadata Kaggle ở `merged/kernel-metadata.json`.
- Phần A: `du-lieu-do-thi` (Flickr8k + Action Genome, ViT-L/14).
- Phần B: notebook ảnh + video ở trên.
- Phần C: Level 4(b) v2-graphfix, nằm ở mục 4u, xem mục dưới.

Thời gian khoảng 7–8 giờ trên T4 nếu output của phần C đã được Add Input. Nếu chưa, phần C phải chạy lại đầy đủ và tổng lên khoảng 10 giờ.

## Nhánh video MSVD / MSR-VTT — Level 4(b) v2-graphfix (`level4b_v2_graphfix/`)

Notebook `lamvo30ai/project03-level4b-v2-graphfix` (version 1), text → video. Nó dựng đồ thị v3 có vị từ (agent / object / svo / prep), mô hình TypedSAGE và QC-RGAT, tagger học từ caption train và cạnh ConceptNet. Nó cũng tạo cache đặc trưng frame CLIP ViT-L/14 mà notebook video v5 (mục 4v) dùng lại.

Số dưới đây đọc từ `level4b_v2_graphfix/results/graph_experiment_results.csv` và `graph_confirmatory_tests.csv`, là output thật của lần chạy trên Kaggle (khoảng 2,4 giờ).

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

Kết luận đầy đủ: `level4b_v2_graphfix/results/level4b_conclusions.md`.

**Cách chạy riêng phần B:** xem `image_video_pipeline/HUONG_DAN_CHAY_VI.md`. Cần GPU T4, Internet, 3 dataset VG / COCO và notebook nguồn `lamvo30ai/project03-level4b-v2-graphfix` cho cache video; tổng khoảng 3 giờ.

## Hạn chế và việc tiếp theo

- Mọi mức tăng cần gallery có chú thích của con người.
- Flickr8k: đồ thị vẫn thua CLIP ở bộ quan hệ khó; 16 truy vấn gián tiếp bị tụt.
- Nhánh khớp coi node như túi từ ⇒ kém nhạy với động từ hơn GNN.
- Phần 4 (MSVD → MSR-VTT) chưa chạy lại với các cải tiến mới (kết quả cũ: CLIP 0,505 → 0,514).
- Đang chạy: notebook gộp ba phần (Kaggle `hieutrinh123/project3-merged`) để có số chạy thật cho nhánh Visual Genome / ARO / SugarCrepe / MSVD.

## Nguồn

Nhánh Visual Genome / ARO / SugarCrepe: hằng `REF` trong `image_video_pipeline/Project3_Final_Image_Video.ipynb` (cell 2).

Output của notebook Kaggle `hieutrinh123/du-lieu-do-thi` (version 4, 7, 8, 9): `results/` (Flickr8k) và `output/action_genome/results/` (Action Genome) — `main_results_with_controls.csv`, `match_vs_clip.csv`, `match_paired.csv`, `relational_hard_negatives_match*.csv`, `counterfactual_verb_swap_match*.csv`, `paired_graph_vs_clip.csv`.
