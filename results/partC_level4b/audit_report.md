# Audit nghiệm thu — Project 3 · Level 4(b) text-to-video (tự sinh)

2026-10-31 13:18:10 | chính thức: True | CLIP ViT-L-14 | seeds [0, 1, 2, 3, 4] | epochs 15 | tập ngoài: OK | Kaggle=True

PASS ở đây là PASS của **kiểm thử / sản phẩm**, không phải kết luận khoa học. Kết luận khoa học nằm ở `level4b_conclusions.md`.

| id | yêu cầu | trạng thái | bằng chứng |
|---|---|---|---|
| A1 | Baseline CLIP đóng băng | **PASS** | B0=PASS ; G0_clip_only=DONE |
| A2 | Đồ thị từ nguồn hợp lệ (caption TRAIN + CLIP zero-shot trên frame), đúng schema | **PASS** | G2=PASS / C1=PASS / C9=PASS |
| A3 | Ít nhất một GNN đã học (GraphSAGE có kiểu cạnh = R3; QC-RGAT = G3) | **PASS** | V3REF=DONE, G3_v4_full=DONE |
| A4 | CLIP-only và CLIP + Graph so sánh trên cùng split, cùng ứng viên | **PASS** | D2=PASS ; G0_clip_only=DONE, G3_v4_full=DONE |
| A5 | Ablation quan hệ (G1, G7, G8 huấn luyện lại) + đồ thị ngẫu nhiên hợp lệ (G4, G5) | **PASS** | G1_contains_only=DONE, G4_pred_shuffled=DONE, G5_rewired=DONE, G7_no_role_edges=DONE, G8_no_svo_prep_edges=DONE ; G6=PASS / G7=PASS / AB=PASS |
| A7 | v2: tri thức ngoài ConceptNet (G9) + đối chứng rewired (G10) hợp lệ | **PASS** | G9_conceptnet=DONE, G10_conceptnet_rewired=DONE ; KG1=PASS |
| A8 | v2: tagger học từ caption TRAIN được đánh giá trên VAL (V2) và đối chứng zero-shot | **PASS** | VAL F1 học 0.487 vs zero-shot 0.324 |
| A6 | R@1, R@5, R@10, MRR; ≥ 3 seed cho thí nghiệm chính | **PASS** | 5 seed [0, 1, 2, 3, 4] ; D3=PASS |
| B1 | Schema node / cạnh rõ ràng | **PASS** | graph_schema_v3.json |
| B2 | Nguồn gốc cạnh được ghi (bảng cạnh + provenance) | **PASS** | graph_edges_v3.csv, provenance.csv |
| B3 | Không rò rỉ test | **PASS** | A5=PASS / C1=PASS / C1b=PASS / C2=PASS / C3=PASS / C8=PASS / C9=PASS / D1=PASS / D4=PASS |
| B4 | Audit cấu trúc (inventory cố định, schema, trọng số, self-loop / trùng / bậc) | **PASS** | G1=PASS / G2=PASS / G4=PASS / I2=PASS |
| B5 | Đối chứng ngẫu nhiên đúng thuộc tính bảo toàn đã công bố | **PASS** | G6=PASS / G7=PASS |
| B6 | Chất lượng ngữ nghĩa của cạnh được người xác minh | **NOT RUN** | chưa có nhãn người chấm ⇒ chưa xác nhận; chỉ có kiểm chứng tự động SILVER (spaCy) |
| C1 | Pipeline text-to-video thật (giải mã video, lấy mẫu frame, gộp, gallery) | **PASS** | A3=PASS / A4=PASS / B1=PASS / B3=PASS / B6=PASS / B7=PASS / B8=PASS |
| C2 | Encoder tiền huấn luyện đóng băng + lấy mẫu frame | **PASS** | B0=PASS / B6=PASS |
| C3 | Gallery, bộ truy vấn và metric xác định rõ (gồm bộ ≥ 20 truy vấn / nhóm) | **PASS** | B8=PASS / QS1=PASS |
| C4 | Xếp hạng lại bằng đồ thị + baseline, đánh giá trên tập CHƯA nhìn | **PASS** | D3=PASS ; tập ngoài: OK; kind=confirmatory |
| C5 | Giải thích lấy từ chính cơ chế xếp hạng | **PASS** | Q3=PASS / X1=PASS / X3=PASS / X5=PASS |
| C6 | Phân tích thành công / thất bại và giới hạn | **PASS** | K1=PASS |
| D1 | Đường đi / quan hệ giải thích là thành phần thật của phép tính | **PASS** | Q3=PASS / X3=PASS |
| D2 | Validity, sparsity, fidelity (điểm + thứ hạng) trên ≥ 10 truy vấn | **PASS** | X1=PASS / X5=PASS ; xai_rank_fidelity.csv |
| D3 | ≥ 5 thành công + ≥ 5 thất bại được phân tích | **PASS** | K1=PASS |
| D4 | Demo: nhập truy vấn → top-k + điểm + đường đi giải thích | **PASS** | DM1=PASS |
| E1 | Notebook chạy hết (hoặc báo rõ giới hạn) | **PASS** | chạy chính thức tới ô này |
| E2 | File môi trường + manifest | **PASS** | pip_freeze.txt, env_versions.json, experiment_manifest.json, data_inventory.json |
| E3 | Kết quả báo cáo khớp file nguồn | **PASS** | R1=PASS |

## Kết quả khoa học được báo cáo (FINDINGS)

| id | câu hỏi | kết quả | diễn giải |
|---|---|---|---|
| F-GRP-all | QC-RGAT (G3) vs CLIP-only trên nhóm 'all' của ext (mọi caption; so sánh PHỤ, không Holm) | n=20000: ΔMRR +0.0099 [CI95 theo video +0.0065, +0.0133] | đồ thị giúp nhóm này |
| F-GRP-direct | QC-RGAT (G3) vs CLIP-only trên nhóm 'direct' của ext (mọi caption; so sánh PHỤ, không Holm) | n=11517: ΔMRR +0.0102 [CI95 theo video +0.0057, +0.0146] | đồ thị giúp nhóm này |
| F-GRP-indirect | QC-RGAT (G3) vs CLIP-only trên nhóm 'indirect' của ext (mọi caption; so sánh PHỤ, không Holm) | n=4089: ΔMRR +0.0117 [CI95 theo video +0.0075, +0.0158] | đồ thị giúp nhóm này |
| F-QS-indirect | G3 vs CLIP-only trên 20 truy vấn GIÁN TIẾP đăng ký trước (ext) | MRR 0.5239 → 0.5438 (Δ +0.0200, CI95 [-0.0175, +0.0602]) | n nhỏ ⇒ CI rộng; chỉ là bằng chứng bổ sung cho kết quả trên mọi caption |
| F-X4 | xoá các đường đi QUAN HỆ của giải thích (v4) có đổi thứ hạng video đúng? | 20 truy vấn: hạng đổi 15% (xoá ngẫu nhiên 5%); Δ điểm đồ thị -0.0171 vs ngẫu nhiên +0.0029 | xoá quan hệ của giải thích có làm đổi thứ hạng ở một phần truy vấn |
| F-X6 | xoá giải thích ĐẦY ĐỦ (contains + quan hệ) có làm video đúng tụt hạng nhiều hơn xoá ngẫu nhiên cùng số cạnh? | 20 truy vấn: tụt hạng 50% vs 0%; hạng TB 5.00 → 7.05 (ngẫu nhiên 4.60); Δ điểm cuối -0.200 vs +0.136; top-1 đổi 5% | giải thích chỉ ra đúng phần đồ thị quyết định thứ hạng (mạnh hơn ngẫu nhiên) |

## Thí nghiệm NOT RUN

- V7 G11_zeroshot_tags: V2 không chọn tagger học ⇒ G3 đã dùng nhãn zero-shot
