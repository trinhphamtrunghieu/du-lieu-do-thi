# Kết quả pipeline Project 3

Chế độ: REAL · RelTR ckpt sha256 2b3601e5cc5d835d9538d78b698dd7b3f4f1646bdb0e5c459d2cbe959d5906b1 | CLIP ViT-L-14/laion2b_s32b_b82k | open_clip 3.2.0 | torch 2.11.0+cu130 | cuda · 10585s

## TN1 — Văn bản → ảnh, full-gallery (P-CHECK, trung bình ± std qua 3 seed caption)

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 61.1 ± 2.7 | 85.0 | 92.6 | 0.7184 ± 0.0223 |
| G-full | 61.4 ± 2.5 | 85.3 | 92.9 | 0.7214 ± 0.0206 |
| G-obj | 61.6 ± 2.5 | 85.3 | 92.8 | 0.7221 ± 0.0209 |
| G-rand | 61.0 ± 2.8 | 85.0 | 92.6 | 0.7175 ± 0.0230 |
| G-shuf | 61.3 ± 2.2 | 85.4 | 92.8 | 0.7202 ± 0.0191 |

Quyết định đăng ký trước: **NO-GO**

## TN1b — GraphSAGE (GNN đã học) trong cùng pipeline văn bản → ảnh (P-CHECK, 3 seed)

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 61.1 ± 2.7 | 85.0 | 92.6 | 0.7184 ± 0.0223 |
| SAGE | 61.1 ± 2.6 | 85.0 | 92.3 | 0.7175 ± 0.0224 |
| SAGE-rand | 61.1 ± 2.6 | 85.1 | 92.6 | 0.7180 ± 0.0218 |
| SAGE-0 | 60.9 ± 2.1 | 85.2 | 92.3 | 0.7169 ± 0.0185 |

Quyết định đăng ký trước: **{'G1 SAGE-CLIP': 'KHÔNG DƯƠNG', 'G2 SAGE-rand': 'KHÔNG DƯƠNG', 'G3 SAGE-noedge': 'KHÔNG DƯƠNG'}** (SAGE-rand = cạnh ngẫu nhiên, SAGE-0 = bỏ cạnh)

## TN1c — SCENIR (ICML 2025) trong cùng pipeline văn bản → ảnh (P-CHECK, 3 seed; đặc trưng đỉnh: sentence-transformers/all-mpnet-base-v2)

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 61.1 ± 2.7 | 85.0 | 92.6 | 0.7184 ± 0.0223 |
| SCENIR | 61.4 ± 2.3 | 85.0 | 92.4 | 0.7191 ± 0.0210 |
| SCENIR-rand | 61.2 ± 2.7 | 85.0 | 92.5 | 0.7188 ± 0.0224 |
| SCENIR-noedge | 61.1 ± 2.7 | 85.0 | 92.6 | 0.7184 ± 0.0223 |

Quyết định đăng ký trước: **{'S1 SCENIR-CLIP': 'KHÔNG DƯƠNG', 'S2 SCENIR-rand': 'KHÔNG DƯƠNG', 'S3 SCENIR-noedge': 'KHÔNG DƯƠNG'}**

## TN2 — Benchmark kết hợp (độ chính xác CHECK %, trung bình ± std qua 3 seed chia)

| Nhóm | CLIP | Chỉ đồ thị | CLIP + đồ thị | Đồ thị mục khác | Bỏ vị trí |
|---|---|---|---|---|---|
| ARO_all | 50.1 ± 0.5 | 65.8 ± 1.3 | 66.1 ± 1.6 | 53.3 ± 0.1 | 50.0 ± 0.0 |
| ARO_geo | 49.7 ± 0.6 | 66.3 ± 1.4 | 66.7 ± 1.8 | 50.9 ± 0.5 | 50.0 ± 0.0 |
| ARO_pred | 50.5 ± 3.0 | 66.3 ± 1.5 | 65.2 ± 2.0 | 60.3 ± 1.2 | — |
| ARO_other | 58.9 ± 6.3 | — | — | — | — |
| ARO_all_excl_v3bench | 50.2 ± 0.7 | 65.7 ± 1.2 | 65.8 ± 1.5 | 53.5 ± 0.4 | 50.0 ± 0.0 |
| SC_replace_rel | 69.9 ± 1.2 | 54.4 ± 0.9 | 70.3 ± 1.2 | 50.6 ± 1.1 | — |
| SC_swap_obj | 63.5 ± 1.6 | 49.7 ± 1.2 | 61.3 ± 1.1 | 46.0 ± 4.2 | — |

Quyết định đăng ký trước: **{'ARO_all': 'DƯƠNG', 'SC_replace_rel': 'KHÔNG DƯƠNG'}**

## TN2b — Cấp 4 (a): cạnh hình học trên COCO-spatial (292 mục sau lọc, 132 mục đồ thị chấm được; loại {'vg_not_VAL': 148})

| Nhóm | CLIP | Chỉ đồ thị | CLIP + đồ thị | Đồ thị mục khác | Bỏ vị trí |
|---|---|---|---|---|---|
| COCOSP_all | 46.8 ± 4.0 | 69.8 ± 0.3 | 63.9 ± 4.7 | 51.3 ± 2.2 | 50.0 ± 0.0 |
| COCOSP_leftright | 50.0 ± 5.3 | 71.7 ± 1.6 | 69.5 ± 4.6 | 50.4 ± 4.4 | 50.0 ± 0.0 |
| COCOSP_abovebelow | 41.1 ± 4.0 | 66.6 ± 2.8 | 51.8 ± 6.7 | 52.6 ± 2.4 | 50.0 ± 0.0 |

Quyết định đăng ký trước: **{'A1 fusion-CLIP': 'DƯƠNG', 'A2 graph-nogeo': 'DƯƠNG', 'A3 graph-rand': 'DƯƠNG'}** · Lật ảnh: đồ thị đảo 87.9%, CLIP đảo 8.3% (n=132)

## XAI (ARO, CHECK seed 0)

| Nhóm | n | Lật ảnh: đồ thị đảo / CLIP đảo | Xoá bằng chứng: đổi / ngẫu nhiên | Validity cả hai / ít nhất một | Sufficiency | Sparsity: kích thước / độ tập trung |
|---|---|---|---|---|---|---|
| all | 622 | 87.9 / 4.4 | 12.3 / 0.0 | 54.2 / 92.1 | 100.0 (theo cấu trúc) | 0.5% (thiết kế) / 0.18 (trung vị 0.01) |
| geo | 472 | 87.9 / 4.4 | 13.8 / 0.0 | 49.7 / 90.7 | 100.0 (theo cấu trúc) | 0.5% (thiết kế) / 0.01 (trung vị 0.01) |
| pred | 150 | — | 7.4 / 0.0 | 67.4 / 96.3 | 100.0 (theo cấu trúc) | 0.5% (thiết kế) / 0.69 (trung vị 0.71) |

## Đối chiếu tái lập với các lần chạy riêng trước đó

| Chỉ số | Lần trước | Pipeline này | Chênh |
|---|---|---|---|
| TN1 MRR CLIP | 0.7182 | 0.7184 | +0.0001 |
| TN1 MRR G-full | 0.7212 | 0.7214 | +0.0003 |
| TN1 MRR G-obj | 0.7219 | 0.7221 | +0.0002 |
| TN1 MRR G-rand | 0.7174 | 0.7175 | +0.0001 |
| TN1 MRR G-shuf | 0.7200 | 0.7202 | +0.0002 |
| TN2 ARO_all CLIP | 0.5014 | 0.5012 | -0.0002 |
| TN2 ARO_all graph_only[dg] | 0.6585 | 0.6585 | +0.0000 |
| TN2 ARO_all fusion[dg] | 0.6603 | 0.6605 | +0.0003 |
| TN2 ARO_geo CLIP | 0.4971 | 0.4965 | -0.0006 |
| TN2 ARO_geo graph_only[dg] | 0.6627 | 0.6627 | +0.0000 |
| TN2 ARO_geo fusion[dg] | 0.6672 | 0.6666 | -0.0006 |
| TN2 ARO_pred CLIP | 0.5072 | 0.5055 | -0.0017 |
| TN2 ARO_pred graph_only[dg] | 0.6633 | 0.6633 | +0.0000 |
| TN2 ARO_pred fusion[dg] | 0.6520 | 0.6520 | +0.0000 |
| TN2 SC_replace_rel CLIP | 0.6988 | 0.6988 | +0.0000 |
| TN2 SC_replace_rel graph_only[dg] | 0.5445 | 0.5445 | +0.0000 |
| TN2 SC_replace_rel fusion[dg] | 0.7032 | 0.7032 | +0.0000 |
| TN2 SC_swap_obj CLIP | 0.6353 | 0.6353 | +0.0000 |
| TN2 SC_swap_obj graph_only[dg] | 0.4966 | 0.4966 | +0.0000 |
| TN2 SC_swap_obj fusion[dg] | 0.6133 | 0.6133 | +0.0000 |
| XAI all graph_acc | 0.8569 | 0.8617 | +0.0048 |
| XAI all fidelity_del_changed | 0.1111 | 0.1225 | +0.0114 |
| XAI all validity_both_iou50 | 0.5492 | 0.5424 | -0.0069 |
| XAI all counterfactual_graph_reversed | 0.8814 | 0.8792 | -0.0021 |
| XAI geo graph_acc | 0.8411 | 0.8475 | +0.0064 |
| XAI geo fidelity_del_changed | 0.1250 | 0.1379 | +0.0129 |
| XAI geo validity_both_iou50 | 0.5064 | 0.4975 | -0.0089 |
| XAI geo counterfactual_graph_reversed | 0.8814 | 0.8792 | -0.0021 |
| XAI pred graph_acc | 0.9067 | 0.9067 | +0.0000 |
| XAI pred fidelity_del_changed | 0.0676 | 0.0743 | +0.0068 |
| XAI pred validity_both_iou50 | 0.6741 | 0.6741 | +0.0000 |

## So sánh các cách tạo cạnh

| Thí nghiệm | Cách tạo cạnh / biến thể | Giá trị (trung bình ± std, 3 seed) |
|---|---|---|
| TN1 | Không đồ thị (CLIP) | 0.7184 ± 0.0223 |
| TN1 | Chỉ đỉnh đối tượng (G-obj) | 0.7221 ± 0.0209 |
| TN1 | Bộ ba + cặp + đối tượng (G-full) | 0.7214 ± 0.0206 |
| TN1 | G-full, vị từ hoán vị (G-shuf) | 0.7202 ± 0.0191 |
| TN1 | G-full, đồ thị ảnh khác (G-rand) | 0.7175 ± 0.0230 |
| TN1b | GraphSAGE: ảnh – lớp – bộ ba | 0.7175 ± 0.0224 |
| TN1b | GraphSAGE, cạnh ngẫu nhiên | 0.7180 ± 0.0218 |
| TN1b | GraphSAGE, bỏ cạnh | 0.7169 ± 0.0185 |
| TN1c | SCENIR: scene graph lớp | 0.7191 ± 0.0210 |
| TN1c | SCENIR, cạnh ngẫu nhiên | 0.7188 ± 0.0224 |
| TN1c | SCENIR, bỏ cạnh | 0.7184 ± 0.0223 |
| ARO trái/phải | Cạnh vị từ / hình học (RelTR) | 66.3% ± 1.4 |
| ARO trái/phải | Đồ thị của mục khác | 50.9% ± 0.5 |
| ARO trái/phải | Bỏ vị trí (cạnh hình học = hằng) | 50.0% ± 0.0 |
| ARO vị từ | Cạnh vị từ / hình học (RelTR) | 66.3% ± 1.5 |
| ARO vị từ | Đồ thị của mục khác | 60.3% ± 1.2 |
| ARO vị từ | Vị từ hoán vị | 52.9% ± 2.6 |
| COCO-spatial | Cạnh vị từ / hình học (RelTR) | 69.8% ± 0.3 |
| COCO-spatial | Đồ thị của mục khác | 51.3% ± 2.2 |
| COCO-spatial | Bỏ vị trí (cạnh hình học = hằng) | 50.0% ± 0.0 |

## Cấp 4 (b) — Video MSVD (notebook Level4B v5, trạng thái: ok, 145.7 phút)

Lần nhìn test MSVD: **lần thứ 8 ⇒ THĂM DÒ** — số test chỉ là thăm dò, không phải xác nhận độc lập.

| Mô hình | Tập | R@1 | R@5 | R@10 | MRR | n truy vấn |
|---|---|---|---|---|---|---|
| M0_clip | test_all | 47.5 | 73.7 | 82.0 | 0.5931 | 27763 |
| G1_clip_graph | test_all | 49.0 | 76.2 | 84.2 | 0.6109 | 27763 |
| M1_clip_pe_flat | test_all | 59.2 | 83.0 | 88.4 | 0.6982 | 27763 |
| M2_clip_pe_hub | test_all | 59.8 | 83.3 | 88.6 | 0.7030 | 27763 |
| M3_clip_pe_bridge | test_all | 59.4 | 83.2 | 88.5 | 0.7001 | 27763 |
| M4_full_graph | test_all | 60.2 | 83.9 | 88.8 | 0.7069 | 27763 |
| M4_ctrl_hub_random | test_all | 59.1 | 83.2 | 88.6 | 0.6987 | 27763 |
| M4_ctrl_bridge_shuffled | test_all | 59.6 | 83.3 | 88.4 | 0.7010 | 27763 |
| M4_ctrl_both | test_all | 59.2 | 82.9 | 88.3 | 0.6979 | 27763 |
| G1_ctrl_both | test_all | 47.7 | 74.2 | 82.6 | 0.5963 | 27763 |
| M0_clip | test_vlm_subset | 45.1 | 69.9 | 77.8 | 0.5633 | 670 |
| M1_clip_pe_flat | test_vlm_subset | 56.3 | 78.4 | 83.6 | 0.6590 | 670 |
| M4_full_graph | test_vlm_subset | 55.8 | 78.8 | 84.3 | 0.6603 | 670 |
| G1_clip_graph | test_vlm_subset | 45.5 | 72.5 | 79.3 | 0.5736 | 670 |
| M5_graph_plus_vlm | test_vlm_subset | 56.7 | 80.0 | 84.3 | 0.6716 | 670 |
| M5_ctrl_vlm_shuffled | test_vlm_subset | 40.3 | 75.8 | 84.3 | 0.5543 | 670 |
| M5b_clip_topM_vlm | test_vlm_subset | 50.6 | 74.0 | 77.8 | 0.6107 | 670 |

| So sánh | a vs b | ΔMRR [CI 95%] | p (Holm) |
|---|---|---|---|
| T1_total_full_vs_clip | M4_full_graph vs M0_clip | +0.1138 [+0.1012, +0.1273] | 0.0 |
| T2_graph_on_clip_only | G1_clip_graph vs M0_clip | +0.0178 [+0.0095, +0.0262] | 0.0 |
| T3_structure_over_flat_clip_pe | M4_full_graph vs M1_clip_pe_flat | +0.0087 [+0.0039, +0.0136] | 0.0 |
| T4_structure_vs_controls | M4_full_graph vs M4_ctrl_both | +0.0090 [+0.0027, +0.0154] | 0.006 |
| T4b_clip_graph_vs_controls | G1_clip_graph vs G1_ctrl_both | +0.0145 [+0.0057, +0.0235] | 0.0015 |
| T5_vlm_node_over_graph | M5_graph_plus_vlm vs M4_full_graph | +0.0113 [-0.0031, +0.0252] | 0.0565 |
| T6_full_vs_clip_subset | M5_graph_plus_vlm vs M0_clip | +0.1082 [+0.0854, +0.1309] | 0.0 |
| T7_vlm_edges_vs_shuffled | M5_graph_plus_vlm vs M5_ctrl_vlm_shuffled | +0.1172 [+0.0946, +0.1395] | 0.0 |
| T8_full_vs_clip_topM_vlm | M5_graph_plus_vlm vs M5b_clip_topM_vlm | +0.0609 [+0.0417, +0.0796] | 0.0 |

Fidelity (video): {'n': 60, 'worse_top3': 0.5666666666666667, 'worse_random3': 0.25, 'mean_drank_top3': 1.1333333333333333, 'mean_drank_random3': 0.2833333333333333}

## Đối chiếu yêu cầu đề cương

| Cấp | Yêu cầu | Trạng thái | Bằng chứng (lần chạy này) |
|---|---|---|---|
| Cấp 1 | Baseline CLIP đóng băng trên ≥ 1 tập công khai | ✅ | CLIP ViT-L-14; TN1 MRR 0.7184 |
| Cấp 1 | Đồ thị từ dữ liệu có sẵn (scene graph VG) | ✅ | Scene graph GT (TN1), scene graph RelTR (TN2, 2b) |
| Cấp 1 | ≥ 1 thuật toán đã học (GraphSAGE) | ✅ | TN1b: SAGE 0.7175 vs CLIP 0.7184 (không dương) |
| Cấp 1 | Xếp hạng lại α·cos + (1−α)·graph, so CLIP vs CLIP+Graph cùng split | ✅ | TN1 NO-GO; ARO DƯƠNG |
| Cấp 1 | Ablation bỏ cạnh / cạnh ngẫu nhiên | ✅ | ARO: mục khác 53.3%, bỏ vị trí 50.0% |
| Cấp 1 | Hiển thị quan hệ/đường đi giải thích | ✅ | Demo search() + hình XAI |
| Cấp 1 | 5 truy vấn thành công + 5 thất bại | ✅ | 10 ca văn bản → ảnh (case_*.png) + 10 ca ARO (xai_ex*.png); phân tích: CASE_ANALYSIS_LEVEL1_VI.md |
| Cấp 1 | R@1/5/10, MRR, ≥ 3 seed, α chọn trên validation | ✅ | Mọi thí nghiệm: 3 seed, α trên TUNE |
| Cấp 2 | ≥ 2 tập công khai | ✅ | Visual Genome + COCO, ARO, SugarCrepe, COCO-spatial |
| Cấp 2 | 1 phương pháp mới từ 2024, cùng pipeline/split | ✅ | SCENIR (ICML 2025) TN1c: 0.7191 vs CLIP (không dương) |
| Cấp 2 | Bộ ≥ 20 truy vấn có nhóm gián tiếp, báo cáo tách nhóm | ⏸ | Hoãn (cần MMKG) — ngoài phạm vi notebook này |
| Cấp 2 | Ablation xáo trộn / đồ thị ngẫu nhiên; so sánh cách tạo cạnh; độ nhạy α | ✅ | Bảng so sánh cạnh + alpha_sensitivity.png |
| Cấp 3 | Fidelity định lượng ≥ 10 truy vấn | ✅ | Lật ảnh 87.9% vs CLIP 4.4%; xoá 12.3% vs ngẫu nhiên 0.0% (n=622) |
| Cấp 3 | Validity | ✅ | Khớp GT IoU≥0.5: cả hai 54.2%, ít nhất một 92.1% |
| Cấp 3 | Sparsity | ✅ | Kích thước 0.5% (thiết kế); độ tập trung trung bình 0.18 |
| Cấp 3 | MMKG tự xây + data card | ⏸ | Hoãn — ngoài phạm vi notebook này |
| Cấp 4 (a) | Cải tiến của nhóm + giả thuyết + ablation + ≥ 2 tập công khai | ✅ | Cạnh hình học: ARO trái/phải 49.7→66.7%; COCO-spatial {'A1 fusion-CLIP': 'DƯƠNG', 'A2 graph-nogeo': 'DƯƠNG', 'A3 graph-rand': 'DƯƠNG'} |
| Cấp 4 (b) | Mở rộng video (text-to-video) | ✅ | MSVD test: CLIP 0.5931 → đồ thị đầy đủ 0.7069 (lần nhìn test: lần thứ 8 ⇒ THĂM DÒ) |
| Mở rộng | Chiều ảnh → văn bản (nhánh B) | ✅ | TN2, TN2b |
| Nộp | Mã tạo split / file split cố định | ✅ | splits/ (CSV + SPLIT_RULES.json) |
| Nộp | Notebook nhập truy vấn → top-k + giải thích | ✅ | Hàm search() (mục 10) |

> **Ghi chú thêm 17/11/2026 (không phải output của lần chạy):** hai dòng "Hoãn" ở trên là bảng tự kiểm tra cố định của notebook phần B. Hai mục này đã được làm ở chỗ khác: bộ ≥ 20 truy vấn gián tiếp ở phần A (`results/project3-merged-v1/partA_*/eval_queries_20.json`) và MMKG tự xây + data card ở `mmkg/` (Kaggle `hieutrinh123/project3-mmkg-vn` v4). Xem `REQUIREMENTS_CHECK.md`.