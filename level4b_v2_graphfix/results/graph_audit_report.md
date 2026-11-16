# Audit đồ thị — Level 4(b): QC-RGAT + đối chứng cho text→video (MSVD → MSR-VTT)

2026-10-31 13:18:10 | dataset=msvd | dev_mode=False | Kaggle=True | Tesla T4 | CLIP ViT-L-14 | seeds [0, 1, 2, 3, 4] | epochs 15 | tập ngoài: OK

## Tiêu chí nghiệm thu

| # | Tiêu chí | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1 | Quan hệ được DÙNG có điều kiện theo truy vấn (vị từ, vai, loại cạnh tác động attention) | **PASS** | Q1=PASS | Q2=PASS |
| 2 | Điểm = Σ số hạng giải thích (phân rã chính xác); xoá cạnh ⇒ đúng phần đóng góp | **PASS** | Q3=PASS | X1=PASS | X3=PASS |
| 3 | Có đối chứng tách quan hệ / vị từ / topology / điều kiện truy vấn (G1, G2, G4, G5, G6 + audit) | **PASS** | G6=PASS | G7=PASS |
| 4 | Ablation công bằng (inventory cố định, cùng ứng viên, cùng giao thức huấn luyện) | **PASS** | G1=PASS | D2=PASS |
| 5 | Không dùng test đã xem để chọn mô hình | **PASS** | D1=PASS | D4=PASS | C3=PASS |
| 6 | Đủ bằng chứng xác định tốt hơn / tương đương / kém hơn (tập xác nhận độc lập, Holm) | **PASS** | D3=PASS |
| 7 | Mọi kết luận có nguồn số liệu kiểm chứng | **NOT RUN** | không chạy |

## File bàn giao

| File | Có dữ liệu |
|---|---|
| graph_schema_v3.json | PASS |
| graph_edges_v3.csv | PASS |
| relation_quality_audit.csv | PASS |
| graph_randomization_audit.csv | PASS |
| graph_ablation_results.csv | PASS |
| graph_relation_analysis.csv | PASS |
| graph_xai_fidelity.csv | PASS |
| frozen_config.json | PASS |
| experiment_manifest.json | PASS |
| graph_conclusion.md | PASS |

## Quyết định theo stage (chỉ VAL; incumbent = phương án đơn giản hơn)

| stage | incumbent | ứng viên | Δ val MRR | p Holm | chấp nhận |
|---|---|---|---|---|---|
| V1 | E10_v4_T1 | E11_v4_T2 | +0.0019 | 0.092 | False |
| V2 | E10_v4_T1 | E20_tags_probe | -0.0002 | 0.551 | False |

## Thí nghiệm NOT RUN

- V7 G11_zeroshot_tags: V2 không chọn tagger học ⇒ G3 đã dùng nhãn zero-shot

## Kết luận

- ⚠ Khai báo: test MSVD đã được nhìn ở v1–v4 và Level 4(b) v1 ⇒ số test MSVD ở đây là THĂM DÒ (lần nhìn thứ 6). Xác nhận độc lập: OK (MSR-VTT fresh1k-C: 1000 video train-9k (ngoài 1k-A, dev, fresh1k, fresh1k-B; seed 20261031) — CHƯA dùng ở v1–v4 và Level 4(b) v1).
- Mô hình v4 = QC-RGAT (64 chiều, 1 lớp), đồ thị = cấu hình cuối của v3 ({"taxonomy": "none", "mask": "no_related", "rel_policy": "all", "contains": "binary"}); k = 50; chọn số lớp trên VAL: E10_v4_T1.
- Chất lượng quan hệ — kiểm chứng TỰ ĐỘNG (spaCy 3.8.16 / en_core_web_sm 3.8.0): tỉ lệ quan hệ heuristic được parser độc lập xác nhận — agent_of 0.89 [0.88, 0.89]; object_of 0.60 [0.59, 0.61]; prep 0.25 [0.24, 0.26]; svo 0.80 [0.79, 0.80] (23s). SILVER — không phải precision do người chấm.
- v2: tagger học 0.487 vs zero-shot 0.324 (VAL F1); G3 dùng nhãn 'base'; ConceptNet: OK — 8990 cạnh ConceptNet giữa khoá từ vựng TRAIN
- H1_graph_vs_clip — v4 (G3) vs CLIP-only (G0):
-      VAL (phát triển): G3_v4_full vs G0_clip_only (VAL): ΔMRR +0.0110 [+0.0031, +0.0199]
-      test MSVD (THĂM DÒ): G3_v4_full vs G0_clip_only: ΔMRR +0.0151 (CI95 theo video [+0.0093, +0.0209], p Holm 0.000) ⇒ TỐT HƠN có ý nghĩa — ỦNG HỘ giả thuyết
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G0_clip_only: ΔMRR +0.0099 (CI95 theo video [+0.0065, +0.0133], p Holm 0.000) ⇒ TỐT HƠN có ý nghĩa — ỦNG HỘ giả thuyết
- H2_relations_vs_contains_only — Quan hệ vs chỉ contains (G3 vs G1):
-      VAL (phát triển): G3_v4_full vs G1_contains_only (VAL): ΔMRR -0.0011 [-0.0041, +0.0018]
-      test MSVD (THĂM DÒ): G3_v4_full vs G1_contains_only: ΔMRR -0.0016 (CI95 theo video [-0.0041, +0.0010], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G1_contains_only: ΔMRR -0.0017 (CI95 theo video [-0.0033, -0.0002], p Holm 1.000) ⇒ KÉM HƠN có ý nghĩa
- H3_predicate_vs_family_only — Khớp vị từ vs không vị từ (G3 vs G2):
-      VAL (phát triển): G3_v4_full vs G2_no_predicate (VAL): ΔMRR -0.0001 [-0.0016, +0.0013]
-      test MSVD (THĂM DÒ): G3_v4_full vs G2_no_predicate: ΔMRR -0.0008 (CI95 theo video [-0.0017, +0.0001], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G2_no_predicate: ΔMRR +0.0000 (CI95 theo video [-0.0006, +0.0006], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H4_true_vs_shuffled_predicates — Vị từ đúng cạnh vs hoán vị (G3 vs G4):
-      VAL (phát triển): G3_v4_full vs G4_pred_shuffled (VAL): ΔMRR -0.0004 [-0.0014, +0.0006]
-      test MSVD (THĂM DÒ): G3_v4_full vs G4_pred_shuffled: ΔMRR -0.0003 (CI95 theo video [-0.0009, +0.0002], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G4_pred_shuffled: ΔMRR +0.0001 (CI95 theo video [-0.0003, +0.0005], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H5_true_vs_rewired_topology — Topology thật vs rewiring có ràng buộc (G3 vs G5):
-      VAL (phát triển): G3_v4_full vs G5_rewired (VAL): ΔMRR -0.0032 [-0.0069, -0.0001]
-      test MSVD (THĂM DÒ): G3_v4_full vs G5_rewired: ΔMRR -0.0009 (CI95 theo video [-0.0036, +0.0016], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G5_rewired: ΔMRR -0.0014 (CI95 theo video [-0.0029, +0.0001], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H6_query_conditioned_vs_static — Attention theo truy vấn vs tĩnh (G3 vs G6):
-      VAL (phát triển): G3_v4_full vs G6_static_attention (VAL): ΔMRR -0.0013 [-0.0028, +0.0001]
-      test MSVD (THĂM DÒ): G3_v4_full vs G6_static_attention: ΔMRR +0.0001 (CI95 theo video [-0.0009, +0.0012], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G6_static_attention: ΔMRR +0.0002 (CI95 theo video [-0.0004, +0.0009], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H7_v4_vs_v3 — QC-RGAT vs GraphSAGE có kiểu cạnh (G3 vs R3):
-      VAL (phát triển): G3_v4_full vs V3REF (VAL): ΔMRR -0.0054 [-0.0137, +0.0020]
-      test MSVD (THĂM DÒ): G3_v4_full vs V3REF: ΔMRR +0.0064 (CI95 theo video [+0.0020, +0.0108], p Holm 0.013) ⇒ TỐT HƠN có ý nghĩa — ỦNG HỘ giả thuyết
-      tập ngoài (XÁC NHẬN): G3_v4_full vs V3REF: ΔMRR +0.0032 (CI95 theo video [+0.0004, +0.0059], p Holm 0.141) ⇒ TỐT HƠN có ý nghĩa
- H8_role_edges — Họ cạnh vai agent_of / object_of (G3 vs G7 huấn luyện lại):
-      VAL (phát triển): G3_v4_full vs G7_no_role_edges (VAL): ΔMRR -0.0021 [-0.0044, +0.0002]
-      test MSVD (THĂM DÒ): G3_v4_full vs G7_no_role_edges: ΔMRR -0.0002 (CI95 theo video [-0.0020, +0.0015], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G7_no_role_edges: ΔMRR -0.0008 (CI95 theo video [-0.0018, +0.0002], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H9_svo_prep_edges — Họ cạnh svo / prep (G3 vs G8 huấn luyện lại):
-      VAL (phát triển): G3_v4_full vs G8_no_svo_prep_edges (VAL): ΔMRR -0.0025 [-0.0046, -0.0006]
-      test MSVD (THĂM DÒ): G3_v4_full vs G8_no_svo_prep_edges: ΔMRR -0.0007 (CI95 theo video [-0.0025, +0.0010], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G3_v4_full vs G8_no_svo_prep_edges: ΔMRR -0.0007 (CI95 theo video [-0.0017, +0.0004], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H10_kg_vs_contains_only — ConceptNet: G9 (đồ thị + KG) vs chỉ-contains G1:
-      VAL (phát triển): G9_conceptnet vs G1_contains_only (VAL): ΔMRR +0.0001 [-0.0032, +0.0033]
-      test MSVD (THĂM DÒ): G9_conceptnet vs G1_contains_only: ΔMRR -0.0024 (CI95 theo video [-0.0051, +0.0002], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G9_conceptnet vs G1_contains_only: ΔMRR -0.0014 (CI95 theo video [-0.0029, +0.0001], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H11_kg_true_vs_rewired — ConceptNet: topology thật vs rewired (G9 vs G10):
-      VAL (phát triển): G9_conceptnet vs G10_conceptnet_rewired (VAL): ΔMRR -0.0011 [-0.0041, +0.0019]
-      test MSVD (THĂM DÒ): G9_conceptnet vs G10_conceptnet_rewired: ΔMRR -0.0025 (CI95 theo video [-0.0053, +0.0002], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G9_conceptnet vs G10_conceptnet_rewired: ΔMRR -0.0012 (CI95 theo video [-0.0027, +0.0003], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H12_kg_adds_to_graph — ConceptNet thêm vào đồ thị hiện tại (G9 vs G3):
-      VAL (phát triển): G9_conceptnet vs G3_v4_full (VAL): ΔMRR +0.0012 [-0.0009, +0.0034]
-      test MSVD (THĂM DÒ): G9_conceptnet vs G3_v4_full: ΔMRR -0.0009 (CI95 theo video [-0.0022, +0.0005], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
-      tập ngoài (XÁC NHẬN): G9_conceptnet vs G3_v4_full: ΔMRR +0.0003 (CI95 theo video [-0.0005, +0.0012], p Holm 1.000) ⇒ không khác biệt có ý nghĩa
- H13_learned_vs_zeroshot_tags — Nhãn video học từ caption train vs zero-shot (G3 vs G11):
-      VAL (phát triển): NOT RUN
-      test MSVD (THĂM DÒ): NOT RUN — G11_zeroshot_tags: V2 không chọn tagger học ⇒ G3 đã dùng nhãn zero-shot
-      tập ngoài (XÁC NHẬN): NOT RUN — G11_zeroshot_tags: V2 không chọn tagger học ⇒ G3 đã dùng nhãn zero-shot
- Đối chứng (dưới cấu hình G3): rewiring hợp lệ: True; hoán vị vị từ hợp lệ: True. Đối chứng không hợp lệ ⇒ KHÔNG rút kết luận tương ứng.
- Phân tích quan hệ (val): bỏ mọi cạnh quan hệ lúc suy luận: MRR 0.8204 → 0.8232; attention cạnh có vị từ khớp truy vấn 0.116 vs không khớp 0.049.
- Phân tích quan hệ (ext): bỏ mọi cạnh quan hệ lúc suy luận: MRR 0.5196 → 0.5192; attention cạnh có vị từ khớp truy vấn 0.130 vs không khớp 0.048.
- Giải thích (G3_v4_full, ext, 20 truy vấn): hợp lệ 1.00, vị từ đúng cạnh 1.00, Σ số hạng = điểm 1.00, xoá cạnh khớp phân rã 1.00; Δ điểm khi xoá giải thích -0.0171 vs ngẫu nhiên +0.0029; hạng đổi 15% (ngẫu nhiên 5%); phần điểm từ quan hệ -1.8%; Jaccard theo seed 0.27.
- KẾT LUẬN: Quan hệ CHƯA được xác nhận đầy đủ: trên tập xác nhận, v4 không vượt có ý nghĩa mọi đối chứng quan hệ (ủng hộ: ['H1_graph_vs_clip']). Cơ chế dùng quan hệ có thật (P/Q/X-test) nhưng tác động lên MRR không đủ để kết luận.
- Giới hạn: nhãn video từ CLIP zero-shot (F1 explore 0.296, confirm 0.236); quan hệ là thống kê cấp LỚP từ caption train (không có quan hệ cấp video cho video mới); parser quan hệ heuristic; val chỉ 100 video; PASS của kiểm thử phần mềm không xác nhận giả thuyết khoa học.

## Mọi kiểm tra

| id | kiểm tra | trạng thái |
|---|---|---|
| A1 | số video / caption theo split khớp dataset card MSVD | PASS |
| A1b | cột 'video' của HF chỉ là tên file .avi (video thật lấy từ zip/thư mục) | PASS |
| A2 | mỗi video_id có đúng một file video (không thiếu file) | PASS |
| A6 | tập ngoài: đủ file video; fresh1k / fresh1k-B = đúng CFG.fresh_n video, rời 1k-A, rời 200 video dev (fresh1k-B rời cả fresh1k); 1kA = đúng 1000 video test 1k-A; dev = rời hẳn 1k-A | PASS |
| A3 | giải mã được video mẫu của TỪNG split (3 video/split, đếm frame bằng grab) | PASS |
| A5 | split THEO VIDEO, không video nào ở hai split; manifest đã lưu | PASS |
| A4 | TOÀN BỘ video giải mã được và lấy đủ F_max frame | PASS |
| B0 | CLIP đóng băng (eval, không tham số nào requires_grad) | PASS |
| B1 | đặc trưng frame có shape [N_video, F_max, D] | PASS |
| B2 | đặc trưng frame hữu hạn, chuẩn L2 ≈ 1 | PASS |
| B3 | đúng 1 embedding / video, hữu hạn, chuẩn 1, video_id duy nhất | PASS |
| B4 | đúng 1 embedding / caption, hữu hạn, cùng chiều với video | PASS |
| B5 | cosine văn bản–video nằm trong [−1, 1] | PASS |
| B6 | embedding video thật sự gộp từ NHIỀU frame (không vô tình dùng 1 frame) | PASS |
| B7 | lấy mẫu ổn định: đủ F_max frame / video, timestamp tăng dần trong [0, thời lượng) | PASS |
| B8 | mọi caption-truy vấn ánh xạ đúng 1 video dương cùng split; gallery mỗi split không nhân bản video | PASS |
| B9 | cache đặc trưng nạp lại được mà KHÔNG chạy encoder, giống hệt bản trong bộ nhớ | PASS |
| C1 | [12.8] canary chỉ có trong caption TEST / tập ngoài không lọt vào từ vựng, quan hệ, từ vựng vị từ | PASS |
| C1b | đối chứng dương: canary đặt vào TRAIN thì bị phát hiện | PASS |
| C2 | [12.8] xáo/thay caption TEST + tập ngoài: từ vựng, quan hệ, vị từ, tagger, taxonomy, inventory, đồ thị, mẫu gán nhãn GIỐNG HỆT | PASS |
| C3 | [12.8] hàng rào: chặn mở test / tập ngoài trước đóng băng + caption sai mục đích | PASS |
| G1 | [12.5] inventory CỐ ĐỊNH giữa mọi biến thể đồ thị; cùng cấu hình mô hình ⇒ cùng số tham số | PASS |
| G2 | mọi cạnh tuân thủ schema (kiểu node, POS, trọng số, vị từ đúng họ, không lẫn tên loại quan hệ) | PASS |
| G3 | [12.9] điểm đường đi ma trận = liệt kê ở cả 5 path_mode; tính lại R khớp; đường đi khớp vị từ ghi đúng vị từ + chiều | PASS |
| G4 | [12.6, 12.7] message passing: trọng số dùng đúng công thức, chỉ nhân MỘT lần (khớp tính tay, bất biến tỉ lệ), w=0 ≡ không cạnh, node cô lập hữu hạn, bất biến thứ tự, gradient hữu hạn | PASS |
| P1 | [12.1] embedding vị từ TÁC ĐỘNG forward khi pred_mode ≠ none; pred_mode none (≡ v2) KHÔNG dùng vị từ | PASS |
| P2 | [12.2] hai vị từ khác nhau trên cùng cặp node ⇒ thông điệp khác nhau | PASS |
| AB | [12.5] mỗi ablation tắt ĐÚNG thành phần mô tả (mask, chính sách support, pred_mode, path_mode) | PASS |
| G5 | [12.7] strict inductive (video test MSVD + tập ngoài): KHÔNG đổi node lớp/khái niệm hay video khác; transductive thì đổi; không NaN | PASS |
| G6 | [12.3] đối chứng rewiring giữ kiểu node, họ quan hệ, bậc, trọng số, đa tập vị từ; đổi ≥ ngưỡng (nếu không: KHÔNG kết luận về topology) | PASS |
| G7 | [12.4] đối chứng hoán vị vị từ KHÔNG đổi topology / trọng số; chỉ đổi vị từ trong họ svo/prep (nếu không: KHÔNG kết luận về vị từ) | PASS |
| C7 | split MSVD tái lập được (sha1 giống v2) | PASS |
| Q3 | [v4] Σ số hạng (thị giác + nhãn + đường đi quan hệ) = điểm mô hình (T = 1, 2); xoá một cạnh ⇒ điểm đổi ĐÚNG bằng phần đóng góp các đường đi qua cạnh đó | PASS |
| Q1 | [v4] vị từ của truy vấn tác động attention trên ĐÚNG cạnh mang vị từ đó; v4_pred = False thì không (đối chứng âm); hoán vị vị từ đổi tín hiệu khớp nhưng giữ topology | PASS |
| Q2 | [v4] attention phụ thuộc truy vấn khi v4_qcond = True; giống hệt giữa các truy vấn khi False (đối chứng âm) | PASS |
| Q4 | [v4] inductive: đổi đặc trưng/nhãn của MỘT video test (tập ngoài) không đổi điểm video khác (dung sai 1e-6 như G5: GPU cộng dồn atomic không tất định); rel = False ⇒ không số hạng quan hệ; điểm + gradient hữu hạn | PASS |
| KG1 | cạnh ConceptNet trong đồ thị G9: chỉ nối lớp–lớp, không self-loop, hai đầu đều là node đang hoạt động; rewiring hợp lệ | PASS |
| D1 | mọi lựa chọn của v4 chỉ dựa trên VAL và đóng băng TRƯỚC khi chạm test MSVD / tập ngoài | PASS |
| QS1 | bộ truy vấn trực tiếp / gián tiếp ≥ 20 / nhóm trên tập đánh giá chính, theo luật đăng ký trước, dựng TRƯỚC khi chấm mô hình (không dùng điểm mô hình) | PASS |
| D2 | mọi thí nghiệm dùng chung split, gallery, truy vấn, tập ứng viên CLIP và hàm metric (từng tập đánh giá) | PASS |
| D3 | R@1/5/10, MRR, macro-MRR theo video + kiểm định cặp cho G0–G6 + v3 trên tập xác nhận độc lập | PASS |
| X1 | [12.9] giải thích hợp lệ (cạnh thật đúng loại/chiều; vị từ khớp nằm trên ĐÚNG cạnh); Σ số hạng = điểm; ≥ 10 truy vấn | PASS |
| X2 | xoá cạnh của các đường đi ỦNG HỘ (giải thích) làm điểm mô hình giảm nhiều hơn xoá cùng số cạnh quan hệ ngẫu nhiên (vào nhãn của cùng video) | PASS |
| X3 | T = 1: Δ điểm khi xoá cạnh = đúng tổng đóng góp đã báo (fidelity chính xác theo cấu tạo) | PASS |
| X5 | giải thích đầy đủ hợp lệ (mọi cạnh contains / quan hệ có thật, đúng loại, chiều, kiểu node; vị từ đúng cạnh); Σ số hạng = điểm; ≥ 10 truy vấn | PASS |
| K1 | phân tích ≥ 5 thành công + ≥ 5 thất bại, chọn theo luật, có đường đi thật, thứ hạng và nguyên nhân | PASS |
| DM1 | demo: văn bản tự do → top-k video + điểm CLIP / đồ thị / cuối + giải thích (hoặc báo rõ không có đường đi hợp lệ) | PASS |
| C8 | [12.8] nhật ký: caption TEST / tập ngoài chỉ dùng cho encode / query_input / eval / report | PASS |
| C9 | [12.8] provenance: artifact học được chỉ từ caption TRAIN; lựa chọn chỉ từ VAL | PASS |
| D4 | mọi lần chạm test MSVD / tập ngoài sau thời điểm đóng băng | PASS |
| R1 | [12.10] kết quả trong báo cáo / kết luận khớp các file CSV đầu ra (đọc lại từ đĩa) | PASS |
