# Kết luận Level 4(b) — trả lời 7 câu hỏi bắt buộc (tự sinh từ số liệu của lần chạy này)

Tập chính: **ext** (confirmatory). Kết luận chung về quan hệ: Quan hệ CHƯA được xác nhận đầy đủ: trên tập xác nhận, v4 không vượt có ý nghĩa mọi đối chứng quan hệ (ủng hộ: ['H1_graph_vs_clip']). Cơ chế dùng quan hệ có thật (P/Q/X-test) nhưng tác động lên MRR không đủ để kết luận.

## 1. CLIP + Graph có tốt hơn CLIP-only không?

Trên ext (confirmatory): CLIP-only MRR 0.4960 ± 0.0000; QC-RGAT (G3) 0.5059 ± 0.0011; GraphSAGE có kiểu cạnh (R3) 0.5027 ± 0.0006; đồ thị + ConceptNet (G9) 0.5062 ± 0.0009. H1 (G3 > CLIP): ΔMRR +0.0099 (CI95 theo video, chưa hiệu chỉnh bội [+0.0065, +0.0133] ⇒ TỐT HƠN có ý nghĩa; giả thuyết một phía sau Holm: p 0.000 ⇒ ỦNG HỘ). Nhãn học vs zero-shot (H13): NOT RUN.

## 2. Đồ thị có giúp nhóm truy vấn gián tiếp không?

Mọi caption nhóm gián tiếp (n=4089): ΔMRR +0.0117 [CI +0.0075, +0.0158]; bộ 20 truy vấn gián tiếp đăng ký trước: ΔMRR +0.0200 [CI -0.0175, +0.0602]. So sánh PHỤ (không Holm).

## 3. Những loại cạnh nào thực sự có giá trị?

Quan hệ vs chỉ-contains (H2): ΔMRR -0.0017 (CI95 theo video, chưa hiệu chỉnh bội [-0.0033, -0.0002] ⇒ KÉM HƠN có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ). Họ vai agent_of/object_of (H8, huấn luyện lại không có họ): ΔMRR -0.0008 (CI95 theo video, chưa hiệu chỉnh bội [-0.0018, +0.0002] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ). Họ svo/prep (H9): ΔMRR -0.0007 (CI95 theo video, chưa hiệu chỉnh bội [-0.0017, +0.0004] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ). Vị từ đúng cạnh (H4): ΔMRR +0.0001 (CI95 theo video, chưa hiệu chỉnh bội [-0.0003, +0.0005] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ). Topology thật (H5): ΔMRR -0.0014 (CI95 theo video, chưa hiệu chỉnh bội [-0.0029, +0.0001] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ). ConceptNet vs chỉ-contains (H10): ΔMRR -0.0014 (CI95 theo video, chưa hiệu chỉnh bội [-0.0029, +0.0001] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ); ConceptNet thật vs rewired (H11): ΔMRR -0.0012 (CI95 theo video, chưa hiệu chỉnh bội [-0.0027, +0.0003] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ); thêm ConceptNet vào G3 (H12): ΔMRR +0.0003 (CI95 theo video, chưa hiệu chỉnh bội [-0.0005, +0.0012] ⇒ không khác biệt có ý nghĩa; giả thuyết một phía sau Holm: p 1.000 ⇒ KHÔNG ủng hộ). Knockout lúc suy luận (ext): agent -0.0000; object -0.0004; prep +0.0000; svo -0.0003; ALL_relations -0.0004. Một họ cạnh chỉ được coi là có giá trị khi H tương ứng được ủng hộ.

## 4. Dùng đồ thị có làm lời giải thích đáng tin cậy hơn không?

Giải thích = số hạng thật của điểm (Σ = điểm: 1.00; hợp lệ 1.00). Xoá giải thích đầy đủ: tụt hạng 50% truy vấn vs ngẫu nhiên 0%; chỉ xoá quan hệ: hạng đổi 15% (ngẫu nhiên 5%). Jaccard giải thích giữa seed 0.27. ⇒ Giải thích trung thực với ĐIỂM theo cấu tạo (phân rã chính xác). Về QUYẾT ĐỊNH xếp hạng: xoá giải thích đầy đủ làm tụt hạng nhiều hơn xoá ngẫu nhiên, nên giải thích có chỉ ra phần đồ thị gây ra thứ hạng; phần quan hệ có đổi thứ hạng ở một số truy vấn. Độ ổn định giữa seed thấp (Jaccard 0.27) ⇒ đường đi cụ thể phụ thuộc lần huấn luyện.

## 5. Phần video tái sử dụng kiến trúc retrieval và graph explanation đến mức nào?

Tái sử dụng nguyên: CLIP đóng băng (ảnh → frame), top-k theo cosine, điểm cuối α·z(CLIP) + (1−α)·z(đồ thị) chọn α trên val, schema node / cạnh (video ~ ảnh), GNN (GraphSAGE có kiểu cạnh, QC-RGAT), giải thích bằng phân rã điểm và xoá cạnh. Phần riêng của video: lấy mẫu 12 frame, gộp theo thời gian (chọn trên val), nhãn video = CLIP zero-shot trên frame, quan hệ từ caption train cấp lớp. Bằng chứng chạy: A4=PASS | B6=PASS | B8=PASS.

## 6. Yêu cầu nào PASS, FAIL hay NOT RUN?

{'PASS': 26, 'NOT RUN': 1}. FAIL: —; NOT RUN: B6 (chi tiết: audit_report.md).

## 7. Level 4(b) được thực hiện đến đâu và giới hạn là gì?

Đã có pipeline text→video đầy đủ, xếp hạng lại bằng đồ thị, đối chứng, bộ truy vấn trực tiếp/gián tiếp, xAI theo thứ hạng, 5+5 ca, demo. Tập xác nhận: OK (MSR-VTT fresh1k-C: 1000 video train-9k (ngoài 1k-A, dev, fresh1k, fresh1k-B; seed 20261031) — CHƯA dùng ở v1–v4 và Level 4(b) v1). Giới hạn: nhãn video là CLIP zero-shot (F1 explore 0.296, confirm 0.236); quan hệ là thống kê cấp lớp từ caption train (video mới không có quan hệ riêng); parser heuristic; chất lượng ngữ nghĩa cạnh chưa được người xác minh; val chỉ 100 video; test MSVD đã nhìn 4 lần trước (thăm dò); fresh1k-B cùng miền với fresh1k đã ảnh hưởng tới thiết kế. PASS kiểm thử phần mềm không phải kết luận khoa học.
