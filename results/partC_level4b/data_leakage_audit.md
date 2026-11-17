# Audit rò rỉ dữ liệu — Level 4(b) (tự sinh)

2026-10-31 13:18:10 | chạy chính thức: True | tập ngoài: OK

## Trạng thái từng benchmark

| Tập | Đã nhìn trước | Trạng thái |
|---|---|---|
| msvd_test | 5 | **EXPLORATORY** |
| msrvtt_fresh1kC | 0 | **CONFIRMATORY** |

## Kiểm tra chống rò rỉ

| id | nội dung | trạng thái | bằng chứng |
|---|---|---|---|
| A5 | split theo video, không video nào ở hai split | PASS | sha1=56f2e89a1135 → split_manifest.json |
| C7 | split MSVD tái lập được | PASS | 56f2e89a1135 |
| C1 | canary chỉ có ở caption test / tập ngoài KHÔNG lọt vào từ vựng, quan hệ, vị từ | PASS | {'canary': 'flamingo –knit→ accordion', 'splits': ['test', 'ext'], 'n_videos': 80, 'leak_vocab': [], 'leak_relations': [], 'leak_predicates': [], 'positive_control_detected': True} |
| C1b | đối chứng dương của canary | PASS | {'canary': 'flamingo –knit→ accordion', 'splits': ['test', 'ext'], 'n_videos': 80, 'leak_vocab': [], 'leak_relations': [], 'leak_predicates': [], 'positive_control_detected': True} |
| C2 | xáo caption test / tập ngoài ⇒ mọi artifact học được GIỐNG HỆT | PASS | {'vocab': True, 'relations': True, 'predicate_vocab': True, 'tagger_selection': True, 'tags': True, 'taxonomy': True, 'inventory': True, 'graph': True, 'annotation_sample': True, 'automatic_verification': True} |
| C3 | hàng rào chặn mở test / dùng caption sai mục đích | PASS | {'test_before_freeze_blocked': True, 'test→gnn_train_blocked': True, 'test→tune_blocked': True, 'test→vocab_blocked': True, 'test→relations_blocked': True, 'test→annotation_blocked': True, 'test→agreement_blocked': True, |
| C8 | nhật ký: caption test / tập ngoài chỉ dùng cho encode / query_input / eval / report | PASS | {'agreement:train': 1200, 'annotation:train': 1200, 'encode:ext': 40000, 'encode:test': 55526, 'encode:train': 97548, 'encode:val': 8580, 'eval:ext': 21000, 'eval:test': 27763, 'gnn_train:train': 3170310, 'query_input:ex |
| C9 | provenance: học chỉ từ train, chọn chỉ từ val | PASS | 25 artifact |
| D1 | mọi lựa chọn trước đóng băng, chỉ dựa trên val | PASS | sha1=a665437cfe8f; final=E10_v4_T1; G3=G3_v4_full; quyết định=[('V1', 'E11_v4_T2', False), ('V2', 'E20_tags_probe', False)] |
| D4 | mọi lần chạm test / tập ngoài sau đóng băng | PASS | 13 lần ({'test': 5, 'ext': 8}) |
| A6 | tập ngoài rời các tập đã dùng | PASS | {'label': 'MSR-VTT fresh1k-C: 1000 video train-9k (ngoài 1k-A, dev, fresh1k, fresh1k-B; seed 20261031) — CHƯA dùng ở v1–v4 và Level 4(b) v1', 'set': 'fresh1kC', 'videos': 1000, 'captions': 20000, 'captions_per_video': [2 |
| QS1 | bộ truy vấn dựng trước khi chấm mô hình | PASS | {'explore': {'direct': 20, 'indirect': 20}, 'confirm': {'direct': 20, 'indirect': 20}} |

## Nhật ký dùng caption (mục đích × split)

| mục đích | split | số lần |
|---|---|---|
| agreement | train | 1200 |
| annotation | train | 1200 |
| encode | ext | 40000 |
| encode | test | 55526 |
| encode | train | 97548 |
| encode | val | 8580 |
| eval | ext | 21000 |
| eval | test | 27763 |
| gnn_train | train | 3170310 |
| query_input | ext | 60000 |
| query_input | test | 83289 |
| query_input | train | 97548 |
| query_input | val | 8580 |
| relations | train | 49974 |
| report | ext | 6000 |
| report | test | 1340 |
| tagger_select | val | 200 |
| tagger_train | train | 1200 |
| taxonomy | train | 1200 |
| tune | val | 326040 |
| vocab | train | 1200 |

## Chất lượng ngữ nghĩa của cạnh

- Kiểm chứng tự động (parser độc lập, SILVER — không phải precision): kiểm chứng TỰ ĐỘNG (spaCy 3.8.16 / en_core_web_sm 3.8.0): tỉ lệ quan hệ heuristic được parser độc lập xác nhận — agent_of 0.89 [0.88, 0.89]; object_of 0.60 [0.59, 0.61]; prep 0.25 [0.24, 0.26]; svo 0.80 [0.79, 0.80] (23s). SILVER — không phải precision do người chấm.
- Nhãn người chấm: không có nhãn người chấm (tuỳ chọn) — chỉ có kiểm chứng tự động
- ⇒ Chất lượng ngữ nghĩa của cạnh quan hệ **CHƯA được xác nhận bởi người**. Tỉ lệ đồng thuận giữa hai parser không phải precision thật.

## Nguồn và giấy phép

- **MSVD (Microsoft Research Video Description Corpus)** (Chen & Dolan, ACL 2011) — `friedrichor/MSVD@9aa95a30` — KHÔNG XÁC NHẬN ĐƯỢC — dataset card HF không khai báo license; dữ liệu gốc (clip YouTube) phát hành cho nghiên cứu — vai trò: train: từ vựng, quan hệ, huấn luyện GNN | val: chọn cấu hình | test: THĂM DÒ
- **MSR-VTT** (Xu et al., CVPR 2016) — `friedrichor/MSR-VTT@c1af215a` — KHÔNG XÁC NHẬN ĐƯỢC — dataset card HF không khai báo license; dữ liệu gốc của Microsoft Research cho nghiên cứu — vai trò: chỉ đánh giá (chuyển giao zero-shot): không caption nào dùng để học hay chọn
