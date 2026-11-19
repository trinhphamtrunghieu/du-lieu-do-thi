---
title: MMKG Am thuc VN - truy van co giai thich
emoji: 🍜
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: cc-by-sa-4.0
---

Demo Project 3 (Học máy với dữ liệu đồ thị): nhập truy vấn tiếng Việt → top-k ảnh món ăn + quan hệ/đường đi giải thích
trên MMKG Ẩm thực Việt Nam (148 thực thể, 750 triple, 316 ảnh Wikimedia Commons).

Điểm = α·z(cos CLIP) + (1−α)·z(Σ_d P(món d | truy vấn)·p(món d | ảnh)); α, λ, η, T, w2 chọn trên val.
Mô hình trên Space: CLIP đa ngữ `xlm-roberta-base-ViT-B-32` (chạy CPU). Báo cáo dùng `xlm-roberta-large-ViT-H-14` (GPU).

Ảnh lấy từ Wikimedia Commons; tác giả và giấy phép của từng ảnh nằm trong `mmkg/images.csv` và hiện dưới mỗi kết quả.
