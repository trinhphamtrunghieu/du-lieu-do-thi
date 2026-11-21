---
title: Du Lieu Do Thi
emoji: 👀
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.32.0
python_version: 3.12.12
app_file: app.py
pinned: false
license: mit
short_description: Demo DLDT
---

Demo Project 3 (Học máy với dữ liệu đồ thị): nhập truy vấn tiếng Việt → top-k ảnh món ăn + quan hệ/đường đi giải thích
trên MMKG Ẩm thực Việt Nam (148 thực thể, 750 triple, 316 ảnh Wikimedia Commons).

Điểm = α·z(cos CLIP) + (1−α)·z(Σ_d P(món d | truy vấn)·p(món d | ảnh)); α, λ, η, T, w2 chọn trên val.
Mô hình trên Space: CLIP đa ngữ `xlm-roberta-base-ViT-B-32` (chạy CPU; Space dùng phần cứng ZeroGPU, hàm truy vấn gắn `@spaces.GPU`). Báo cáo dùng `xlm-roberta-large-ViT-H-14` (GPU).

Mã nguồn: MIT. Ảnh lấy từ Wikimedia Commons, mỗi ảnh giữ giấy phép riêng (CC0 / CC BY / CC BY-SA / public domain); tác giả và giấy phép của từng ảnh nằm trong `mmkg/images.csv` và hiện dưới mỗi kết quả.
