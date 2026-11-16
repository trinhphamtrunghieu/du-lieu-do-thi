# Bản Kaggle của notebook gộp

Kaggle giới hạn mã nguồn notebook ở 1 MB, còn `merged/Project3_Merged.ipynb` khoảng 2,3 MB. Bản này khác bản gộp đầy đủ đúng ba điểm, logic chạy không đổi:

1. Cell "2. DỮ LIỆU NHÚNG" của phần B được chuyển ra `assets/p3_embedded_data.py`, nạp lại bằng `exec` và có kiểm tra MD5.
2. Notebook phần C (`lamvo30ai/project03-level4b-v2-graphfix`) được nạp từ file `project03-level4b-v2-graphfix.ipynb`, MD5 giữ như cũ. Bản sao của file này nằm ở `level4b_v2_graphfix/`.
3. Bỏ các cell đã bị tắt sẵn (`%%script true`, gồm Phần 4 MSVD → MSR-VTT cũ).

Hai file payload nằm trong dataset Kaggle private `hieutrinh123/project3-merged-assets`. Kernel: `hieutrinh123/project3-merged`.
