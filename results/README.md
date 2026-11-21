# Kết quả thật từ Kaggle

Mọi số trong `RESULTS.md` lấy từ các thư mục này. Phần lớn số là trên tập test, trung bình 3 seed; ngoại lệ (MSVD v5 là thăm dò) được ghi trong `RESULTS.md`.

| Thư mục | Phần | Lần chạy Kaggle | Tệp nên mở trước |
| --- | --- | --- | --- |
| `partA_vitl14_flickr8k/` | A, Flickr8k, CLIP ViT-L/14 | `hieutrinh123/project3-merged` v1 (chạy lại mô hình version 9) | `final_model_test.csv`, `match_vs_clip.csv`, `xai_summary.json` |
| `partA_vitl14_action_genome/` | A, Action Genome, CLIP ViT-L/14 | như trên | `match_vs_clip.csv`, `paired_ablation_test.csv`, `xai_summary.json` |
| `partA_vith14_flickr8k/` | A, Flickr8k, CLIP ViT-H/14 | `hieutrinh123/du-lieu-do-thi` v11 | như phần ViT-L |
| `partA_vith14_action_genome/` | A, Action Genome, CLIP ViT-H/14 | `hieutrinh123/du-lieu-do-thi` v11 | như phần ViT-L |
| `partB_image_video/` | B: Visual Genome, COCO, ARO, SugarCrepe, COCO-spatial, video MSVD v5 (`video/`) | `hieutrinh123/project3-merged` v2 | `REPORT_generated.md`, `summary.json` |
| `partC_level4b/` | C: video MSVD + MSR-VTT (Level 4b v2-graphfix) | `lamvo30ai/project03-level4b-v2-graphfix`; notebook gộp v2 dùng lại đúng output này | `level4b_conclusions.md`, `graph_experiment_results.csv`, `graph_confirmatory_tests.csv` |
| `mmkg/` | Cấp 3: MMKG Ẩm thực Việt Nam | `hieutrinh123/project3-mmkg-vn` v4 | `summary.json`, `main_results.csv`, `paired_tests.csv`, `xai_summary.json`, `transfer_results.csv` |
| `logs/` | Log Kaggle của notebook gộp v1, v2 và log riêng của phần B | — | — |

Ghi chú:
- `partB_image_video/REPORT_generated.md` do notebook tự sinh, giữ nguyên như lúc chạy. Bảng tự kiểm tra trong đó cố định, có ghi chú bổ sung bên dưới.
- Lần chạy v1 của notebook gộp lỗi ở phần B; xem `logs/merged_v1_partB_failed_log.txt`. Phần B được chạy lại ở v2.
- CSV của các version cũ của phần A (v4–v9) không lưu trong repo. Số của các version đó trong `RESULTS.md` lấy từ output kernel `hieutrinh123/du-lieu-do-thi`.
