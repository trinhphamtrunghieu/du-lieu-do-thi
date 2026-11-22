# Tập chia dữ liệu cố định

Mọi thí nghiệm dùng một split cố định cho mỗi tập dữ liệu. Split chia theo ảnh hoặc video, không chia theo caption. Test không dùng để huấn luyện, chọn siêu tham số hay xây cạnh thống kê.

## Danh mục

| Phần | Tập dữ liệu | File split | Quy tắc / mã tạo split | sha1 |
| --- | --- | --- | --- | --- |
| A | Flickr8k | `splits/partA_flickr8k_split.json`, `.csv` | Split chính thức HF `jxie/flickr8k`: train 6000 / val 1000 / test 1000 | `005202c69674` |
| A | Action Genome | `splits/partA_action_genome_split.json`, `.csv` | Train/test chính thức của AG; val = 15% train gốc, `random.Random(20240101)`: train 6611 / val 1166 / test 1810 video | `1f1f4e811772` |
| B | VG + COCO (TN1), ARO / SugarCrepe / COCO-spatial (TN2, TN2b) | `results/partB_image_video/splits/*.csv` | `results/partB_image_video/splits/SPLIT_RULES.json` (băm sha1 theo id, có md5 kiểm tra) | xem `SPLIT_RULES.json` |
| B, C | MSVD (+ MSR-VTT ở C, chỉ để xác nhận) | `results/partC_level4b/split_manifest.json`, `results/partB_image_video/video/split_manifest.json` | Split chính thức MSVD; MSR-VTT là tập ngoài `ext` | `56f2e89a1135` |
| MMKG | Ẩm thực Việt Nam | `results/mmkg/splits.json` | Một split cho mỗi seed 0, 1, 2 | — |

## Phần A

`partA_*_split.json` là file `split.json` gốc lấy từ output Kaggle `hieutrinh123/du-lieu-do-thi` (v11, ViT-H/14). Notebook gộp `hieutrinh123/project3-merged` (ViT-L/14) in ra đúng sha1 này (`results/logs/merged_v*_kaggle_log.txt`), nên hai backbone dùng chung split.

Định dạng: `{"train": [image_id...], "val": [...], "test": [...], "seed": ..., "sha1": ...}`. `image_id` là chỉ số nội bộ của notebook. File CSV đi kèm ánh xạ chỉ số đó sang định danh gốc:

- Flickr8k: `hf_split`, `hf_row` = tên split và số thứ tự hàng trong HF `jxie/flickr8k`.
- Action Genome: `video_id` của Charades/AG, và `ag_official_split` (train/test gốc).

Sinh lại và kiểm tra:

```bash
python splits/make_splits.py --ag-records /kaggle/input/action-genome/action-genome/ag_records.json
```

Script sinh lại các danh sách theo đúng quy tắc của `make_split` / `ag_make_split` trong notebook phần A. Nó so với file JSON đã lưu, kiểm tra sha1 và việc không chồng lấn giữa các split, rồi ghi lại CSV. Thiếu `--ag-records` thì chỉ sinh lại Flickr8k, còn AG chỉ được kiểm tra sha1 và chồng lấn. `ag_records.json` nằm trong dataset Kaggle `hieutrinh123/action-genome`.
