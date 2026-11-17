# Data card — MMKG Ẩm thực Việt Nam (`mmkg/`)

MMKG nhỏ do nhóm tự xây cho cấp 3 của Project 3. Phục vụ truy vấn văn bản → ảnh có giải thích bằng đường đi trong đồ thị tri thức.

## Tóm tắt

| Mục | Giá trị | Yêu cầu đề bài |
| --- | --- | --- |
| Miền | Ẩm thực Việt Nam (món ăn, đồ uống) | Miền hẹp, ưu tiên tiếng Việt |
| Ngôn ngữ | Tiếng Việt (tên thực thể và quan hệ); có bản dịch tiếng Anh để phân tích chuyển giao | — |
| Thực thể | **148**: 45 món, 54 nguyên liệu, 10 nhóm nguyên liệu, 12 loại món, 10 cách chế biến, 13 tỉnh/thành, 4 vùng | 50–150 |
| Triple | **750**: 434 triple tri thức + 316 triple ảnh–món (`mô_tả`) | 300–1.000 |
| Ảnh | **316**, của 44/45 món (Lẩu mắm không có ảnh đạt yêu cầu); 1–10 ảnh/món, trung vị 8 | 200–500 |

Ảnh là node của đồ thị nhưng không tính vào số thực thể.

## Tệp

| Tệp | Nội dung |
| --- | --- |
| `kg_data.py` | Nguồn của tri thức: danh sách món kèm loại, xuất xứ, cách chế biến, nguyên liệu; nhóm nguyên liệu; tỉnh → vùng; tên tiếng Anh. `build_kg()` sinh thực thể và triple |
| `images.csv` | Một dòng/ảnh: `image_id`, đường dẫn, món, tiêu đề tệp trên Commons, URL trang nguồn, tác giả, giấy phép, URL giấy phép, cách liên kết, từ khoá khớp, nhóm gần trùng |
| `image_candidates_review.csv` | 410 ứng viên và quyết định cho từng ảnh: giữ, loại khi duyệt bằng mắt, gần trùng, vượt hạn 10 ảnh/món |
| `images/<món>/` | Ảnh thu nhỏ rộng 384 px |
| `collect_images.py` | Mã thu thập và lọc ảnh (xem quy trình bên dưới) |
| `Project3_MMKG_VN.ipynb` | Notebook dựng KG, chạy phương pháp, ablation, xAI, phân tích chuyển giao; có output của lần chạy Kaggle |
| `../results/mmkg/` | Output của lần chạy Kaggle `hieutrinh123/project3-mmkg-vn` v4: `entities.csv`, `triples.csv`, `splits.json`, `queries.json`, `main_results.csv`, `paired_tests.csv`, `transfer_results.csv`, `xai_per_query.csv`, `xai_summary.json`, `summary.json`, `kaggle_log.txt` |
| `kaggle/build_kernel.py` | Sinh script kernel Kaggle (repo private nên nhúng mã nguồn; ảnh lấy từ dataset `hieutrinh123/project3-mmkg-vn-images`) |

## Lược đồ

Loại thực thể: `món`, `nguyên liệu`, `nhóm nguyên liệu`, `loại món`, `cách chế biến`, `tỉnh`, `vùng`, cộng node `ảnh`.

| Quan hệ | Đầu → đuôi | Số triple |
| --- | --- | --- |
| `thuộc_loại` | món → loại món | 45 |
| `xuất_xứ` | món → tỉnh (hoặc vùng, khi món phổ biến khắp vùng và không có nơi gốc được thừa nhận rộng rãi: Bánh xèo, Bánh chưng, Bánh tét, Gỏi cuốn…) | 45 |
| `chế_biến_bằng` | món → cách chế biến | 56 |
| `có_nguyên_liệu` | món → nguyên liệu | 221 |
| `thuộc_nhóm` | nguyên liệu → nhóm nguyên liệu | 54 |
| `thuộc_vùng` | tỉnh → vùng | 13 |
| `mô_tả` | ảnh → món | 316 |

Đường đi 2 bước dùng để suy luận và giải thích: món → nguyên liệu → nhóm (ví dụ "có hải sản"), món → tỉnh → vùng (ví dụ "của Nam Bộ").

## Nguồn dữ liệu và cách tạo triple

**Triple tri thức (434).** Nhóm tự biên soạn với sự hỗ trợ của công cụ AI (Claude), dựa trên kiến thức phổ thông về món ăn Việt Nam. Nhóm có đối chiếu với các bài Wikipedia tiếng Việt/tiếng Anh của từng món, nhưng chưa có người duyệt độc lập. Các quy ước:
- Nguyên liệu là **nguyên liệu đặc trưng** (4–8 món mỗi món), không phải công thức đầy đủ. Gia vị phổ biến (đường, tiêu, tỏi) bị bỏ để giữ số thực thể ≤ 150.
- Một số nguyên liệu được gộp để giảm số thực thể: `cua` (cua đồng + cua biển), `trứng` (trứng gà + trứng vịt), `sợi mì gạo` (sợi mì Quảng + sợi cao lầu). Cách chế biến "ninh" gộp vào "nấu".
- Xuất xứ dùng **địa danh truyền thống**, trước đợt sắp xếp đơn vị hành chính năm 2025 (ví dụ Quảng Nam, Sóc Trăng, Ninh Thuận). Xuất xứ của nhiều món còn tranh cãi (Phở: Hà Nội hay Nam Định), KG chọn một nơi phổ biến nhất.
- Vùng: Bắc Bộ, Trung Bộ (gồm Thanh Hóa), Nam Bộ, Tây Nguyên (Lâm Đồng).

**Ảnh (316) và triple `mô_tả`.** Lấy từ [Wikimedia Commons](https://commons.wikimedia.org) qua API, ngày 17/11/2026 (`collect_images.py`):
1. Với mỗi món, lấy tệp trong `Category:<tên món>` rồi tìm kiếm Commons theo tên có dấu, không dấu và tên tiếng Anh.
2. Lọc tự động: JPEG/PNG rộng ≥ 300 px; giấy phép CC0, CC BY, CC BY-SA hoặc Public domain; tiêu đề / danh mục / mô tả phải chứa từ khoá của món. Khi khớp nhiều món, chọn từ khoá dài nhất ("bun cha ca" thắng "bun cha"). Bỏ tiêu đề chứa từ chỉ cửa hàng, biển hiệu, chợ, thực đơn.
3. Giới hạn đa dạng: tối đa 12 ứng viên/món, không quá 2 tệp cùng gốc tên, 4 tệp cùng tác giả.
4. **Duyệt bằng mắt** 410 ứng viên trên bảng ảnh theo món (do Claude thực hiện trong phiên làm việc, cần nhóm xem lại). Đã loại 77 ảnh: tháp Chăm "Bánh Ít", biển hiệu, quầy hàng, chân dung, tranh vẽ, ảnh không thấy rõ món, ảnh chụp món khác (chả cá chiên khác chả cá Lã Vọng, các loại chả nướng khác giò lụa…). Loại thêm 4 ảnh gần trùng (dHash) và 13 ảnh vượt hạn 10 ảnh/món.

Liên kết ảnh–món: 186 ảnh từ danh mục Commons của món, 125 từ tìm kiếm, 5 tìm thấy qua truy vấn của món khác nhưng từ khoá khớp món này. 289 ảnh khớp từ khoá ở tiêu đề, 25 ở danh mục, 2 ở mô tả.

## Chia dữ liệu và truy vấn

- Chia **theo ảnh trong từng món**, train / val / test ≈ 46 / 22 / 32%, 3 seed. Ảnh cùng nhóm gần trùng (cùng tác giả + cùng gốc tên tệp) luôn cùng một tập. Cạnh `mô_tả` của ảnh val/test không có trong đồ thị khi chạy mô hình.
- Truy vấn trực tiếp: tên món (44).
- Truy vấn gián tiếp: sinh tự động từ KG, tổ hợp 2–3 thuộc tính xác định duy nhất một món; bộ val và bộ test khác cách diễn đạt và khác tổ hợp (44 + 44). 41/44 truy vấn test cần đường đi 2 bước.
- Diễn đạt tự nhiên: 24 câu viết tay, dùng từ ngoài KG ("heo", "miền Tây", "chả giò"), chỉ dùng trên test.

## Điều kiện sử dụng

- **Ảnh** giữ giấy phép gốc của từng tệp trên Wikimedia Commons (cột `license`, `license_url`, `author`, `source_url` trong `images.csv`): 33 CC0, 40 Public domain, 58 CC BY (2.0/2.5/3.0/4.0), 185 CC BY-SA (2.0/3.0/4.0). Khi dùng lại phải ghi tác giả và giấy phép; ảnh CC BY-SA phải giữ cùng giấy phép cho tác phẩm phái sinh. Ảnh trong repo là bản thu nhỏ 384 px, không chỉnh sửa nội dung. Một ảnh thiếu tên tác giả trong siêu dữ liệu; xem trang nguồn.
- **Triple tri thức** do nhóm biên soạn, dùng cho mục đích học tập và nghiên cứu. Không dùng làm nguồn tra cứu ẩm thực chính thức.

## Hạn chế

- Tri thức chưa được người duyệt độc lập; nguyên liệu và xuất xứ là lựa chọn đơn giản hoá.
- Việc duyệt ảnh bằng mắt do AI làm trên ảnh thu nhỏ; có thể còn ảnh sai món, nhất là các cặp khó phân biệt (Phở bò / Phở gà, Bánh bèo / Bánh bột lọc, Bánh căn / Bánh khọt).
- Lệch nguồn: ảnh Commons thiên về ảnh du lịch, nhà hàng ở nước ngoài, một số tác giả đóng góp nhiều ảnh. Số ảnh mỗi món lệch (1–10).
- Truy vấn gián tiếp sinh theo mẫu dùng đúng từ vựng của KG, nên liên kết thực thể gần như luôn đúng; đây là điều kiện thuận lợi cho phương pháp đồ thị. Bộ diễn đạt tự nhiên được thêm để đo phần này.
- Đáp án của truy vấn gián tiếp được xác định theo KG, không phải theo thế giới thật: một món có thể thật sự chứa nguyên liệu mà KG không liệt kê.
