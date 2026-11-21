# Demo: nhập truy vấn, xem top-k kết quả và đường đi giải thích

Thư mục này là sản phẩm cho mục 8 của đề: *"Notebook hoặc giao diện nhỏ: nhập truy vấn, trả về top-k kết quả và hiển thị quan hệ/đường đi giải thích"*.

`app.py` có 3 tab. Mỗi tab xếp hạng và giải thích theo đúng cách của notebook gốc:

| Tab | Dữ liệu | Engine | Dữ liệu demo lấy ở đâu |
| --- | --- | --- | --- |
| MMKG Ẩm thực Việt Nam | `mmkg/` (tiếng Việt, 316 ảnh) | `mmkg_engine.py`, tính lại khi khởi động | Có sẵn trong repo |
| Action Genome | gallery test phần A (video Charades) | `partA_engine.py` | Bundle `bundles/action_genome`, notebook phần A xuất trên Kaggle |
| MSVD | gallery test MSVD phần C | `partC_engine.py` + `partC_lib.py` | Bundle `bundles/msvd`, notebook phần C xuất trên Kaggle |

| Tệp | Làm gì |
| --- | --- |
| `Demo_Truy_Van.ipynb` | Notebook MMKG đã chạy sẵn output: gọi `search()`, vẽ đồ thị con giải thích, đo fidelity, có **giao diện ipywidgets** |
| `app.py` | Giao diện web Gradio 3 tab, dùng khi demo trực tiếp và trên Hugging Face Spaces |
| `mmkg_engine.py` | Chấm điểm MMKG theo công thức của `mmkg/Project3_MMKG_VN.ipynb` (mục 5, 9), chọn α, λ, η, T, w2 trên val |
| `partA_engine.py` | Bản port `demo_query`, `ag_parse_query`, `enumerate_paths`, `explain_and_score` của notebook phần A, chạy trên bundle |
| `partC_engine.py`, `partC_lib.py` | Bản port `demo_query` và `explain_full` của notebook phần C. Script `notebooks/tools/gen_partC_lib.py` sinh `partC_lib.py` bằng cách chép nguyên văn mã xử lý truy vấn và lớp QC-RGAT |
| `fetch_bundles.py` | Tải bundle AG và MSVD từ output kernel Kaggle về `demo/bundles/` (thư mục này không vào git) |
| `viz.py` | Vẽ đồ thị con giải thích (networkx) và thẻ kết quả HTML |
| `build_space.sh`, `space_template/` | Đóng gói thư mục cho Hugging Face Spaces. `.github/workflows/deploy-hf-space.yml` gọi script này tự động |
| `cache/` | Đặc trưng CLIP và GNN đã huấn luyện của MMKG (B/32, seed 0) |
| `export_static.py`, `static_template/` | Xuất bản tĩnh, tính sẵn kết quả, cho GitHub Pages |

## Mỗi kết quả MMKG có gì

- Ảnh, tác giả và giấy phép trên Wikimedia Commons.
- Điểm tách làm hai: `điểm = CLIP + đồ thị`, tức α·z(cos) cộng (1−α)·z(điểm đồ thị).
- **Món trung gian d\***: món đóng góp P(d|q)·p(d|ảnh) lớn nhất, kèm phần trăm điểm đồ thị nó chiếm. Dấu ✓ nghĩa là d\* đúng là món trong ảnh, ✗ là sai.
- **Đường đi KG** từ thực thể trong truy vấn tới d\*: một bước (`Bánh bột lọc —xuất_xứ→ Huế`) hoặc hai bước (`Bún mắm —xuất_xứ→ Sóc Trăng —thuộc_vùng→ Nam Bộ`). Ràng buộc nào d\* không thỏa, app tô đỏ.
- **Ảnh bằng chứng**: các ảnh train gần nhất có cạnh `mô_tả` trỏ tới d\*.
- **Đồ thị con giải thích**: truy vấn → thực thể ← d\* ← ảnh kết quả và ảnh bằng chứng.
- **Fidelity** (bật nếu muốn): app xoá đúng các cạnh trong giải thích rồi chấm lại, và so với việc xoá cùng số cạnh chọn ngẫu nhiên. Với «đồ uống của Hà Nội có trứng», ảnh #1 rơi từ hạng 1 xuống 14 khi xoá 6 cạnh giải thích, nhưng vẫn giữ hạng 1 khi xoá 6 cạnh ngẫu nhiên.

Tab AG hiện đường đi video → đối tượng → khái niệm và đóng góp của từng đường đi. Tab MSVD hiện các số hạng đồ thị dương lớn nhất (video chứa khái niệm, hoặc đường đi quan hệ giữa các khái niệm).

MMKG chạy mặc định bằng CLIP đa ngữ `xlm-roberta-base-ViT-B-32` trên CPU. Muốn dùng mô hình chính của báo cáo, bạn đặt `P3_DEMO_MODEL=mclip_h14` (cần GPU). B/32 đoán món của ảnh kém hơn H/14, nên có truy vấn mà KG chọn đúng món nhưng ảnh trả về lại sai. Bạn dùng các ca này cho phần phân tích truy vấn thất bại. Số liệu chính thức nằm ở `results/mmkg/`.

## Bundle Action Genome / MSVD và checkpoint

Ảnh, video, đồ thị và mô hình của phần A và phần C chỉ có trên Kaggle. Vì vậy hai notebook tự xuất một **bundle** cho demo:

- **Notebook phần A**, mục 1.17 mới. `save_ckpt` lưu trọng số cùng α, λ của từng tổ hợp (bộ dữ liệu, CLIP, ablation, seed) vào `checkpoints/<tag>_<clip>_<ablation>_s<seed>.pt`. Khi bạn chạy lại, `train_or_load` tìm file cùng tên trong `checkpoints/`, trong `P3_RESUME_DIR` và trong `/kaggle/input/**/checkpoints`, rồi nạp lại và **bỏ qua huấn luyện**. Hàm chỉ nạp khi đồ thị dựng lại có cùng số node và số cạnh. Đặt `P3_RESUME=0` để tắt. Ở bước 9b/9 của AG, `export_demo_bundle` ghi đặc trưng CLIP và nhúng GNN của gallery test, ma trận đường đi, đồ thị dùng cho giải thích, đầu chiếu truy vấn, α, λ, thumbnail và **kết quả tham chiếu** của `demo_query`.
- **Notebook phần C**. `run_exp` nạp `checkpoints/<exp>_s<seed>.pt` nếu tìm thấy: hàm dựng mô hình đúng kiến trúc rồi gọi `load_state_dict`, không huấn luyện lại. Đặt `MSVD_RESUME=0` để tắt. Một cell mới sau cell demo xuất bundle MSVD gồm mô hình G3 seed đầu, tensor đồ thị, từ vựng node và vị từ, đặc trưng gallery, thumbnail và kết quả tham chiếu.
- Hai script `notebooks/tools/patch_partA_checkpoint.py` và `patch_partC_checkpoint.py` thêm các thay đổi trên vào notebook. Chạy lại nhiều lần cũng không sao.
- `python notebooks/tools/build_export_kernels.py` dựng hai kernel xuất bundle trong `notebooks/kaggle_export/{ag,msvd}`. Bạn đẩy kernel và tải kết quả bằng `notebooks/tools/kaggle_push.py` (gọi REST API, cần Kaggle API key):
  ```bash
  python notebooks/tools/kaggle_push.py push notebooks/kaggle_export/ag
  ```
  ```bash
  python demo/fetch_bundles.py
  ```
- Để kiểm tra bản port, mỗi bundle có `reference_outputs.json` chứa top-k, điểm, giải thích **và vector truy vấn** do notebook tính. Engine chạy lại các truy vấn đó với cùng vector. Lần kiểm tra gần nhất khớp cả 26 truy vấn AG và 20 truy vấn MSVD, sai số điểm dưới 1e-6.

Bundle không vào git vì ảnh Charades và clip YouTube có giấy phép riêng. Để chép bundle lên Space, chạy `P3_WITH_BUNDLES=1 bash demo/build_space.sh` rồi upload. GitHub Actions không xoá bundle đã có trên Space.

## Chạy trên máy

```bash
pip install -r requirements.txt
```

```bash
python demo/app.py
```

Mở http://127.0.0.1:7860. Lần đầu app mất 1–2 phút để tải CLIP (~1,5 GB) và trích đặc trưng 316 ảnh MMKG, rồi lưu đặc trưng vào `demo/cache/`. Từ lần sau, app khởi động trong khoảng 10 giây. Nếu có `demo/bundles/`, app tải thêm CLIP ViT-L/14 cho hai tab AG và MSVD.
Muốn chạy notebook, mở `demo/Demo_Truy_Van.ipynb` và chọn Run All.

Sau khi đã tải mô hình một lần, bạn nên đặt `HF_HUB_OFFLINE=1` lúc demo. Có lần app treo khi nạp mô hình vì kiểm tra mạng tới HF Hub:

```bash
HF_HUB_OFFLINE=1 python demo/app.py
```

## Triển khai khi không có host

| Cách | Nhập truy vấn tự do | Chi phí | Ghi chú |
| --- | --- | --- | --- |
| **Hugging Face Spaces, phần cứng ZeroGPU** (khuyên dùng) | Có | Miễn phí: tài khoản free host được tối đa 2 Space ZeroGPU, cần email đã xác minh và tài khoản trên 30 ngày | Link cố định. App chỉ dùng GPU để mã hoá truy vấn (`duration=10`). Truy vấn ví dụ có vector sẵn. Khi người xem hết quota GPU, app mã hoá bằng CPU và vẫn trả kết quả |
| Colab hoặc Kaggle với `P3_SHARE=1` | Có | Miễn phí | Link `*.gradio.live` sống khoảng 72 giờ và mất khi bạn tắt notebook |
| GitHub Pages (bản tĩnh) | Không, chỉ chọn trong 116 truy vấn tính sẵn | Miễn phí với repo public. Repo private cần GitHub Pro | Không chạy mô hình. Chỉ để dự phòng |
| Máy cá nhân lúc vấn đáp | Có | Không tốn | Ổn định nhất. Đề yêu cầu kèm video dự phòng không quá 5 phút |

### Hugging Face Spaces (ZeroGPU)

Theo tài liệu HF hiện tại, tạo Space Gradio hoặc Docker trên CPU basic cần gói trả phí. Tài khoản miễn phí chỉ host được **ZeroGPU**: tối đa 2 Space, cần email đã xác minh và tài khoản tạo trên 30 ngày.

`app.py` chạy được trên ZeroGPU:
- Chỉ hàm mã hoá truy vấn có `@spaces.GPU(duration=10)`. Chấm điểm, giải thích và fidelity chạy trên CPU.
- Space nạp GNN đã huấn luyện từ `demo/cache/`, nên không gọi optimizer lúc khởi động. Trên ZeroGPU, `optimizer.step()` khởi tạo CUDA ngoài hàm `@spaces.GPU` và làm app sập.
- Khách chưa đăng nhập có rất ít quota GPU. Khi họ dùng hết, app chuyển sang CPU.
- `space_template/README.md` đặt `python_version: 3.12.12` và `requirements.txt` dùng `torch>=2.8`, theo yêu cầu của ZeroGPU.

Các bước:
1. Tạo tài khoản ở huggingface.co, rồi chọn **New Space**: SDK **Gradio**, phần cứng **ZeroGPU**, chế độ Public để giảng viên mở được mà không cần đăng nhập. Nếu tài khoản của bạn chưa đủ 30 ngày, dùng tài khoản của một thành viên khác trong nhóm.
2. Đóng gói. Thư mục kết quả gồm app, các engine, `mmkg/` (KG và ảnh), split, truy vấn và đặc trưng đã cache. Chỉ MMKG thì khoảng 23 MB. Thêm `P3_WITH_BUNDLES=1` để gói cả bundle AG và MSVD, khi đó khoảng 100 MB:
   ```bash
   bash demo/build_space.sh
   ```
3. Upload. `hf upload` tự xử lý file ảnh nhị phân, bạn không cần cấu hình Git LFS:
   ```bash
   pip install -U huggingface_hub
   ```
   ```bash
   hf auth login
   ```
   ```bash
   hf upload <tài-khoản>/<tên-space> build/hf_space . --repo-type space
   ```
4. Space cài `requirements.txt` rồi chạy `app.py`. Lần build đầu mất 5–10 phút. Link có dạng `https://huggingface.co/spaces/<tài-khoản>/<tên-space>`.
5. Space ngủ khi không ai truy cập một thời gian. Trước buổi vấn đáp, bạn mở link một lần để đánh thức nó.

Nếu bạn không đủ điều kiện ZeroGPU, dùng link tạm từ Colab hoặc Kaggle (bên dưới), hoặc chạy trên máy cá nhân.

### Tự động cập nhật Space bằng GitHub Actions

Workflow `.github/workflows/deploy-hf-space.yml` đóng gói bằng `build_space.sh` rồi upload lên Space mỗi khi bạn push vào `kaggle-notebook-v11` và thay đổi một trong các phần: `demo/`, dữ liệu MMKG (`mmkg/kg_data.py`, `images.csv`, `images/`), hoặc `results/mmkg/{splits,queries}.json`. Bạn cũng chạy tay được ở **Actions → Deploy HF Space → Run workflow**.

Cài đặt một lần:
1. Tạo token Hugging Face quyền **Write** (huggingface.co → Settings → Access Tokens).
2. Trong repo GitHub, vào **Settings → Secrets and variables → Actions → New repository secret**, đặt tên `HF_TOKEN` và dán token vào.
3. (Tuỳ chọn) Ở tab **Variables**, đặt `HF_SPACE` = `<tài-khoản>/<tên-space>` nếu bạn muốn đổi Space đích. Mặc định là `trinhphamtrunghieu/du-lieu-do-thi`.

Repo đã commit sẵn `demo/cache/feats_mclip_b32.npz` và `demo/cache/gnn_mclip_b32_seed0.pt`, để Space khởi động nhanh và không phải huấn luyện GNN trên ZeroGPU. Khoá của cache tính từ đặc trưng, split và mã huấn luyện. Khi một trong ba thứ này đổi, workflow tạo lại cache trên CPU của runner (mất vài phút) và ghi cảnh báo vào log. Để lần deploy sau nhanh hơn, bạn chạy `python demo/app.py` ở máy rồi commit `demo/cache/`.

### Link tạm từ Colab/Kaggle

```bash
P3_SHARE=1 python demo/app.py
```

### Bản tĩnh cho GitHub Pages

```bash
python demo/export_static.py --out build/static_demo
```

Sau đó bạn đưa `build/static_demo/` (~5 MB) lên nhánh hoặc thư mục mà GitHub Pages phục vụ, hoặc lên một repo public riêng.
