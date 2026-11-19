# Demo: nhập truy vấn → top-k kết quả + quan hệ/đường đi giải thích

Sản phẩm nộp ở mục 8 của đề: *"Notebook hoặc giao diện nhỏ: nhập truy vấn, trả về top-k kết quả và hiển thị quan hệ/đường đi giải thích"*.

`app.py` có 3 tab, mỗi tab dùng đúng cơ chế xếp hạng + giải thích của notebook tương ứng:

| Tab | Dữ liệu | Engine | Nguồn dữ liệu cho demo |
| --- | --- | --- | --- |
| MMKG Ẩm thực Việt Nam | `mmkg/` (tiếng Việt, 316 ảnh) | `mmkg_engine.py` (tính lại khi khởi động) | Có sẵn trong repo |
| Action Genome | gallery test phần A (video Charades) | `partA_engine.py` | Bundle `bundles/action_genome` do notebook phần A xuất trên Kaggle |
| MSVD | gallery test MSVD phần C | `partC_engine.py` + `partC_lib.py` | Bundle `bundles/msvd` do notebook phần C xuất trên Kaggle |

| Tệp | Vai trò |
| --- | --- |
| `Demo_Truy_Van.ipynb` | Notebook (đã chạy, giữ output) cho MMKG: gọi `search()`, vẽ đồ thị con giải thích, đo fidelity, **giao diện ipywidgets** |
| `app.py` | Giao diện web nhỏ (Gradio) 3 tab, dùng cho demo trực tiếp và cho Hugging Face Spaces |
| `mmkg_engine.py` | Bộ máy truy vấn MMKG: cùng công thức điểm với `mmkg/Project3_MMKG_VN.ipynb` (mục 5, 9), chọn α, λ, η, T, w2 trên val |
| `partA_engine.py` | Port `demo_query`, `ag_parse_query`, `enumerate_paths`, `explain_and_score` của notebook phần A, chạy trên bundle (khớp notebook tới ~1e-7) |
| `partC_engine.py`, `partC_lib.py` | Port `demo_query` + `explain_full` của notebook phần C; `partC_lib.py` **sinh tự động** (`notebooks/tools/gen_partC_lib.py`), trích nguyên văn mã xử lý truy vấn và lớp QC-RGAT |
| `fetch_bundles.py` | Tải bundle AG / MSVD từ output kernel Kaggle về `demo/bundles/` (không nằm trong git) |
| `viz.py` | Vẽ đồ thị con giải thích (networkx) và thẻ kết quả HTML |
| `build_space.sh`, `space_template/` | Đóng gói thư mục để đưa lên Hugging Face Spaces (tự động qua `.github/workflows/deploy-hf-space.yml`) |
| `cache/` | Đặc trưng CLIP + GNN đã huấn luyện (B/32, seed 0), commit sẵn cho Space |
| `export_static.py`, `static_template/` | Xuất bản tĩnh (tính sẵn kết quả) cho GitHub Pages |

## Mỗi kết quả hiển thị gì

- Ảnh, tác giả và giấy phép (Wikimedia Commons).
- Điểm tách hai phần: `điểm = CLIP + đồ thị`, tức α·z(cos) cộng (1−α)·z(điểm đồ thị).
- **Món trung gian d\***: món có đóng góp P(d|q)·p(d|ảnh) lớn nhất, kèm tỉ lệ điểm đồ thị mà nó chiếm. Có dấu ✓/✗ cho biết d\* có đúng là món của ảnh hay không.
- **Đường đi KG** từ thực thể trong truy vấn tới d\*, 1 bước (`Bánh bột lọc —xuất_xứ→ Huế`) hoặc 2 bước (`Bún mắm —xuất_xứ→ Sóc Trăng —thuộc_vùng→ Nam Bộ`). Ràng buộc mà d\* không thỏa được tô đỏ.
- **Ảnh bằng chứng**: các ảnh train gần nhất có cạnh `mô_tả` → d\*.
- **Đồ thị con giải thích**: truy vấn → thực thể ← d\* ← ảnh kết quả / ảnh bằng chứng.
- **Fidelity** (tuỳ chọn): xoá đúng các cạnh của giải thích rồi chấm lại, so với xoá cùng số cạnh ngẫu nhiên. Ví dụ với «đồ uống của Hà Nội có trứng», ảnh #1 tụt từ hạng 1 xuống 14 khi xoá 6 cạnh giải thích, còn khi xoá 6 cạnh ngẫu nhiên thì vẫn ở hạng 1.

Mô hình mặc định là CLIP đa ngữ `xlm-roberta-base-ViT-B-32`, chạy được trên CPU. Đặt `P3_DEMO_MODEL=mclip_h14` để dùng mô hình chính của báo cáo (cần GPU). Với B/32, độ chính xác ảnh→món thấp hơn H/14, nên một số truy vấn chọn đúng món qua KG nhưng ảnh trả về lại sai. Đây là chất liệu tốt cho phần "truy vấn thất bại". Số liệu chính thức nằm ở `results/mmkg/`.

## Bundle Action Genome / MSVD và checkpoint

Ảnh/video, đồ thị và mô hình của phần A, C chỉ có trên Kaggle, nên demo dùng **bundle** do chính notebook xuất ra:

- **Notebook phần A** (mục 1.17 mới): `save_ckpt` / `train_or_load` lưu trọng số + (α, λ) của từng (bộ dữ liệu, CLIP, ablation, seed) vào `checkpoints/<tag>_<clip>_<ablation>_s<seed>.pt`; chạy lại thì tìm checkpoint cùng tên trong `checkpoints/`, `P3_RESUME_DIR` và `/kaggle/input/**/checkpoints`, nạp lại và **bỏ qua huấn luyện** (chỉ khi đồ thị dựng lại khớp số node/cạnh; `P3_RESUME=0` để tắt). `export_demo_bundle` (bước 9b/9 của AG) ghi đặc trưng CLIP + nhúng GNN của gallery test, ma trận đường đi, đồ thị cho giải thích, đầu chiếu truy vấn, α, λ, thumbnail và **kết quả tham chiếu** của `demo_query`.
- **Notebook phần C**: `run_exp` nạp `checkpoints/<exp>_s<seed>.pt` nếu có (dựng đúng kiến trúc rồi `load_state_dict`), không huấn luyện lại; `MSVD_RESUME=0` để tắt. Cell mới sau cell demo xuất bundle MSVD (mô hình G3 seed đầu, tensor đồ thị, từ vựng node/vị từ, đặc trưng gallery, thumbnail, kết quả tham chiếu).
- Patch notebook: `notebooks/tools/patch_partA_checkpoint.py`, `patch_partC_checkpoint.py` (idempotent).
- Kernel Kaggle xuất bundle: `python notebooks/tools/build_export_kernels.py` dựng `notebooks/kaggle_export/{ag,msvd}`; đẩy và lấy kết quả bằng `notebooks/tools/kaggle_push.py` (REST API, cần Kaggle API key):
  ```bash
  python notebooks/tools/kaggle_push.py push notebooks/kaggle_export/ag
  ```
  ```bash
  python demo/fetch_bundles.py
  ```
- Kiểm tra bản port: mỗi bundle có `reference_outputs.json` (top-k, điểm, giải thích **và vector truy vấn** do notebook tính). Engine chạy lại đúng các truy vấn đó bằng cùng vector và phải cho cùng kết quả.

Bundle không nằm trong git (ảnh Charades / clip YouTube có giấy phép riêng). Chép lên Space bằng `P3_WITH_BUNDLES=1 bash demo/build_space.sh` rồi upload; GitHub Actions không xoá bundle đã có trên Space.

## Chạy trên máy

```bash
pip install -r requirements.txt
```

```bash
python demo/app.py
```

Mở http://127.0.0.1:7860. Lần đầu mất khoảng 1–2 phút để tải CLIP (~1,5 GB) và trích đặc trưng 316 ảnh. Đặc trưng được cache ở `demo/cache/`, các lần sau khởi động khoảng 10 giây.
Notebook: mở `demo/Demo_Truy_Van.ipynb` và Run All.

Khi đã tải mô hình một lần, nên đặt `HF_HUB_OFFLINE=1` để không phụ thuộc mạng lúc demo (có lần nạp mô hình bị treo ở bước kiểm tra mạng tới HF Hub):

```bash
HF_HUB_OFFLINE=1 python demo/app.py
```

## Triển khai khi không có host

| Cách | Nhập truy vấn tự do | Chi phí | Ghi chú |
| --- | --- | --- | --- |
| **Hugging Face Spaces — ZeroGPU** (khuyên dùng) | Có | Miễn phí: tài khoản free host được tối đa 2 Space ZeroGPU (cần email đã xác minh, tài khoản > 30 ngày) | Link cố định. Chỉ bước mã hoá truy vấn mới chạy GPU (`duration=10`); truy vấn ví dụ dùng vector có sẵn; hết quota GPU của người xem thì tự mã hoá bằng CPU, nên không báo lỗi |
| Colab / Kaggle + `P3_SHARE=1` | Có | Miễn phí | Link `*.gradio.live` tạm thời (~72 giờ), mất khi tắt notebook |
| GitHub Pages (bản tĩnh) | Không, chỉ chọn trong 116 truy vấn tính sẵn | Miễn phí với repo public; repo private cần GitHub Pro | Không chạy được mô hình. Dùng làm dự phòng |
| Máy cá nhân lúc vấn đáp | Có | — | Ổn định nhất. Nên kèm video ≤ 5 phút như đề yêu cầu |

### Hugging Face Spaces (ZeroGPU)

Theo tài liệu hiện tại của HF, tạo Space Gradio/Docker chạy trên CPU basic cần gói trả phí. Tài khoản miễn phí chỉ host được **ZeroGPU** (tối đa 2 Space; cần email đã xác minh và tài khoản đã tạo > 30 ngày). `app.py` đã tương thích ZeroGPU: chỉ hàm mã hoá truy vấn gắn `@spaces.GPU(duration=10)`; chấm điểm, giải thích, fidelity chạy trên CPU. GNN đã huấn luyện được ship trong `demo/cache/` nên Space không chạy optimizer lúc khởi động (trên ZeroGPU, `optimizer.step()` làm khởi tạo CUDA ngoài hàm `@spaces.GPU` và app sập). Khách chưa đăng nhập có quota GPU rất nhỏ; khi hết, app tự chuyển sang CPU. `space_template/README.md` đặt `python_version: 3.12.12`, `requirements.txt` dùng `torch>=2.8`, đúng yêu cầu của ZeroGPU.

1. Tạo tài khoản tại huggingface.co, rồi tạo **New Space**: SDK **Gradio**, hardware **ZeroGPU**, chọn Public để giảng viên mở được mà không cần đăng nhập. Nếu tài khoản chưa đủ 30 ngày, dùng tài khoản của thành viên khác trong nhóm.
2. Đóng gói. Thư mục tạo ra (~23 MB) gồm app, engine, `mmkg/` (KG + ảnh), split, truy vấn và đặc trưng đã cache nếu có:
   ```bash
   bash demo/build_space.sh
   ```
3. Upload. `hf upload` tự xử lý file ảnh nhị phân, không cần cấu hình Git LFS:
   ```bash
   pip install -U huggingface_hub
   ```
   ```bash
   hf auth login
   ```
   ```bash
   hf upload <tài-khoản>/<tên-space> build/hf_space . --repo-type space
   ```
4. Space tự cài `requirements.txt` rồi chạy `app.py`. Lần build đầu mất khoảng 5–10 phút, link có dạng `https://huggingface.co/spaces/<tài-khoản>/<tên-space>`.
5. Trước buổi vấn đáp, mở link một lần để đánh thức Space (Space không có ai truy cập sẽ ngủ).

Nếu không đủ điều kiện ZeroGPU, dùng link tạm từ Colab/Kaggle bên dưới, hoặc chạy trên máy cá nhân.

### Tự động cập nhật Space bằng GitHub Actions

Workflow `.github/workflows/deploy-hf-space.yml` đóng gói (`build_space.sh`) và upload lên Space mỗi khi có push vào `kaggle-notebook-v11` làm thay đổi `demo/`, dữ liệu MMKG (`mmkg/kg_data.py`, `images.csv`, `images/`) hoặc `results/mmkg/{splits,queries}.json`. Có thể chạy tay ở tab **Actions → Deploy HF Space → Run workflow**.

Cài đặt một lần:
1. Tạo token Hugging Face quyền **Write** (huggingface.co → Settings → Access Tokens).
2. GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**: tên `HF_TOKEN`, giá trị là token trên.
3. (Tuỳ chọn) tab **Variables**: `HF_SPACE` = `<tài-khoản>/<tên-space>` nếu muốn đổi Space đích (mặc định `trinhphamtrunghieu/du-lieu-do-thi`).

`demo/cache/feats_mclip_b32.npz` và `demo/cache/gnn_mclip_b32_seed0.pt` được commit để Space khởi động nhanh và không phải huấn luyện GNN trên ZeroGPU. Khoá cache gồm đặc trưng, split và mã huấn luyện; khi một trong số đó đổi, workflow tự tạo lại cache trên CPU của runner (vài phút) và cảnh báo trong log. Nên chạy `python demo/app.py` ở máy rồi commit `demo/cache/` để lần sau deploy nhanh hơn.

### Link tạm từ Colab/Kaggle

```bash
P3_SHARE=1 python demo/app.py
```

### Bản tĩnh cho GitHub Pages

```bash
python demo/export_static.py --out build/static_demo
```

Sau đó đưa `build/static_demo/` (~5 MB) lên nhánh/thư mục mà GitHub Pages phục vụ, hoặc lên một repo public riêng.
