# Demo: nhập truy vấn → top-k ảnh + quan hệ/đường đi giải thích

Sản phẩm nộp ở mục 8 của đề: *"Notebook hoặc giao diện nhỏ: nhập truy vấn, trả về top-k kết quả và hiển thị quan hệ/đường đi giải thích"*.
Demo chạy trên **MMKG Ẩm thực Việt Nam** (`mmkg/`). Ảnh, KG, split và bộ truy vấn đều có sẵn trong repo, nên không cần tải dữ liệu Kaggle.

| Tệp | Vai trò |
| --- | --- |
| `Demo_Truy_Van.ipynb` | Notebook (đã chạy, giữ output): gọi `search()`, vẽ đồ thị con giải thích, đo fidelity, **giao diện ipywidgets** |
| `app.py` | Giao diện web nhỏ (Gradio), dùng cho demo trực tiếp và cho Hugging Face Spaces |
| `mmkg_engine.py` | Bộ máy truy vấn: cùng công thức điểm với `mmkg/Project3_MMKG_VN.ipynb` (mục 5, 9), chọn α, λ, η, T, w2 trên val |
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
