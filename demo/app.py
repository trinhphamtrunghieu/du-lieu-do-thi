# -*- coding: utf-8 -*-
"""Giao diện nhỏ (Gradio): nhập truy vấn → top-k ảnh + quan hệ/đường đi giải thích.

Chạy:  python demo/app.py            (mở http://127.0.0.1:7860)
       P3_SHARE=1 python demo/app.py (thêm link công khai tạm thời *.gradio.live, ~72 giờ)
Trên Hugging Face Spaces: file này là entry point (xem demo/README.md).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Hugging Face ZeroGPU (gói `spaces` có sẵn trên Space). Chỉ bước mã hoá truy vấn bằng CLIP chạy trên GPU,
# trong hàm @spaces.GPU ngắn (duration=10); chấm điểm, giải thích và fidelity chạy trên CPU nên không tốn quota
# của người xem. Truy vấn đã có vector (ví dụ, bộ truy vấn đánh giá) không gọi GPU. Hết quota GPU thì mã hoá
# bằng CPU, nên người xem không gặp lỗi.
try:
    import spaces
    ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")   # biến do ZeroGPU đặt
    os.environ.setdefault("P3_DEVICE", "cpu")
    import torch
    # ZeroGPU giả lập CUDA: optimizer.step() trên CPU vẫn gọi kiểm tra accelerator và làm khởi tạo CUDA thật
    # (RuntimeError "Low-level CUDA init"). Bình thường GNN lấy từ cache nên không huấn luyện; đây là dự phòng.
    torch.optim.Optimizer._accelerator_graph_capture_health_check = lambda self: None
except ImportError:          # chạy trên máy / Colab: không cần ZeroGPU
    ZERO_GPU = False

import gradio as gr
from mmkg_engine import MMKGSearch
from viz import result_html

ENG = MMKGSearch()
ENG._load_clip()   # bộ mã hoá CPU: dùng khi chạy trên máy và làm dự phòng khi hết quota GPU

if ZERO_GPU:
    import copy
    import torch.nn.functional as F
    try:
        _GPU_CLIP = copy.deepcopy(ENG._clip[0]).to("cuda")   # ZeroGPU: .to("cuda") lúc khởi động là hợp lệ
    except Exception as ex:
        print("Không chuyển được CLIP sang GPU, dùng CPU:", repr(ex)[:200], flush=True)
        ZERO_GPU = False

if "spaces" in sys.modules:     # Space ZeroGPU bắt buộc có ít nhất một hàm @spaces.GPU
    import torch.nn.functional as F

    @spaces.GPU(duration=10)
    def _encode_gpu(text):
        with torch.no_grad():
            t = ENG._tokenize(ENG._clip[2], [text]).to("cuda")
            return F.normalize(_GPU_CLIP.encode_text(t).float(), dim=-1)[0].cpu().numpy()


def query_vec(text):
    """Vector CLIP của truy vấn: cache → GPU (ZeroGPU) → CPU."""
    if text in ENG.Ttab:
        return ENG.Ttab[text], "cache"
    how = "cpu"
    if ZERO_GPU:
        try:
            ENG.Ttab[text] = _encode_gpu(text); how = "gpu"
        except Exception as ex:               # hết quota / hàng đợi GPU: dùng CPU
            print("ZeroGPU không dùng được, mã hoá bằng CPU:", repr(ex)[:200], flush=True)
    if text not in ENG.Ttab:
        ENG.Ttab[text] = ENG.encode_texts([text])[0]
    return ENG.Ttab[text], how


EXAMPLES = [
    "bánh mặn của Huế, có tôm",
    "món sợi nước của Nam Bộ, có mắm cá",
    "đồ uống của Hà Nội có trứng",
    "bánh gói lá dong ngày Tết",
    "món kho của Nam Bộ có cá",
    "chả giò chiên giòn",
    "Phở bò",
    "bun bo hue",
]
for _q in EXAMPLES:      # mã hoá sẵn các ví dụ (CPU, lúc khởi động) để bấm ví dụ không tốn quota GPU
    ENG.text_vec(_q)


def run(query, k, gallery, check_fid):
    query = (query or "").strip()
    if not query:
        return "<i>Nhập một truy vấn.</i>"
    _, how = query_vec(query)
    print(f"truy vấn {query!r}: vector từ {how}", flush=True)
    R = ENG.search(query, k=int(k), gallery=gallery)
    fid = {it["rank"]: ENG.fidelity(R, it) for it in R["results"][:3]} if check_fid else None
    return result_html(ENG, R, with_graph=True, fid=fid)


with gr.Blocks(title="MMKG Ẩm thực VN — truy vấn có giải thích") as demo:
    gr.Markdown(
        "## Truy vấn ảnh món ăn Việt Nam có giải thích bằng đồ thị tri thức\n"
        "Điểm = α·z(cos CLIP) + (1−α)·z(Σ_d P(món d | truy vấn)·p(món d | ảnh)). "
        "Giải thích là các số hạng của điểm đồ thị: món trung gian d*, đường đi KG từ thực thể trong truy vấn tới d*, "
        "và ảnh train có cạnh `mô_tả` → d*. "
        f"Mô hình: `{ENG.model_key}`; tham số chọn trên val: {ENG.hp}.")
    with gr.Row():
        q = gr.Textbox(label="Truy vấn (tiếng Việt, có hoặc không dấu)", value=EXAMPLES[0], scale=4)
        k = gr.Slider(1, 10, value=5, step=1, label="top-k", scale=1)
    with gr.Row():
        gal = gr.Radio(["test", "all"], value="test", label="Gallery",
                       info="test = ảnh test của seed 0 (chưa có cạnh mô_tả trong đồ thị); all = cả 316 ảnh")
        fid = gr.Checkbox(value=False, label="Kiểm tra fidelity cho top-3 (chậm hơn ~3 s)")
        btn = gr.Button("Tìm", variant="primary")
    gr.Examples(EXAMPLES, inputs=q)
    out = gr.HTML()
    btn.click(run, [q, k, gal, fid], out, api_name="run")
    q.submit(run, [q, k, gal, fid], out, api_name=False)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
                share=os.environ.get("P3_SHARE", "0") == "1")
