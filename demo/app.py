# -*- coding: utf-8 -*-
"""Giao diện nhỏ (Gradio): nhập truy vấn → top-k ảnh + quan hệ/đường đi giải thích.

Chạy:  python demo/app.py            (mở http://127.0.0.1:7860)
       P3_SHARE=1 python demo/app.py (thêm link công khai tạm thời *.gradio.live, ~72 giờ)
Trên Hugging Face Spaces: file này là entry point (xem demo/README.md).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Hugging Face ZeroGPU: gói `spaces` có sẵn trên Space; Space ZeroGPU bắt buộc có ít nhất một hàm @spaces.GPU.
# Mô hình B/32 chạy đủ nhanh trên CPU, nên engine giữ trên CPU (P3_DEVICE=cpu) và chỉ hàm xử lý truy vấn được
# đánh dấu @spaces.GPU. Mỗi truy vấn tốn vài trăm ms (fidelity ~3 s) trong quota ZeroGPU của người xem.
try:
    import spaces
    os.environ.setdefault("P3_DEVICE", "cpu")
    gpu = spaces.GPU(duration=60)
    # ZeroGPU giả lập CUDA: optimizer.step() trên CPU vẫn gọi kiểm tra accelerator và làm khởi tạo CUDA thật
    # (RuntimeError "Low-level CUDA init"). Bình thường GNN lấy từ cache nên không huấn luyện; đây là dự phòng.
    import torch
    torch.optim.Optimizer._accelerator_graph_capture_health_check = lambda self: None
except ImportError:          # chạy trên máy / Colab: không cần ZeroGPU
    gpu = lambda f: f

import gradio as gr
from mmkg_engine import MMKGSearch
from viz import result_html

ENG = MMKGSearch()
ENG._load_clip()   # nạp sẵn bộ mã hoá văn bản lúc khởi động (hàm @spaces.GPU chạy ở tiến trình con)

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


@gpu
def run(query, k, gallery, check_fid):
    query = (query or "").strip()
    if not query:
        return "<i>Nhập một truy vấn.</i>"
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
    btn.click(run, [q, k, gal, fid], out)
    q.submit(run, [q, k, gal, fid], out)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
                share=os.environ.get("P3_SHARE", "0") == "1")
