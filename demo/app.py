# -*- coding: utf-8 -*-
"""Giao diện nhỏ (Gradio): nhập truy vấn → top-k kết quả + quan hệ/đường đi giải thích, trên 3 bộ dữ liệu:

  • MMKG Ẩm thực Việt Nam (tự xây, tiếng Việt)   — demo/mmkg_engine.py (tính lại từ mmkg/ khi khởi động)
  • Action Genome (phần A, tiếng Anh)              — demo/partA_engine.py + bundle bundles/action_genome
  • MSVD (phần C, tiếng Anh)                       — demo/partC_engine.py + bundle bundles/msvd
Bundle AG / MSVD do notebook phần A / C xuất trên Kaggle (xem demo/README.md); thiếu bundle thì tab tương ứng báo rõ.

Chạy:  python demo/app.py            (mở http://127.0.0.1:7860)
       P3_SHARE=1 python demo/app.py (thêm link công khai tạm thời *.gradio.live, ~72 giờ)
Trên Hugging Face Spaces: file này là entry point (xem demo/README.md).
"""
import os, sys, traceback
from pathlib import Path

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

import torch
import torch.nn.functional as F
import gradio as gr
from mmkg_engine import MMKGSearch
from viz import result_html, bundle_result_html

BUNDLES = Path(os.environ.get("P3_BUNDLES") or (Path(HERE) / "bundles"))

# ------------------------------------------------------------------ MMKG
ENG = MMKGSearch()
ENG._load_clip()   # bộ mã hoá CPU: dùng khi chạy trên máy và làm dự phòng khi hết quota GPU


class ClipText:
    """Bộ mã hoá văn bản CLIP dùng chung: cache → GPU (ZeroGPU) → CPU."""

    def __init__(self, model, tok, tokenize):
        self.cpu, self.tok, self.tokenize, self.gpu, self.cache = model, tok, tokenize, None, {}

    def to_gpu(self):
        import copy
        self.gpu = copy.deepcopy(self.cpu).to("cuda")    # ZeroGPU: .to("cuda") lúc khởi động là hợp lệ

    @torch.no_grad()
    def encode_cpu(self, text):
        t = self.tokenize(self.tok, [text])
        return F.normalize(self.cpu.encode_text(t).float(), dim=-1)[0].numpy()


ENCODERS = {"mmkg": ClipText(ENG._clip[0], ENG._clip[2], ENG._tokenize)}

# ------------------------------------------------------------------ AG + MSVD (bundle từ Kaggle)
AG = MSVD = None
LOAD_ERR = {}
_bundle_ag, _bundle_msvd = BUNDLES / "action_genome", BUNDLES / "msvd"
if (_bundle_ag / "meta.json").exists() or (_bundle_msvd / "bundle.pt").exists():
    try:
        import open_clip
        _m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
        ENCODERS["vitl"] = ClipText(_m.eval(), open_clip.get_tokenizer("ViT-L-14"), ENG._tokenize)
    except Exception as ex:
        LOAD_ERR["clip"] = repr(ex)
enc_l = ENCODERS.get("vitl")
if enc_l is not None and (_bundle_ag / "meta.json").exists():
    try:
        from partA_engine import PartABundle
        AG = PartABundle(_bundle_ag)
        assert AG.meta["clip"] == ["ViT-L-14", "laion2b_s32b_b82k"], AG.meta["clip"]
    except Exception as ex:
        traceback.print_exc(); LOAD_ERR["ag"] = repr(ex)
if enc_l is not None and (_bundle_msvd / "bundle.pt").exists():
    try:
        from partC_engine import PartCBundle
        MSVD = PartCBundle(_bundle_msvd)
        assert list(MSVD.meta["clip"]) == ["ViT-L-14", "laion2b_s32b_b82k"], MSVD.meta["clip"]
    except Exception as ex:
        traceback.print_exc(); LOAD_ERR["msvd"] = repr(ex)

if ZERO_GPU:
    try:
        for e in ENCODERS.values():
            e.to_gpu()
    except Exception as ex:
        print("Không chuyển được CLIP sang GPU, dùng CPU:", repr(ex)[:200], flush=True)
        ZERO_GPU = False

if "spaces" in sys.modules:     # Space ZeroGPU bắt buộc có ít nhất một hàm @spaces.GPU
    @spaces.GPU(duration=10)
    def _encode_gpu(which, text):
        e = ENCODERS[which]
        with torch.no_grad():
            t = e.tokenize(e.tok, [text]).to("cuda")
            return F.normalize(e.gpu.encode_text(t).float(), dim=-1)[0].cpu().numpy()


def query_vec(which, text, cache=None):
    """Vector CLIP của truy vấn: cache → GPU (ZeroGPU) → CPU."""
    cache = ENCODERS[which].cache if cache is None else cache
    if text in cache:
        return cache[text], "cache"
    how = "cpu"
    if ZERO_GPU and ENCODERS[which].gpu is not None:
        try:
            cache[text] = _encode_gpu(which, text); how = "gpu"
        except Exception as ex:               # hết quota / hàng đợi GPU: dùng CPU
            print("ZeroGPU không dùng được, mã hoá bằng CPU:", repr(ex)[:200], flush=True)
    if text not in cache:
        cache[text] = ENCODERS[which].encode_cpu(text)
    return cache[text], how


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
EX_AG = ["a person drinking from a glass while sitting on the couch", "someone watching tv and holding a phone",
         "a person opens the door and walks in", "someone is cooking food in the kitchen",
         "a person holding a book and reading on the bed", "a person takes a towel then wipes the table"]
EX_MSVD = ["a man is playing a guitar", "a woman is slicing a tomato", "a dog is running on the beach",
           "a man is riding a horse", "a cat is playing with a toy", "two people are dancing on a stage"]


def _prefill(engine, which):
    """Vector truy vấn của bộ tham chiếu (tính trên Kaggle) + ví dụ: bấm ví dụ không tốn quota GPU."""
    import json
    ref = engine.dir / "reference_outputs.json"
    if ref.exists():
        for o in json.load(open(ref)):
            if o.get("q_vec"):
                ENCODERS[which].cache.setdefault(o["query"], o["q_vec"])


for _e in (AG, MSVD):
    if _e is not None:
        _prefill(_e, "vitl")
for _q in (EX_AG if AG else []) + (EX_MSVD if MSVD else []):
    query_vec("vitl", _q)


# ------------------------------------------------------------------ hàm xử lý
def run(query, k, gallery, check_fid):
    query = (query or "").strip()
    if not query:
        return "<i>Nhập một truy vấn.</i>"
    _, how = query_vec("mmkg", query, cache=ENG.Ttab)
    print(f"[mmkg] truy vấn {query!r}: vector từ {how}", flush=True)
    R = ENG.search(query, k=int(k), gallery=gallery)
    fid = {it["rank"]: ENG.fidelity(R, it) for it in R["results"][:3]} if check_fid else None
    return result_html(ENG, R, with_graph=True, fid=fid)


def _missing(name, key):
    return (f"<div style='color:#a00'>Chưa có bundle {name} (thư mục <code>{BUNDLES}</code>). "
            f"Xem demo/README.md để xuất bundle từ notebook trên Kaggle. {LOAD_ERR.get(key, '')}</div>")


def run_ag(query, k):
    query = (query or "").strip()
    if AG is None:
        return _missing("Action Genome", "ag")
    if not query:
        return "<i>Nhập một truy vấn (tiếng Anh).</i>"
    q, how = query_vec("vitl", query)
    print(f"[ag] truy vấn {query!r}: vector từ {how}", flush=True)
    return bundle_result_html(AG.search(query, k=int(k), q_vec=q), "ag")


def run_msvd(query, k):
    query = (query or "").strip()
    if MSVD is None:
        return _missing("MSVD", "msvd")
    if not query:
        return "<i>Nhập một truy vấn (tiếng Anh).</i>"
    q, how = query_vec("vitl", query)
    print(f"[msvd] truy vấn {query!r}: vector từ {how}", flush=True)
    return bundle_result_html(MSVD.search(query, k=int(k), q_vec=q), "msvd")


# ------------------------------------------------------------------ giao diện
with gr.Blocks(title="Project 3 — truy vấn đa phương thức có giải thích") as demo:
    gr.Markdown("## Truy vấn đa phương thức có giải thích bằng đồ thị\n"
                "Nhập truy vấn → top-k kết quả, mỗi kết quả kèm điểm (CLIP + đồ thị) và quan hệ / đường đi đã đóng góp "
                "vào điểm. Giải thích lấy từ chính các số hạng của điểm đồ thị, không sinh sau khi đã có kết quả.")
    with gr.Tab("MMKG Ẩm thực Việt Nam"):
        gr.Markdown(
            "Điểm = α·z(cos CLIP) + (1−α)·z(Σ_d P(món d | truy vấn)·p(món d | ảnh)). "
            "Giải thích: món trung gian d*, đường đi KG từ thực thể trong truy vấn tới d*, ảnh train có cạnh `mô_tả` → d*. "
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
    with gr.Tab("Action Genome"):
        gr.Markdown("Video Charades + đồ thị cảnh Action Genome (phần A). Điểm = α·z(cos) + (1−α)·(λ·z(s_GNN) + "
                    "(1−λ)·z(s_đường_đi)); giải thích = đường đi video → đối tượng → khái niệm truy vấn có đóng góp biên "
                    "lớn nhất vào s_đường_đi. " + (f"Gallery test {len(AG.items)} video; α={AG.alpha}, λ={AG.lam}."
                                                   if AG else "**Chưa có bundle.**"))
        with gr.Row():
            q2 = gr.Textbox(label="Truy vấn (tiếng Anh)", value=EX_AG[0], scale=4)
            k2 = gr.Slider(1, 10, value=5, step=1, label="top-k", scale=1)
            b2 = gr.Button("Tìm", variant="primary", scale=1)
        gr.Examples(EX_AG, inputs=q2)
        o2 = gr.HTML()
        b2.click(run_ag, [q2, k2], o2, api_name="run_ag")
        q2.submit(run_ag, [q2, k2], o2, api_name=False)
    with gr.Tab("MSVD"):
        gr.Markdown("Video MSVD + đồ thị khái niệm (phần C, mô hình G3 = QC-RGAT, attention theo truy vấn trên cạnh quan "
                    "hệ). Điểm = α·z(cos) + (1−α)·z(s_đồ_thị); giải thích = các số hạng đồ thị dương lớn nhất trong phân rã "
                    "chính xác của s_đồ_thị (video –chứa→ khái niệm, đường đi quan hệ). " +
                    (f"Gallery {MSVD.meta['part']} ({len(MSVD.items)} video); α={MSVD.alpha}." if MSVD else "**Chưa có bundle.**"))
        with gr.Row():
            q3 = gr.Textbox(label="Truy vấn (tiếng Anh)", value=EX_MSVD[0], scale=4)
            k3 = gr.Slider(1, 10, value=5, step=1, label="top-k", scale=1)
            b3 = gr.Button("Tìm", variant="primary", scale=1)
        gr.Examples(EX_MSVD, inputs=q3)
        o3 = gr.HTML()
        b3.click(run_msvd, [q3, k3], o3, api_name="run_msvd")
        q3.submit(run_msvd, [q3, k3], o3, api_name=False)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
                share=os.environ.get("P3_SHARE", "0") == "1")
