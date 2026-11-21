#!/usr/bin/env bash
# Đóng gói thư mục cho Hugging Face Space (Gradio, ZeroGPU). P3_WITH_BUNDLES=1 để chép cả demo/bundles/.
#   bash demo/build_space.sh [thư_mục_đích]     (mặc định: build/hf_space)
# Sau đó upload:  hf upload <user>/<tên-space> build/hf_space . --repo-type space
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/hf_space}"
rm -rf "$OUT"; mkdir -p "$OUT/mmkg" "$OUT/results/mmkg" "$OUT/demo/cache"
cp "$ROOT/demo/space_template/README.md" "$ROOT/demo/space_template/requirements.txt" "$OUT/"
cp "$ROOT/demo/app.py" "$ROOT/demo/mmkg_engine.py" "$ROOT/demo/viz.py" "$ROOT/demo/partA_engine.py" \
   "$ROOT/demo/partC_engine.py" "$ROOT/demo/partC_lib.py" "$OUT/"
cp "$ROOT/mmkg/kg_data.py" "$ROOT/mmkg/images.csv" "$ROOT/mmkg/DATA_CARD.md" "$OUT/mmkg/"
cp -r "$ROOT/mmkg/images" "$OUT/mmkg/"
cp "$ROOT/results/mmkg/splits.json" "$ROOT/results/mmkg/queries.json" "$OUT/results/mmkg/"
# đặc trưng CLIP đã trích (nếu có) để Space khởi động nhanh, khỏi mã hoá lại 316 ảnh,
# và GNN đã huấn luyện (seed 0) để Space không phải chạy optimizer lúc khởi động
for f in feats_mclip_b32.npz gnn_mclip_b32_seed0.pt; do
  if [ -f "$ROOT/demo/cache/$f" ]; then cp "$ROOT/demo/cache/$f" "$OUT/demo/cache/"; else echo "Chưa có demo/cache/$f — chạy 'python demo/app.py' một lần để tạo"; fi
done
# bundle Action Genome / MSVD (xuất trên Kaggle, KHÔNG nằm trong git): chỉ chép khi có sẵn ở demo/bundles/
# (python demo/fetch_bundles.py). Space giữ bundle đã upload trước đó vì upload_folder không xoá file cũ.
if [ -d "$ROOT/demo/bundles" ] && [ "${P3_WITH_BUNDLES:-0}" = "1" ]; then cp -r "$ROOT/demo/bundles" "$OUT/bundles"; fi
du -sh "$OUT"; echo "Đã đóng gói vào $OUT"
