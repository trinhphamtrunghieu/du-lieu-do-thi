#!/usr/bin/env bash
# Đóng gói thư mục cho Hugging Face Space (Gradio, CPU miễn phí).
#   bash demo/build_space.sh [thư_mục_đích]     (mặc định: build/hf_space)
# Sau đó upload:  hf upload <user>/<tên-space> build/hf_space . --repo-type space
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/hf_space}"
rm -rf "$OUT"; mkdir -p "$OUT/mmkg" "$OUT/results/mmkg" "$OUT/demo/cache"
cp "$ROOT/demo/space_template/README.md" "$ROOT/demo/space_template/requirements.txt" "$OUT/"
cp "$ROOT/demo/app.py" "$ROOT/demo/mmkg_engine.py" "$ROOT/demo/viz.py" "$OUT/"
cp "$ROOT/mmkg/kg_data.py" "$ROOT/mmkg/images.csv" "$ROOT/mmkg/DATA_CARD.md" "$OUT/mmkg/"
cp -r "$ROOT/mmkg/images" "$OUT/mmkg/"
cp "$ROOT/results/mmkg/splits.json" "$ROOT/results/mmkg/queries.json" "$OUT/results/mmkg/"
# đặc trưng CLIP đã trích (nếu có) để Space khởi động nhanh, khỏi mã hoá lại 316 ảnh
if [ -f "$ROOT/demo/cache/feats_mclip_b32.npz" ]; then cp "$ROOT/demo/cache/feats_mclip_b32.npz" "$OUT/demo/cache/"; fi
du -sh "$OUT"; echo "Đã đóng gói vào $OUT"
