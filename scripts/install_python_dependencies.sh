#!/usr/bin/env bash

set -euo pipefail

python3 -m pip install \
  --user \
  --break-system-packages \
  --no-deps \
  transitions \
  ultralytics

python3 -m pip install \
  --user \
  --break-system-packages \
  polars \
  ultralytics-thop \
  onnx

python3 - <<'PY'
import cv2
import onnx
import tensorrt
import torch
import torchvision
import ultralytics

print("OpenCV:", cv2.__version__)
print("ONNX:", onnx.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("TensorRT:", tensorrt.__version__)
print("Ultralytics:", ultralytics.__version__)

assert torch.cuda.is_available(), \
    "Jetson GPU를 PyTorch에서 사용할 수 없습니다."
PY

