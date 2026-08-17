#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/../src/vision_tensorrt" && pwd)"

MODEL_DIR="${PACKAGE_DIR}/model"

shopt -s nullglob
MODEL_FILES=("${MODEL_DIR}"/*.pt)
shopt -u nullglob

if [[ "${#MODEL_FILES[@]}" -eq 0 ]]; then
  echo ".pt 모델 파일을 찾을 수 없습니다: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ "${#MODEL_FILES[@]}" -gt 1 ]]; then
  echo ".pt 모델 파일이 여러 개입니다:" >&2
  printf '  %s\n' "${MODEL_FILES[@]}" >&2
  exit 1
fi

MODEL_PATH="${MODEL_FILES[0]}"

if ! command -v yolo >/dev/null 2>&1; then
  echo "yolo 명령어를 찾을 수 없습니다." >&2
  exit 1
fi

cd "${PACKAGE_DIR}"
yolo export model="${MODEL_PATH}" format=engine quantize=16
