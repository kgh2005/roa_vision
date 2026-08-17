#!/usr/bin/env bash

set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  libnvinfer-dev \
  libopencv-dev
