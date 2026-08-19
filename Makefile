SHELL := /bin/bash

.DEFAULT_GOAL := help

PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
WORKSPACE_DIR ?= $(abspath $(PROJECT_DIR)/../..)
BUILD_PACKAGES := \
	vision_interfaces \
	vision_tensorrt \
	roa_vision_description \
	roa_vision_pipeline

.PHONY: help install install-apt install-python export build

help:
	@printf '%s\n' \
	  'make install                       apt와 Python 의존성 설치' \
	  'make install-apt                   apt 의존성만 설치' \
	  'make install-python                Python 의존성만 설치' \
	  'make export                        .pt 모델을 TensorRT 엔진으로 변환' \
	  'make build                         상위 colcon workspace에서 패키지를 순차 빌드'

install:
	@./scripts/install_apt_dependencies.sh
	@./scripts/install_python_dependencies.sh

install-apt:
	@./scripts/install_apt_dependencies.sh

install-python:
	@./scripts/install_python_dependencies.sh

export:
	@./scripts/export_engine.sh

build:
	@set -euo pipefail; \
	  workspace="$(WORKSPACE_DIR)"; \
	  if [[ ! -d "$${workspace}/src" ]]; then \
	    echo "colcon workspace의 src 디렉터리를 찾을 수 없습니다: $${workspace}/src" >&2; \
	    exit 1; \
	  fi; \
	  cd "$${workspace}"; \
	  echo "Workspace: $${workspace}"; \
	  for package in $(BUILD_PACKAGES); do \
	    echo "==> Building $${package}"; \
	    colcon build --symlink-install --packages-select "$${package}"; \
	    source install/setup.bash; \
	  done
