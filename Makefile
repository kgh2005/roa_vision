SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help install install-apt install-python export

help:
	@printf '%s\n' \
	  'make install                       apt와 Python 의존성 설치' \
	  'make install-apt                   apt 의존성만 설치' \
	  'make install-python                Python 의존성만 설치' \
	  'make export                        .pt 모델을 TensorRT 엔진으로 변환'

install:
	@./scripts/install_apt_dependencies.sh
	@./scripts/install_python_dependencies.sh

install-apt:
	@./scripts/install_apt_dependencies.sh

install-python:
	@./scripts/install_python_dependencies.sh

export:
	@./scripts/export_engine.sh
