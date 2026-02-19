SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

ORT_REPO ?= https://github.com/microsoft/onnxruntime.git
ORT_REF ?= main
ORT_DIR ?= .ort
BUILD_DIR ?= build
ONNXRUNTIME_INCLUDE_DIR ?= $(ORT_DIR)/include/onnxruntime/core/session
TELUM_EP_ENABLE_ZDNN ?= OFF

CMAKE ?= cmake
PYTHON ?= python3

.PHONY: help ort-headers configure build rebuild python-sdist ci-local clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "; printf "\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2} END {printf "\n"}' $(MAKEFILE_LIST)

ort-headers: ## Fetch ONNX Runtime headers into $(ORT_DIR)
	@if [ ! -d "$(ORT_DIR)/.git" ]; then \
		git clone --depth 1 --branch "$(ORT_REF)" "$(ORT_REPO)" "$(ORT_DIR)"; \
	else \
		git -C "$(ORT_DIR)" fetch --depth 1 origin "$(ORT_REF)"; \
		git -C "$(ORT_DIR)" checkout --force FETCH_HEAD; \
	fi

configure: ort-headers ## Configure CMake build directory
	$(CMAKE) -S . -B "$(BUILD_DIR)" -G Ninja \
		-DONNXRUNTIME_INCLUDE_DIR="$(ONNXRUNTIME_INCLUDE_DIR)" \
		-DTELUM_EP_ENABLE_ZDNN="$(TELUM_EP_ENABLE_ZDNN)"

build: configure ## Build the Telum plugin EP
	$(CMAKE) --build "$(BUILD_DIR)" --parallel

rebuild: clean build ## Clean and rebuild

python-sdist: ## Build Python source distribution into dist/
	$(PYTHON) -m pip install --upgrade pip build
	$(PYTHON) -m build --sdist

ci-local: python-sdist build ## Run local equivalents of baseline CI jobs

clean: ## Remove local build artifacts
	rm -rf "$(BUILD_DIR)" dist
