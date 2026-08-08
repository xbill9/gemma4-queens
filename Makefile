# Root Makefile for gemma4-queens
# Coordinates actions across all Gemma 4 devops-agent sub-projects

# Sub-project directories. Keep in sync with PROJECTS in menu.py.
# MCP devops agents (operators — they drive vLLM in a container):
PROJECTS = tpu-12B-v6e1-devops-agent \
           tpu-2B-v5e1-devops-agent \
           g2-4-2B-qat-L4-devops-agent \
           gpu-4B-cloudrun-devops-agent \
           gpu-2B-L4-ec2-agent \
           gpu-4B-inf-devops-agent
# The pure-JAX inference engines (tpu-jax, tpu-jax-inf2) were split out of this
# repo — they live at github.com/xbill9/tpu-jax and .../tpu-jax-inf2.

.PHONY: all help menu launch demos submodules install test lint format clean

# Default target
all: help

help:
	@echo "======================================================================"
	@echo "                      gemma4-queens Root Makefile"
	@echo "======================================================================"
	@echo "Available targets:"
	@echo "  make menu         - Launch the interactive terminal menu (menu.py)"
	@echo "  make launch       - Pick a project + demo and run it in its directory"
	@echo "  make demos        - List every project and demo ./launch can run"
	@echo "  make submodules   - Initialize and update git submodules"
	@echo "  make install      - Install python dependencies in all sub-projects"
	@echo "  make lint         - Run linters (ruff, mypy) across all sub-projects"
	@echo "  make format       - Format code (ruff format) across all sub-projects"
	@echo "  make test         - Run test suites across all sub-projects"
	@echo "  make clean        - Clean pycache, log, and lint cache files everywhere"
	@echo "======================================================================"

menu:
	python3 menu.py

# ARGS lets you skip the picker: make launch ARGS="tpu-12B 1"
launch:
	./launch $(ARGS)

demos:
	@./launch --list

submodules:
	@echo "Initializing and updating git submodules..."
	git submodule update --init --recursive

install:
	@for dir in $(PROJECTS); do \
		if [ -f $$dir/Makefile ]; then \
			echo "Installing dependencies in $$dir..."; \
			$(MAKE) -C $$dir install; \
		elif [ -f $$dir/requirements.txt ]; then \
			echo "Installing dependencies in $$dir via requirements.txt..."; \
			pip install -r $$dir/requirements.txt; \
		fi; \
	done

lint:
	@for dir in $(PROJECTS); do \
		if [ -f $$dir/Makefile ]; then \
			echo "Linting in $$dir..."; \
			$(MAKE) -C $$dir lint || exit 1; \
		fi; \
	done

# Only some sub-projects define a `format` target; skip the ones that don't
# rather than aborting the whole sweep on "No rule to make target".
format:
	@for dir in $(PROJECTS); do \
		if [ ! -f $$dir/Makefile ]; then \
			continue; \
		elif $(MAKE) -C $$dir -n format >/dev/null 2>&1; then \
			echo "Formatting in $$dir..."; \
			$(MAKE) -C $$dir format; \
		else \
			echo "Skipping $$dir (no format target)."; \
		fi; \
	done

test:
	@for dir in $(PROJECTS); do \
		if [ -f $$dir/Makefile ]; then \
			echo "Running tests in $$dir..."; \
			$(MAKE) -C $$dir test || true; \
		fi; \
	done

clean:
	@echo "Cleaning up root directory..."
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type f -name "*.pyc" -delete
	@for dir in $(PROJECTS); do \
		if [ -f $$dir/Makefile ]; then \
			echo "Cleaning in $$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
