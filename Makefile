SHELL := /bin/bash


PYTHON_VERSION := 3.11.3
PYENV_VERSION_INSTALLED := $(shell pyenv versions --bare | grep -e "^$(PYTHON_VERSION)$$")

.PHONY: setup
setup:
	@if [ -z "$(PYENV_VERSION_INSTALLED)" ]; then \
		echo "Installing Python $(PYTHON_VERSION) with pyenv..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	pyenv local $(PYTHON_VERSION)
	@echo "=========|| The pyenv has successfully curated Python version $(PYTHON_VERSION) locally ||========="
	python -m venv .venv 
	@echo "=========|| The environment has been curated successfully ||========="
	.venv/bin/python -m pip install --upgrade pip 
	@echo "=========|| The pipe has been updated successfully ||========="
	.venv/bin/python -m pip install -r requirements.txt 
	@echo "=========|| The requirements have been installed successfully ||========="

.PHONY: setup-notebooks
setup-notebooks:
	@echo "=========|| Setting up notebook tools and git hooks ||========="
	@echo "Installing nbdime, nbstripout, and pre-commit globally (user site-packages)..."
	python -m pip install --user --upgrade nbdime nbstripout pre-commit
	@echo "Configuring git for notebook diffs..."
	python -m nbdime config-git --enable
	@echo "Installing nbstripout git filter..."
	python -m nbstripout --install
	@echo "Installing pre-commit hooks..."
	@python -m pre_commit.main install || pre-commit install
	@echo "=========|| Done. Notebook diffs/merges enabled, outputs stripped, pre-commit installed. ||========="
	@echo "Note: On Windows, ensure Python Scripts directory is in your PATH"
	@echo "      (e.g., %%APPDATA%%\\Python\\PythonXX\\Scripts or %%LOCALAPPDATA%%\\Programs\\Python\\PythonXX\\Scripts)"

