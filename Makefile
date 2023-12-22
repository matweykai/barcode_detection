CALL_CMD=PYTHONPATH=. python
ACTIVATE_VENV=source .venv/bin/activate

SHELL := /bin/bash
.ONESHELL:

setup:
	python3 -m venv .venv
	$(ACTIVATE_VENV)

	pip install -r requirements.txt
	dvc install
	dvc pull
	clearml-init