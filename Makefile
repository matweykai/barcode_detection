CALL_CMD=PYTHONPATH=. python
ACTIVATE_VENV=source .venv/bin/activate
CONFIG_PATH=configs/base_config.yaml

SHELL := /bin/bash
.ONESHELL:

setup:
	python3 -m venv .venv
	$(ACTIVATE_VENV)

	pip install -r requirements.txt
	dvc install
	dvc pull
	clearml-init

train:
	$(ACTIVATE_VENV)
	$(CALL_CMD) src/train.py $(CONFIG_PATH)

EXPORT_MODEL_PATH=weights/yolo_8n_994.pt
export_onnx:
	$(ACTIVATE_VENV)
	$(CALL_CMD) src/export.py $(EXPORT_MODEL_PATH)
