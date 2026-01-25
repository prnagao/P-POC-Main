.PHONY: setup dryrun baseline adapter finetune smoke

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

dryrun:
	python -m src.pathology_poc.train --data_root dataset --dry-run

baseline:
	python -m src.pathology_poc.train --data_root dataset --epochs 1 --batch_size 4 --freeze_backbone

adapter:
	python -m src.pathology_poc.adapter_train --data_root dataset --epochs 3 --batch_size 4

finetune:
	python -m src.pathology_poc.train --data_root dataset --epochs 5 --batch_size 4 --no-freeze_backbone

smoke:
	python scripts/smoke_test.py
