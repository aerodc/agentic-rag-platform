.PHONY: ingest query eval test

ingest:
	python -m src.ingest.cli --source data/docs --collection main

query:
	python -m src.retrieval.cli --collection main --q "$(Q)"

test:
	pytest -q
