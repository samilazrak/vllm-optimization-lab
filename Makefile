# Les cibles setup/serve/bench/sweep tournent sur le pod GPU.
# Les cibles collect/report tournent partout, sur les JSON versionnés.
PY := PYTHONPATH=src python

.PHONY: setup serve bench sweep stop collect report lint

CONFIG ?= base
DATASET ?= sharegpt

setup:
	./scripts/00-setup.sh

serve:
	./scripts/01-serve.sh $(CONFIG)

bench:
	./scripts/02-bench.sh $(CONFIG) $(DATASET)

sweep:
	./scripts/03-sweep.sh

stop:
	./scripts/01-serve.sh stop

collect:
	$(PY) -m lab.collect

report:
	$(PY) -m lab.report

lint:
	ruff check src && ruff format --check src
