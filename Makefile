PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
DATA   := data/creditcard.csv
URL    := https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
SHA256 := 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89

.PHONY: install lint test demo reproduce clean

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -r requirements-lock.txt
	$(BIN)/pip install -e ".[dev]" --no-deps

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

test:
	$(BIN)/pytest

demo:
	$(BIN)/threshold-ml -v run --source synthetic --output results/synthetic

$(DATA):
	mkdir -p data
	curl --fail --location --retry 2 $(URL) --output $@
	echo "$(SHA256)  $@" | shasum -a 256 -c

reproduce: $(DATA)
	$(BIN)/threshold-ml -v run --source csv --csv $(DATA) --output examples/public
	$(BIN)/threshold-ml -v run --source synthetic --output examples/synthetic

clean:
	rm -rf results .pytest_cache .ruff_cache src/*.egg-info
