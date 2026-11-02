# Dataset acquisition

Source: <https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv>, fetched with `curl` under normal HTTPS certificate verification and loaded via `--source csv`.

| | |
|---|---|
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Rows | 284,807 |
| Frauds | 492 |

The raw file is not redistributed here; review the source terms before doing so. `make reproduce` downloads it, verifies the hash and regenerates this directory. With the same bytes, seed 42 and `requirements-lock.txt`, results are reproducible up to platform floating-point differences.
