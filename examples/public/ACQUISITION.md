# Dataset acquisition

Downloaded on 2026-10-29 from https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv using curl with normal HTTPS certificate verification. Python's local certificate configuration prevented urllib from downloading, so the verified file was loaded through `--source csv`. The raw file was retained in the working directory and reused for the rebuild on 2026-10-30.

SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

Rows: 284,807; frauds: 492. Raw source data is not included in the ZIP. See README for acquisition and reproduction commands. Downloading the same bytes and using seed 42 with the locked environment reproduces the benchmark subject to platform floating point differences.
