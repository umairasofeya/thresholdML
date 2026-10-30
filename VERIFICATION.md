# Rebuild verification — 2026-10-30

Reconstructed the deleted project and reran the unchanged modeling recipe against the retained real dataset. Environment: Python 3.13; full package versions in requirements-lock.txt.

- `python -m pytest -q`: **12 passed**.
- Real-data pipeline completed on all 284,807 rows using seed 42 and the fixed model settings in README.
- Synthetic smoke pipeline completed on 20,000 rows using seed 42.
- Both runs generated model bundles, test prediction CSVs, threshold curves, results.json, REPORT.md and evaluation.png.
- Independently checked the real input hash and counts, strict time boundaries, both reloaded models, saved prediction CSVs and every recomputed test metric. All checks passed.
- Visually inspected the regenerated real-data chart.

Public acquisition details and the input SHA-256 are recorded in examples/public/ACQUISITION.md. Raw source data is excluded from the deliverable. Costs remain hypothetical and results are from a single historical holdout.
