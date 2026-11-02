"""Input schema, validation, synthetic generation and provenance-tracked loading."""

from __future__ import annotations

import hashlib
import logging
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PUBLIC_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
LATENT_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))
INPUT_COLUMNS: tuple[str, ...] = ("Time", "Amount", *LATENT_COLUMNS)
LABEL_COLUMN = "Class"

Source = Literal["synthetic", "csv", "download", "auto"]
Provenance = dict[str, Any]


class DataValidationError(ValueError):
    """Raised when an input frame violates the transaction schema."""


def synthetic_transactions(rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate an offline stand-in with the public dataset's schema.

    Fraud risk is linear in two latent columns, has a thresholded interaction between
    two others, and drifts upward late in the window, so the chronological test split
    is mildly out-of-distribution, as it is in the real data.
    """
    if rows < 1_000:
        raise ValueError("Use at least 1000 synthetic rows.")
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, len(LATENT_COLUMNS)))
    times = np.sort(rng.integers(0, 2 * 86_400, rows))
    amount = np.round(rng.lognormal(3.5, 1.2, rows), 2)
    logit = (
        -5.5 + 1.2 * latent[:, 0] - latent[:, 1] + 1.5 * (latent[:, 2] * latent[:, 3] > 1.5) + 0.3 * (times > 120_000)
    )
    labels = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(latent, columns=list(LATENT_COLUMNS)).assign(Time=times, Amount=amount, Class=labels)


def validate(frame: pd.DataFrame, *, require_label: bool = True) -> pd.DataFrame:
    """Return a clean, time-sorted copy restricted to the schema columns.

    Rejects missing columns, non-numeric or non-finite values, negative times or amounts,
    and (when labels are required) any label outside {0, 1}.
    """
    columns = [*INPUT_COLUMNS, LABEL_COLUMN] if require_label else list(INPUT_COLUMNS)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    frame = frame[columns].copy()
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise DataValidationError("All schema columns must be numeric.") from error
    if not np.isfinite(values).all():
        raise DataValidationError("Input contains missing or non-finite values.")
    if (frame["Amount"] < 0).any() or (frame["Time"] < 0).any():
        raise DataValidationError("Amount and Time must be non-negative.")
    if require_label:
        if not frame[LABEL_COLUMN].isin([0, 1]).all():
            raise DataValidationError("Class must be binary (0 or 1).")
        frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)
    return frame.sort_values("Time", kind="stable").reset_index(drop=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_transactions(
    source: Source,
    csv_path: str | Path | None = None,
    rows: int = 20_000,
    seed: int = 42,
    url: str = PUBLIC_URL,
) -> tuple[pd.DataFrame, Provenance]:
    """Load raw transactions and a provenance record describing exactly where they came from.

    ``download`` fails loudly; only ``auto`` falls back to synthetic data, and it records why.
    """
    if source == "synthetic":
        return synthetic_transactions(rows, seed), {"source": "synthetic", "rows_requested": rows, "seed": seed}
    if source == "csv":
        if csv_path is None:
            raise ValueError("--source csv requires --csv PATH.")
        payload = Path(csv_path).read_bytes()
        log.info("Read %s (%.1f MB)", csv_path, len(payload) / 1e6)
        return pd.read_csv(BytesIO(payload)), {"source": "csv", "path": Path(csv_path).name, "sha256": _sha256(payload)}
    if source not in ("download", "auto"):
        raise ValueError(f"Unknown source: {source!r}")
    try:
        log.info("Downloading %s", url)
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        frame = pd.read_csv(BytesIO(payload))
        validate(frame)
        return frame, {"source": "public_download", "url": url, "sha256": _sha256(payload)}
    except Exception as error:
        if source != "auto":
            raise
        warnings.warn(f"Download failed; using SYNTHETIC data: {error}", stacklevel=2)
        return synthetic_transactions(rows, seed), {
            "source": "synthetic",
            "rows_requested": rows,
            "seed": seed,
            "fallback_reason": str(error),
        }
