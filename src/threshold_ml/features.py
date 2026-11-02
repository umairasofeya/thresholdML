"""Row-local feature engineering.

Every feature is a pure function of a single row, so features can never encode labels,
neighbouring transactions or the future. There are no customer identifiers in the data,
so no per-customer history is fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LATENT_COLUMNS

SECONDS_PER_DAY = 86_400
FEATURE_NAMES: tuple[str, ...] = (*LATENT_COLUMNS, "log_amount", "day_sin", "day_cos", "latent_norm")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Latent PCA columns plus log amount, a cyclic time-of-day encoding and the latent vector norm.

    ``Time`` is seconds since the first transaction, not wall-clock time, so the daily phase
    is an assumed periodicity rather than a verified local hour.
    """
    latent = frame[list(LATENT_COLUMNS)]
    phase = 2 * np.pi * (frame["Time"] % SECONDS_PER_DAY) / SECONDS_PER_DAY
    result = latent.copy()
    result["log_amount"] = np.log1p(frame["Amount"])
    result["day_sin"] = np.sin(phase)
    result["day_cos"] = np.cos(phase)
    result["latent_norm"] = np.sqrt((latent**2).sum(axis=1))
    return result
