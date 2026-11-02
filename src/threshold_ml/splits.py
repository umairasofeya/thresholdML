"""Chronological train/validation/test partitioning."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .data import LABEL_COLUMN


class ChronologicalSplit(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame: pd.DataFrame, fractions: tuple[float, float] = (0.6, 0.8)) -> ChronologicalSplit:
    """Partition by *unique timestamps* so rows sharing a timestamp never straddle a boundary.

    ``fractions`` are the cumulative cut points over the sorted unique timestamps. Row
    proportions therefore differ slightly from the nominal 60/20/20. Every partition must
    contain both classes, or threshold selection and evaluation would be meaningless.
    """
    first_cut, second_cut = fractions
    if not 0 < first_cut < second_cut < 1:
        raise ValueError("fractions must satisfy 0 < first < second < 1.")
    times = np.sort(frame["Time"].unique())
    if len(times) < 10:
        raise ValueError("At least 10 unique timestamps are required.")
    first, second = times[int(len(times) * first_cut)], times[int(len(times) * second_cut)]
    split = ChronologicalSplit(
        train=frame[frame["Time"] < first],
        validation=frame[(frame["Time"] >= first) & (frame["Time"] < second)],
        test=frame[frame["Time"] >= second],
    )
    for name, part in split._asdict().items():
        if part[LABEL_COLUMN].nunique() != 2:
            raise ValueError(f"The {name} partition needs both classes; got {part[LABEL_COLUMN].unique().tolist()}.")
    return split
