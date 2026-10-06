"""Time helpers. pandas may store datetimes at us or ns resolution; engine math always uses int64 ns."""
from __future__ import annotations

import pandas as pd


def ns(x) -> "pd.Index":
    """Datetimes (Index/Series/array) -> int64 nanoseconds since epoch, whatever the stored resolution."""
    return pd.DatetimeIndex(x).as_unit("ns").asi8
