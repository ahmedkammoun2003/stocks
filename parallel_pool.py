"""Process pool helpers: load the shared DataFrame once per worker."""
import pickle
from typing import Any, Optional

import pandas as pd

_SHARED_DF: Optional[pd.DataFrame] = None


def init_pool_dataframe(df_bytes: bytes) -> None:
    """Unpickle the dataset once when a worker process starts."""
    global _SHARED_DF
    _SHARED_DF = pickle.loads(df_bytes)


def get_shared_dataframe() -> pd.DataFrame:
    if _SHARED_DF is None:
        raise RuntimeError("Worker dataframe not initialised — pool initializer missing?")
    return _SHARED_DF


def pickle_dataframe(df: pd.DataFrame) -> bytes:
    """Serialise once in the parent; each worker unpickles a single copy."""
    return pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
