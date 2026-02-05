import pandas as pd
from typing import Any, Optional
from datetime import datetime

def normalize_date(date_str: Any) -> Optional[datetime]:
    """
    Attempts to normalize a date string into a datetime object.
    Handles multiple formats.
    """
    if pd.isna(date_str) or date_str == "":
        return None
        
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception:
        return None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs general cleaning on a dataframe.
    - Strips whitespace from string columns.
    """
    df = df.copy()
    for col in df.select_dtypes(['object']):
        df[col] = df[col].str.strip()
    return df
