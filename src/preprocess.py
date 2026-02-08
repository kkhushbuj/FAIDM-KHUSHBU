import numpy as np
import pandas as pd
from .utils import header
from .config import TARGET_COL

def make_target_binary_drop1(df: pd.DataFrame, target_col: str = TARGET_COL):
    header("2) TARGET ENGINEERING — binary_drop1")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    keep = df[target_col].isin([0, 2])
    df_used = df.loc[keep].copy()
    y = df_used[target_col].replace({0: 0, 2: 1}).astype(int)
    problem_type = "binary_drop1 (0 vs 2; dropped prediabetes=1; mapped 2->1)"
    return df_used, y, problem_type

def make_features(df_used: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    X = df_used.drop(columns=[target_col], errors="ignore").copy()

    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Unexpected NaNs after conversion in columns: {bad_cols}")

    return X
