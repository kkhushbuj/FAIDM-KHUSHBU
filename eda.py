import pandas as pd
from .utils import header

def basic_eda(df: pd.DataFrame, target_col: str) -> dict:
    header("1) BASIC EDA")

    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_counts": df.isna().sum().to_dict(),
    }

    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    missing = df.isna().sum()
    missing_nonzero = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values:")
    print(missing_nonzero if len(missing_nonzero) else "No missing values detected.")

    if target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        print(f"\nTarget distribution ({target_col}):")
        print(vc)
        print("\nTarget distribution (%):")
        print((vc / len(df) * 100).round(3))
        info["target_counts"] = vc.to_dict()
    else:
        print(f"\nWARNING: {target_col} not found.")

    print("\nQuick numeric stats:")
    print(df.describe().T[["mean", "std", "min", "max"]].round(3))

    return info
