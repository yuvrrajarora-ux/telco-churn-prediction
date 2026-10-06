"""Loading and cleaning."""
import pandas as pd

from .config import DATA_PATH, ID_COL, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate_data` or add the Kaggle CSV."
        )
    return clean(pd.read_csv(path))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[ID_COL], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]
