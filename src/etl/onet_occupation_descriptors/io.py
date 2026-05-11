import pandas as pd
from pathlib import Path


def load_csv_df(base_dir: Path, filename: str) -> pd.DataFrame:
    """Load a CSV file from a directory into a DataFrame."""
    return pd.read_csv(base_dir / filename)


def save_csv_df(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    """Save a DataFrame to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)
