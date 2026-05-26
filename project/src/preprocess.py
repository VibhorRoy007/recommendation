"""Data preprocessing for creator biographies.

- Loads cleaned creators JSON (output from filter_profiles).
- Removes duplicate usernames and duplicate biographies.
- Drops entries with null/empty biographies.
- Normalizes whitespace (strip, collapse multiple spaces).
- Performs light emoji/noise cleaning while preserving meaningful text.
- Saves the final dataframe as CSV and Parquet.
"""

import json
import pandas as pd
import re
from pathlib import Path

# Paths (adjust if needed)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INPUT_JSON = Path(__file__).resolve().parents[2] / "creators_clean_updated.json"
OUTPUT_CSV = DATA_DIR / "processed_creators.csv"
OUTPUT_PARQUET = DATA_DIR / "processed_creators.parquet"

def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_bio(bio: str) -> str:
    # Strip leading/trailing whitespace
    bio = bio.strip()
    # Collapse multiple whitespace characters into a single space
    bio = re.sub(r"\s+", " ", bio)
    # Remove excessive emoji sequences (keep if length > 1 emoji?)
    # Simple heuristic: remove runs of >3 consecutive emoji-like characters.
    # Emoji are captured by the "\p{Emoji}" Unicode property via regex module; fallback to a basic range.
    try:
        import regex as reg
        bio = reg.sub(r"[\p{Emoji}]{4,}", "", bio)
    except Exception:
        # If regex not available, skip this step.
        pass
    return bio

def main():
    raw = load_json(INPUT_JSON)
    df = pd.DataFrame(raw)
    # Keep only needed columns (username, biography, followersCount)
    df = df[["username", "biography", "followersCount"]].copy()
    # Drop rows with null/empty biography
    df["biography"] = df["biography"].astype(str)
    df = df[df["biography"].str.strip().astype(bool)]
    # Clean biography text
    df["biography"] = df["biography"].apply(clean_bio)
    # Remove duplicate usernames (keep first)
    df = df.drop_duplicates(subset="username", keep="first")
    # Remove duplicate biographies (keep first)
    df = df.drop_duplicates(subset="biography", keep="first")
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Save CSV and Parquet
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Processed {len(df)} creators saved to {OUTPUT_CSV} and {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
