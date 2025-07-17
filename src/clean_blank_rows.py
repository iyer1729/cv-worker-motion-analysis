import os
import re
import pandas as pd

# Path to the dataset
ROOT = "/Users/hariiyer/Desktop/data/UCF Sports Action"

# If True, cleaned files are saved with "_clean.csv" suffix. If False, overwrite original files
SAVE_COPY_INSTEAD = False

# Regex to match landmark point columns (x_0, y_1, ..., z_32)
COORD_RE = re.compile(r"^[xyz]_\d+$")

def clean_file(path: str) -> int:
    """
    Removes rows where all coordinate values are missing or (optionally) all zeros.
    """
    df = pd.read_csv(path)

    # Columns that match the x/y/z coordinate format
    coord_cols = [c for c in df.columns if COORD_RE.match(c)]
    if not coord_cols:
        return 0

    coord_block = df[coord_cols]

    # Rows where all coordinate columns are NaN
    blank_mask = coord_block.isna().all(axis=1)

    n_drop = blank_mask.sum()

    if n_drop:
        # Drop blank rows and save cleaned file
        cleaned = df.loc[~blank_mask].copy()
        out_path = (
            path.replace(".csv", "_clean.csv") if SAVE_COPY_INSTEAD else path
        )
        cleaned.to_csv(out_path, index=False)

    return n_drop


total_files = 0  # CSV files processed
total_removed = 0  # Blank rows removed

for part in sorted(os.listdir(ROOT)):
    part_dir = os.path.join(ROOT, part)
    if not os.path.isdir(part_dir):
        continue

    for fname in os.listdir(part_dir):
        if not fname.lower().endswith(".csv"):
            continue

        fpath = os.path.join(part_dir, fname)
        removed = clean_file(fpath)

        if removed:
            print(f"{fpath}: {removed} blank rows removed")
            total_removed += removed

        total_files += 1

print(f"\nProcessed {total_files} CSVs – deleted {total_removed} blank rows.")
