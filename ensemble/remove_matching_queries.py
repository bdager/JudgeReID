#!/usr/bin/env python3
"""
Remove rows from a base CSV whose query paths appear in a reference CSV.

Rows in the base file whose query_path matches one in the reference are removed.
All other rows are kept as-is.

Usage:
    python ensemble/remove_matching_queries.py \
        --ref path/to/reference.csv \
        --base path/to/base.csv \
        --output path/to/filtered.csv
"""

import argparse
import re

import pandas as pd


def normalize_path(p: str) -> str:
    match = re.search(r"(reid/[^/]+/.+)$", p)
    return match.group(1) if match else p


def main():
    parser = argparse.ArgumentParser(
        description="Remove rows from a base CSV whose query paths appear in a reference CSV.",
    )
    parser.add_argument("--ref", required=True, help="CSV whose query_paths define which rows to remove.")
    parser.add_argument("--base", required=True, help="CSV to remove matching rows from.")
    parser.add_argument("--output", required=True, help="Path to save the filtered CSV.")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.ref)
    base_df = pd.read_csv(args.base)

    ref_df["query_key"] = ref_df["query_path"].apply(normalize_path)
    base_df["query_key"] = base_df["query_path"].apply(normalize_path)

    ref_keys = set(ref_df["query_key"].unique())
    base_total = base_df["query_key"].nunique()

    out_df = base_df[~base_df["query_key"].isin(ref_keys)].drop(columns=["query_key"])

    out_df.to_csv(args.output, index=False)

    n_removed = base_total - out_df["query_path"].apply(normalize_path).nunique()
    n_kept = out_df["query_path"].apply(normalize_path).nunique()
    print(f"Base queries (total):      {base_total}")
    print(f"Queries removed (agreed):  {n_removed}")
    print(f"Queries kept:              {n_kept}")
    print(f"Output rows:               {len(out_df)}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
