#!/usr/bin/env python3
"""
Combine per-model top-k gallery candidates into a single top-(N*k) candidate list per query.

For each query, takes the top-k ranked rows from each of the N input results CSVs and
concatenates them (model 1's top-k first, then model 2's, ...). If the same gallery image
(by normalized gallery_path) appears in more than one model's top-k, only the first
occurrence is kept (models earlier in --results take priority) — so the combined list has
up to N*k rows, but may have fewer once duplicates are removed. The output "rank" column is
renumbered 1..len(combined) to reflect this new concatenated order.

Usage:
    python ensemble/merge_topk.py \
        --results path/to/model_a/results.csv path/to/model_b/results.csv \
        --k 5 \
        --output path/to/combined_top10.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_path(p: str) -> str:
    """Extract the relative suffix starting from reid/... to align paths across models."""
    match = re.search(r"(reid/[^/]+/.+)$", p)
    return match.group(1) if match else p


def main():
    parser = argparse.ArgumentParser(
        description="Combine top-k candidates from multiple model result CSVs into one top-(N*k) list per query."
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="Paths to results CSVs (one per model), in de-duplication priority order.",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Model names (same order as --results), used only for logging. Auto-derived if omitted.",
    )
    parser.add_argument(
        "--k", type=int, required=True,
        help="Number of top-ranked candidates to take from each model.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save the combined CSV.",
    )
    args = parser.parse_args()

    assert len(args.results) >= 2, "Provide at least 2 results CSVs to combine."

    if args.names:
        assert len(args.names) == len(args.results), "Number of names must match number of results."
        model_names = args.names
    else:
        model_names = [Path(p).stem for p in args.results]

    dfs = []
    for path in args.results:
        df = pd.read_csv(path)
        df["query_key"] = df["query_path"].apply(normalize_path)
        df["match_key"] = df["gallery_path"].apply(normalize_path)
        dfs.append(df)

    has_score = all("score" in df.columns for df in dfs)

    query_keys = []
    seen_qkeys = set()
    for df in dfs:
        for qk in df["query_key"]:
            if qk not in seen_qkeys:
                seen_qkeys.add(qk)
                query_keys.append(qk)

    output_rows = []
    n_empty = 0
    for qkey in query_keys:
        combined_rows = []
        seen_match_keys = set()
        query_id = None
        query_path = None

        for df in dfs:
            sub = df[(df["query_key"] == qkey) & (df["rank"] <= args.k)].sort_values("rank")
            if sub.empty:
                continue
            if query_id is None:
                query_id = int(sub["query_id"].iloc[0])
                query_path = sub["query_path"].iloc[0]
            for _, row in sub.iterrows():
                if row["match_key"] in seen_match_keys:
                    continue
                seen_match_keys.add(row["match_key"])
                combined_rows.append(row)

        if not combined_rows:
            n_empty += 1
            continue

        for new_rank, row in enumerate(combined_rows, start=1):
            out_row = {
                "query_path": query_path,
                "query_id": query_id,
                "gallery_path": row["gallery_path"],
                "gallery_id": int(row["gallery_id"]),
                "rank": new_rank,
            }
            if has_score:
                out_row["score"] = row["score"]
            output_rows.append(out_row)

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(args.output, index=False)

    n_queries = len(query_keys) - n_empty
    print(f"Models: {model_names}")
    print(f"k per model: {args.k} (up to top-{args.k * len(dfs)} combined per query, deduplicated)")
    print(f"Queries combined: {n_queries}")
    if n_empty:
        print(f"  Skipped {n_empty} queries with no rows within top-{args.k} in any model")
    print(f"Total output rows: {len(out_df)}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
