#!/usr/bin/env bash
# Run scripts/run_scenario.sh on the four CHIRLA scenarios and print the paper's summary
# (rank-1 and mAP per scenario, averaged over subsets, plus their mean)
#
# Usage
#   bash scripts/run_all.sh                 # P = 16 (K = 8), the paper's main setting
#   K=3 bash scripts/run_all.sh             # P = 6
#   STEPS=final bash scripts/run_all.sh     # only rebuild the final results
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENARIOS=(long_term multi_camera_long_term multi_camera reappearance)
for SCENARIO in "${SCENARIOS[@]}"; do
    bash scripts/run_scenario.sh "$SCENARIO"
done

if [[ -f .env ]]; then set -a; source .env; set +a; fi
RUN_ROOT="$EXPERIMENTS_ROOT/rerank/${VLM:-qwen3.6-27b}" P=$((2 * ${K:-8})) python - <<'PY'
import json, os
from pathlib import Path

scenarios = ["long_term", "multi_camera_long_term", "multi_camera", "reappearance"]
p = os.environ["P"]
rows = []
for s in scenarios:
    report = Path(os.environ["RUN_ROOT"], s, f"t{p}", "final", "final_report.json")
    avg = json.loads(report.read_text())["combined"]["subset_average"]
    rows.append((s, 100 * avg["CMC@1"], 100 * avg["mAP"]))

print(f"\nP = {p}")
print(f"{'scenario':<24}{'R-1':>8}{'mAP':>8}")
for s, r1, m in rows:
    print(f"{s:<24}{r1:>8.2f}{m:>8.2f}")
print(f"{'average':<24}{sum(r[1] for r in rows) / 4:>8.2f}{sum(r[2] for r in rows) / 4:>8.2f}")
PY
