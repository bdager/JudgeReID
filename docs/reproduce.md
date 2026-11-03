# Reproducing the paper step by step

[scripts/run_all.sh](../scripts/run_all.sh) and [scripts/run_scenario.sh](../scripts/run_scenario.sh)
run everything below. This page lists the individual steps they run, for one scenario at
P=16 (k = 8 per expert), plus the ensemble baselines, the prompt ablations and the compute
measurements. It assumes the [Setup](../README.md#setup) in the README is done.

```bash
set -a; source .env; set +a
SCENARIO=long_term
K=8                                   # top-k per expert; the VLM sees up to 2K
EMB=$EXPERIMENTS_ROOT/embeddings
GATE=$EXPERIMENTS_ROOT/gate/$SCENARIO
RUN=$EXPERIMENTS_ROOT/rerank/qwen3.6-27b/$SCENARIO/t$((2*K))
mkdir -p "$GATE"
```

## 1. Expert embeddings

```bash
for SPLIT in gallery query; do
  CSV=$CHIRLA_ROOT/metadata/reid_${SCENARIO}_${SPLIT}.csv
  python create_embeddings.py --csv "$CSV" --input "$CHIRLA_ROOT/benchmark" \
    --output "$EMB/differ" --model differ \
    --config checkpoints/DIFFER/Models/Celeb_light/config.yml \
    --cktp   checkpoints/DIFFER/Models/Celeb_light/eva02_l_bio_best.pth
  python create_embeddings.py --csv "$CSV" --input "$CHIRLA_ROOT/benchmark" \
    --output "$EMB/sapiensid" --model sapiensid \
    --cktp   checkpoints/SapiensID/sapiensid_wb12m
done
```

SapiensID's keypoint detector is not deterministic, so its numbers may differ slightly from the paper.

## 2. Expert rankings and metrics

Each expert writes `results.csv` with the full ranking of every query, which the final
step needs, and prints its CMC / mAP.

```bash
for M in differ sapiensid; do
  python metrics/evaluate_reid.py \
    --gallery "$EMB/$M/${SCENARIO}_gallery_embeddings.h5" \
    --query   "$EMB/$M/${SCENARIO}_query_embeddings.h5" \
    --topk 1 3 5 10 20 --per-subset \
    --out-csv "$EMB/$M/${SCENARIO}_results.csv"
done
```

## 3. Gate and candidate union

Queries where the experts agree at rank 1 keep SapiensID's ranking. The remaining queries
get the union of both experts' top-k, with duplicates removed.

```bash
python ensemble/top1_agreement.py \
  --results "$EMB/differ/${SCENARIO}_results.csv" "$EMB/sapiensid/${SCENARIO}_results.csv" \
  --names differ sapiensid \
  --source-csv "$EMB/sapiensid/${SCENARIO}_results.csv" \
  --per-subset --output "$GATE/agreed.csv"

for M in differ sapiensid; do
  python ensemble/remove_matching_queries.py --ref "$GATE/agreed.csv" \
    --base "$EMB/$M/${SCENARIO}_results.csv" --output "$GATE/${M}_disagreed.csv"
done

python ensemble/merge_topk.py --k $K \
  --results "$GATE/differ_disagreed.csv" "$GATE/sapiensid_disagreed.csv" \
  --names differ sapiensid --output "$GATE/union_k$K.csv"
```

## 4. VLM judge

Start the server (Qwen3.6-27B, tensor-parallel over 2 GPUs; the paper used two NVIDIA
L40S), then sort each disagreement query's candidates.

```bash
bash run_vllm_server.sh qwen3.6-27b --gpus 0,1      # in the vLLM environment

python eval_vlm_rerank.py \
  --results_csv "$GATE/union_k$K.csv" \
  --topk $((2*K)) --vlm qwen3.6-27b \
  --sys_prompt reasoning/prompts/sys_rerank.txt \
  --data_root "$CHIRLA_ROOT/benchmark/" \
  --label_images --process_all --retry_fallbacks \
  --concurrency 70 \
  --output_dir "$RUN"
```

The run is resumable. Running it again skips queries already in `$RUN/results.csv`, and
`--retry_fallbacks` retries the ones whose answer could not be parsed. Sampling settings
(greedy, reasoning off) are in [configs/vlm/qwen3.6-27b.yaml](../configs/vlm/qwen3.6-27b.yaml),
and re-prompting on unparseable answers (`max_retries`) is in
[configs/config.yaml](../configs/config.yaml).

## 5. Final ranking and metrics

Each disagreement query's final ranking is the VLM's top-k, followed by the ranking of the
expert that contributed most of those candidates (ties and unparseable answers go to
SapiensID). The agreement queries are then added back in.

```bash
python ensemble/build_final_results.py \
  --results "$EMB/differ/${SCENARIO}_results.csv" "$EMB/sapiensid/${SCENARIO}_results.csv" \
  --names differ sapiensid \
  --agree-csv "$GATE/agreed.csv" \
  --candidates-csv "$GATE/union_k$K.csv" \
  --rerank-csv "$RUN/results.csv" \
  --merge-k $K --head-size $K \
  --shared-credit all --shared-credit-tiebreak sapiensid --fallback-model sapiensid \
  --topk 1 3 5 10 20 --per-subset \
  --output-dir "$RUN/final"
```

The paper reports `combined.subset_average` from `$RUN/final/final_report.json`. For the
P=6 row, set `K=3`.

## Ensemble baselines

Borda voting, RRF and score fusion are computed directly from the embeddings:

```bash
python ensemble/ensemble.py \
  --query   "$EMB/differ/${SCENARIO}_query_embeddings.h5"   "$EMB/sapiensid/${SCENARIO}_query_embeddings.h5" \
  --gallery "$EMB/differ/${SCENARIO}_gallery_embeddings.h5" "$EMB/sapiensid/${SCENARIO}_gallery_embeddings.h5" \
  --names differ sapiensid --method voting score_fusion rrf --per-subset \
  --output-dir "$EXPERIMENTS_ROOT/baselines/$SCENARIO"
```

## Prompt ablations

The prompt ablations (Reappearance, P=6) run [step 4](#4-vlm-judge) with `K=3` and one
change each. The prompt variants are in [reasoning/prompts/ablations/](../reasoning/prompts/ablations/).

| Paper row | Change to the step 4 command |
|---|---|
| Images before the instructions | `--user_prompt_position after` |
| No anchor sentence | `--user_prompt reasoning/prompts/ablations/user_no_anchor.txt` |
| No label | drop `--label_images` |
| Attribute list, system message | `--sys_prompt reasoning/prompts/ablations/sys_no_attribute_list.txt --user_prompt reasoning/prompts/ablations/user_no_attribute_list.txt` |
| Attribute list, user message | `--user_prompt reasoning/prompts/ablations/user_no_attribute_list.txt` |
| Clothing prohibition and escape hatch | `--sys_prompt reasoning/prompts/ablations/sys_no_clothing_rule.txt` |
| Escape hatch only | `--sys_prompt reasoning/prompts/ablations/sys_no_escape_hatch.txt` |
| Step-by-step directive | `--sys_prompt reasoning/prompts/ablations/sys_no_step_by_step.txt` |

## Compute

[measure_flops.py](../measure_flops.py) measures the experts' per-image FLOPs and latency.
[flops_latency.md](flops_latency.md) gives the commands, the measured values, and how the
VLM's share of compute is counted.
