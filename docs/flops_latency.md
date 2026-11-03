# Compute cost — DIFFER vs SapiensID

Measured with [`measure_flops.py`](../measure_flops.py) (`torch.utils.flop_counter.FlopCounterMode`),
batch size 1, on one idle NVIDIA GPU, 200 timed iterations after 300 warmup passes.

| Model | Params | FLOPs / image | Latency / image |
|---|---:|---:|---:|
| DIFFER (Celeb_light) | 306.97 M | 162.0 G | 13.09 ± 0.11 ms |
| SapiensID wb12m | 422.02 M | 48.0 G | 29.05 ± 0.41 ms |

FLOPs are true FLOPs (multiply and add counted separately), not MACs, and are
device-independent. SapiensID's figure is the backbone + semantic attention head; its
YOLO+DFA aligner adds a further 1–5 GFLOPs depending on how many boxes the detector
returns, and accounts for roughly half the wall-clock latency.

## Commands

```bash
SAMPLE="$CHIRLA_ROOT/benchmark/reid/long_term/test/test_5/\
seq_004/imgs/camera_6_2023-06-08-12:29:08/3/frame_4062.png"

CUDA_VISIBLE_DEVICES=0 python measure_flops.py \
  --model  differ \
  --config checkpoints/DIFFER/Models/Celeb_light/config.yml \
  --cktp   checkpoints/DIFFER/Models/Celeb_light/eva02_l_bio_best.pth \
  --sample-image "$SAMPLE"

CUDA_VISIBLE_DEVICES=0 python measure_flops.py \
  --model   sapiensid \
  --cktp    checkpoints/SapiensID/sapiensid_wb12m \
  --sample-image "$SAMPLE"
```

Select the GPU with `CUDA_VISIBLE_DEVICES`, never `--device cuda:N` — SapiensID's
ultralytics aligner allocates on `cuda:0` regardless of where the backbone lives, so a
non-zero index mixes tensors across GPUs and faults with an illegal memory access.

## Cross-check against the paper

[SapiensID](https://arxiv.org/abs/2504.04708) appendix B.8 reports ~36.9 GFLOPs
(35.39 backbone + 0.45 RetinaPatch + 1.09 head) at 281 tokens on average, with a
336.65 M-param head.

- **Params** — 336.65 M head + ~86 M ViT-base ≈ 422 M, matching the measured 422.02 M.
- **Tokens** — the CHIRLA crop above yields 286 tokens, close to their 281 average, so the
  input is representative.
- **FLOPs** — 48.0 G measured vs ~36.9 G reported (1.3×). SapiensID writes attention as raw
  `q @ k.transpose(-2, -1)` and `attn @ v`, which module-hook counters miss — the reason its
  own code carries a hand-rolled `extra_gflops` correction. `FlopCounterMode` counts those
  matmuls natively, along with softmax, norms and the KPRPE bias.

## How total compute is counted

The PFLOPs column in the paper is the forward compute to run all four scenarios once:

```
FLOPs = (closed-set queries + gallery images) × Σ FLOPs/image of the experts used
      + routed queries × calls/query × 2 × N_active × tokens/call
tokens/call = 1039.7 + 108.17 × candidates shown
```

- **Experts:** 162.0 G (DIFFER) and 48.0 G (SapiensID) per image, from the table above.
  Distractor queries are embedded but never scored closed-set, so they are not counted.
- **VLM:** the usual 2 × N × T forward estimate, with N_active = 27 B (Qwen3.6-27B) and T the
  prompt + completion tokens. It leaves out the vision encoder that turns crops into image
  tokens and attention's quadratic term (small at ~3k tokens).
- **Tokens per call:** a least-squares fit over the runs that logged `token_usage`
  (R² = 0.988). The intercept is the system prompt plus the query crop; each candidate crop
  and its label adds 108 tokens.
- **Candidates shown:** the mean of `n_candidates` in the judge's `results.csv`. The union of
  two top-k lists is deduplicated, so it is below 2k (e.g. 13.9 at P = 16 on long_term).
- **Calls per query:** 1 + the mean of `num_retries` in `results.csv`, since an unparseable
  answer triggers a correction prompt (1.027 at P = 16, 1.156 at P = 40).
- **Routed queries:** every query the gate sends to the VLM, i.e. `no_agreement_subset`
  `n_queries` in `final_report.json` (all queries for the ungated pipelines). This includes
  queries whose ground truth is not in the pool, since a deployed system cannot tell them
  apart.

Example, gated pipeline at P = 16:

| Scenario | Queries | Gallery | Routed | Candidates |
|---|---:|---:|---:|---:|
| long_term | 3692 | 368 | 2749 | 13.88 |
| multi_camera_long_term | 1134 | 252 | 738 | 12.59 |
| multi_camera | 4311 | 305 | 2472 | 13.25 |
| reappearance | 1251 | 258 | 682 | 13.22 |

Experts: 11 571 images × 210 G = 2.4 PFLOPs. VLM: 6 641 routed queries × 1.027 calls ×
2 × 27 B × ~2 490 tokens = 918 PFLOPs. Total: 921 PFLOPs.
