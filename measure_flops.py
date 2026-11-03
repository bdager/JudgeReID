#!/usr/bin/env python3
"""
Measure FLOPs and parameter counts of the DIFFER and SapiensID Re-ID backbones.

Uses torch.utils.flop_counter.FlopCounterMode, PyTorch's built-in TorchDispatchMode-based
profiler: it intercepts real ATen ops during an actual eager forward pass, so it (a) has
native flop formulas for scaled_dot_product_attention (DIFFER's EVA02 attention uses it)
and (b) tolerates SapiensID's YOLO+NMS aligner, which a jit.trace-based profiler would not.
Reports true FLOPs (mul+add counted separately), not MACs.

Requires a real sample image, not a synthetic/noise one: SapiensID's dynamic patch
embedding depends on its YOLO+DFA aligner detecting an actual person (body/face boxes) —
fed a content-free image it produces degenerate boxes that crash later indexing.

Pick the GPU with CUDA_VISIBLE_DEVICES, not `--device cuda:N`: SapiensID's ultralytics
aligner allocates some of its internals on cuda:0 regardless of where the backbone lives,
so a non-zero `--device` index mixes tensors across GPUs and faults with an illegal memory
access at an arbitrary downstream op.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python measure_flops.py \\
        --model  differ \\
        --config checkpoints/DIFFER/Models/Celeb_light/config.yml \\
        --cktp   checkpoints/DIFFER/Models/Celeb_light/eva02_l_bio_best.pth \\
        --sample-image path/to/any/person_crop.jpg

    CUDA_VISIBLE_DEVICES=0 python measure_flops.py \\
        --model   sapiensid \\
        --cktp    checkpoints/SapiensID/sapiensid_wb12m \\
        --sample-image path/to/any/person_crop.jpg
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from create_embeddings import load_model


def count_params(model) -> int:
    total = 0
    for attr in ("model", "aligner"):
        module = getattr(model, attr, None)
        if module is not None:
            total += sum(p.numel() for p in module.parameters())
    return total


def benchmark_latency(
    model, batch: torch.Tensor, device: str, warmup: int, iters: int
) -> tuple[float, float]:
    """Return (mean, std) seconds per forward pass, synchronized around each call."""
    is_cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if is_cuda:
            torch.cuda.synchronize(device)

        times = []
        for _ in range(iters):
            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(batch)
            if is_cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)

    mean_s = sum(times) / len(times)
    var = sum((t - mean_s) ** 2 for t in times) / len(times)
    return mean_s, var**0.5


def measure(
    model_name: str,
    config_file: str | None,
    checkpoint_file: str | None,
    sample_image: str,
    batch_size: int,
    device: str,
    warmup: int,
    iters: int,
) -> None:
    model, emb_dim = load_model(
        model_name=model_name,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        device=device,
    )

    img = Image.open(sample_image).convert("RGB")
    batch = torch.stack([model.transform(img) for _ in range(batch_size)])
    batch = batch.to(device)

    mean_s, std_s = benchmark_latency(model, batch, device, warmup=warmup, iters=iters)

    with torch.no_grad(), FlopCounterMode(display=False) as fcm:
        model(batch)

    total_flops = fcm.get_total_flops()
    params = count_params(model)

    print("Per-module FLOPs (top level):")
    for module_name, op_counts in fcm.get_flop_counts().items():
        if module_name == "Global" or "." in module_name:
            continue
        print(f"  {module_name}: {sum(op_counts.values()) / 1e9:.3f} GFLOPs")

    print(f"\nModel:          {model_name}")
    print(f"Device:         {device}")
    print(f"Emb dim:        {emb_dim}")
    print(f"Params:         {params / 1e6:.2f} M")
    print(f"Batch size:     {batch_size}")
    print(f"Total FLOPs:    {total_flops / 1e9:.3f} GFLOPs")
    print(f"FLOPs / image:  {total_flops / batch_size / 1e9:.3f} GFLOPs")
    print(
        f"Latency:        {mean_s * 1000:.2f} +/- {std_s * 1000:.2f} ms/batch "
        f"({mean_s * 1000 / batch_size:.2f} ms/image, {iters} iters after {warmup} warmup)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure FLOPs and params of the DIFFER / SapiensID backbones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=["differ", "sapiensid"], help="Backbone to measure"
    )
    parser.add_argument("--config", default=None, help="[differ] Path to model config file")
    parser.add_argument(
        "--cktp",
        default=None,
        help="Path to model checkpoint (differ: .pth file; sapiensid: checkpoint dir "
        "with model.yaml + model.pth)",
    )
    parser.add_argument(
        "--sample-image",
        required=True,
        help="Path to a real person-crop image (required — a synthetic/noise image "
        "makes SapiensID's person detector produce degenerate boxes)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Images per forward pass")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device: 'cuda' or 'cpu'. To pick a GPU use CUDA_VISIBLE_DEVICES, "
        "not an index here (see module docstring)",
    )
    # 300 warmup passes (~10 s) rather than a handful: an idle GPU sits at ~210 MHz and
    # a short warmup times it mid-ramp, which showed up as a bimodal ~28 vs ~49 ms.
    parser.add_argument(
        "--warmup", type=int, default=300, help="Warmup forward passes before timing"
    )
    parser.add_argument(
        "--iters", type=int, default=200, help="Timed forward passes to average over"
    )

    args = parser.parse_args()

    if args.model == "differ" and args.config is None:
        parser.error("--config is required for --model differ")
    if args.cktp is None:
        parser.error("--cktp is required")
    if args.device.startswith("cuda:"):
        parser.error(
            f"use CUDA_VISIBLE_DEVICES={args.device.split(':')[1]} with --device cuda "
            "instead; a non-zero device index crashes SapiensID's aligner"
        )

    measure(
        model_name=args.model,
        config_file=args.config,
        checkpoint_file=args.cktp,
        sample_image=args.sample_image,
        batch_size=args.batch_size,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
    )
