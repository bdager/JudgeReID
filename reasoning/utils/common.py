"""
Shared reproducibility helpers for JudgeReID scripts.

Available functions:
    - save_run_cmd      — writes run_cmd.sh from sys.argv
    - save_git_info     — writes git_commit.txt (commit / branch / status / diff)
    - save_prompts      — copies prompt strings into {output_dir}/prompts/
    - save_run_config   — serialises argparse Namespace → run_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def save_run_cmd(output_dir: str) -> None:
    """Write a ``run_cmd.sh`` that reproduces the current invocation."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = " ".join(shlex.quote(a) for a in sys.argv)
    path = os.path.join(output_dir, "run_cmd.sh")
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"python {cmd}\n")
    os.chmod(path, 0o755)
    logger.info("Run command saved to %s", path)


def save_git_info(output_dir: str) -> None:
    """Save current git commit, branch, status, and diff summary."""
    os.makedirs(output_dir, exist_ok=True)

    def _run(cmd: list) -> str:
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    path = os.path.join(output_dir, "git_commit.txt")
    if not commit:
        with open(path, "w") as f:
            f.write("Not a git repository (or git not available).\n")
        return
    branch    = _run(["git", "branch", "--show-current"])
    status    = _run(["git", "status", "--short"])
    diff_stat = _run(["git", "diff", "--stat"])
    with open(path, "w") as f:
        f.write(f"commit:  {commit}\n")
        f.write(f"branch:  {branch}\n")
        if status:
            f.write(f"\nstatus:\n{status}\n")
        if diff_stat:
            f.write(f"\ndiff --stat:\n{diff_stat}\n")
    logger.info("Git info saved to %s", path)


def save_prompts(output_dir: str, prompts: Dict[str, Optional[str]]) -> None:
    """Save each prompt string to ``{output_dir}/prompts/{name}.txt``."""
    prompts_dir = os.path.join(output_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    for name, text in prompts.items():
        if text is None:
            continue
        with open(os.path.join(prompts_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(str(text))
    logger.info("Prompts saved to %s", prompts_dir)


def save_run_config(
    output_dir: str,
    args: argparse.Namespace,
    extra: Optional[dict] = None,
) -> None:
    """Serialise all argparse arguments to ``{output_dir}/run_config.json``.

    Parameters
    ----------
    output_dir:
        Directory where the config file will be written.
    args:
        Parsed CLI arguments.
    extra:
        Additional key-value pairs to include (e.g. resolved model config,
        VLM params).
    """
    os.makedirs(output_dir, exist_ok=True)
    config: dict = {"_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    config.update(vars(args))
    if extra:
        config.update(extra)
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4, default=str)
    logger.info("Run config saved to %s", path)


def save_reproducibility_artifacts(
    output_dir: str,
    args,
    prompts: Optional[Dict[str, Optional[str]]] = None,
    extra: Optional[dict] = None,
) -> None:
    """One-call orchestrator for all reproducibility artefacts.

    Creates: ``run_cmd.sh``, ``git_commit.txt``, ``prompts/``,
    ``run_config.json``, and optionally ``config_snapshot.yaml``.

    Parameters
    ----------
    output_dir : str
        Directory where artefacts will be created.
    args : argparse.Namespace
        Parsed CLI arguments.
    prompts : dict, optional
        Prompt texts to save (see :func:`save_prompts`).
    extra : dict, optional
        Additional key-value pairs forwarded to :func:`save_run_config`
        (e.g. resolved VLM params).
    """
    os.makedirs(output_dir, exist_ok=True)
    save_run_cmd(output_dir)
    save_git_info(output_dir)
    if prompts:
        save_prompts(output_dir, prompts)
    save_run_config(output_dir, args, extra=extra)
    # Copy the config file used (optional)
    config_path = getattr(args, "config", None)
    if config_path and os.path.isfile(config_path):
        dst = os.path.join(output_dir, "config_snapshot.yaml")
        shutil.copy2(config_path, dst)
        logger.info("Config snapshot saved to %s", dst)
