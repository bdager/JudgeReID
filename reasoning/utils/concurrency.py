"""Generic bounded-concurrency task runner shared by the eval scripts."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


async def run_concurrent(
    items: List[T],
    process_one: Callable[[T], Awaitable[R]],
    on_result: Callable[[R], None],
    desc: str = "Processing",
) -> None:
    """Run ``process_one`` concurrently over ``items``, feeding each result
    to ``on_result`` as it completes.

    ``on_result`` runs synchronously in this single consumer loop — it is
    the only place that should touch shared files/dicts, so callers don't
    need locks around incremental writes or stat accumulation.
    """
    tasks = [asyncio.create_task(process_one(item)) for item in items]
    with tqdm(total=len(tasks), desc=desc) as pbar:
        for fut in asyncio.as_completed(tasks):
            on_result(await fut)
            pbar.update(1)
