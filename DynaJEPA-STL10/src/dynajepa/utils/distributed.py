"""Distributed training helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def is_primary() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    if is_dist_initialized():
        dist.barrier()


def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def setup_distributed(backend: str = "nccl") -> None:
    if is_dist_initialized():
        return
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()
