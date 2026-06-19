#!/usr/bin/env python3
"""
Benchmark NCCL all_reduce bandwidth across nodes.
Measures actual inter-node GPU communication bandwidth.
"""

import os
import time
import torch
import torch.distributed as dist


def benchmark_allreduce(sizes_mb=[1, 8, 64, 256, 512, 1024], warmup=5, iters=20):
    """Benchmark all_reduce at various message sizes."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("NCCL All-Reduce Bandwidth Benchmark")
        print(f"{'=' * 60}")
        print(f"World size: {world_size} GPUs")
        print(f"{'=' * 60}")
        print(
            f"{'Size':>12} {'Time (ms)':>12} {'Algbw (GB/s)':>14} {'Busbw (GB/s)':>14}"
        )
        print(f"{'-' * 12} {'-' * 12} {'-' * 14} {'-' * 14}")

    for size_mb in sizes_mb:
        num_elements = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
        size_bytes = tensor.numel() * tensor.element_size()

        # Warmup
        for _ in range(warmup):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Benchmark
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / iters) * 1000
        # Algorithm bandwidth: data_size / time
        algbw = size_bytes / (elapsed / iters) / 1e9
        # Bus bandwidth for all_reduce: 2 * (n-1) / n * data_size / time
        busbw = algbw * 2 * (world_size - 1) / world_size

        if rank == 0:
            print(
                f"{size_mb:>10} MB {avg_time_ms:>10.2f}ms {algbw:>12.2f} {busbw:>12.2f}"
            )

    if rank == 0:
        print(f"{'=' * 60}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_allreduce()
