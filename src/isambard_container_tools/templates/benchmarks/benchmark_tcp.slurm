#!/bin/bash
#SBATCH --job-name=bench_tcp
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq

# Benchmark NCCL bandwidth with TCP sockets (baseline)
#
# Expected env vars (set by benchmark-networking CLI):
#   CONTAINER         - path to .sif container
#   BENCHMARK_SCRIPT  - path to benchmark_allreduce.py

# Validate required environment variables
if [ -z "$SCRATCHDIR" ]; then
    echo "ERROR: SCRATCHDIR is not set"
    exit 1
fi
if [ -z "$PROJECTDIR" ]; then
    echo "ERROR: PROJECTDIR is not set"
    exit 1
fi

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

# Check if LOCALDIR exists and is a directory, fallback to /tmp/$USER
if [ -n "$LOCALDIR" ] && [ -d "$LOCALDIR" ]; then
    CONTAINER_TMP="$LOCALDIR"
else
    CONTAINER_TMP="/tmp/${AISI_PLATFORM_USER:-$USER}"
    # Create on every allocated node; bare `mkdir -p` only runs on the
    # batch head, leaving other nodes failing the bind-mount.
    srun --ntasks-per-node=1 mkdir -p "$CONTAINER_TMP"
    if [ -n "$LOCALDIR" ]; then
        echo "Warning: LOCALDIR is set to '$LOCALDIR' but directory does not exist, using $CONTAINER_TMP"
    fi
fi

echo "=== NCCL Bandwidth: TCP Sockets (baseline) ==="

# Build bind mounts
BIND_MOUNTS="$SCRATCHDIR:$SCRATCHDIR,$CONTAINER_TMP:/tmp"
if [ -n "$PROJECTDIR" ]; then
    BIND_MOUNTS="$BIND_MOUNTS,$PROJECTDIR:$PROJECTDIR"
    # Also bind the resolved path so Python's __file__ paths work inside the container
    REAL_PROJECTDIR=$(readlink -f "$PROJECTDIR")
    if [ "$REAL_PROJECTDIR" != "$PROJECTDIR" ]; then
        BIND_MOUNTS="$BIND_MOUNTS,$REAL_PROJECTDIR:$REAL_PROJECTDIR"
    fi
fi

# Override container's default NCCL_NET to force TCP sockets
# (container sets NCCL_NET="AWS Libfabric" for Slingshot by default)
srun singularity exec --nv \
    --bind $BIND_MOUNTS \
    --env SCRATCH="${SCRATCH:-}" \
    --env SCRATCHDIR="$SCRATCHDIR" \
    --env PROJECTDIR="$PROJECTDIR" \
    --env NCCL_NET="Socket" \
    $CONTAINER \
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29500 \
        $BENCHMARK_SCRIPT
