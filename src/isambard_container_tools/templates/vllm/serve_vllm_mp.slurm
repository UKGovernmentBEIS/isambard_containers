#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# Required environment variables (passed via sbatch --export):
#   MODEL          - HuggingFace model ID
#   CONTAINER      - Path to container image
#   VLLM_ARGS      - CLI args for vLLM serve (space-separated)
#   WORK_DIR       - Project root directory
#   SERVE_PORT     - Port for the vLLM API server
#   MASTER_PORT    - Port for PyTorch distributed master
#
# Optional:
#   USER_SCRIPT    - Python script to run after vLLM starts
#   SCRIPT_ARGS    - Arguments for USER_SCRIPT
#   CONTAINER_ENV  - Semicolon-separated KEY=VALUE pairs to export into container
#                    (e.g., VLLM_USE_DEEP_GEMM=0;VLLM_ATTENTION_BACKEND=TRITON_MLA)
#   DEBUG_MODE     - Set to "1" to enable verbose logging and per-node resource monitoring

echo "Using backend MP"

set -euo pipefail

# ── Unset login-node vars that break Apptainer ────────────────────────
# https://docs.isambard.ac.uk/service-status/known_issues/#use-of-environment-variables-in-apptainer
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR LOCALDIR TMPDIR 2>/dev/null || true

# ── Per-job scratch tmp dir ───────────────────────────────────────────
# Isolate XDG dirs per job on Lustre scratch so concurrent jobs don't collide
# on NFS $HOME (e.g. inspect_ai trace rotation causing stale file handles).
JOB_TMP="$SCRATCHDIR/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"
export XDG_DATA_HOME="$JOB_TMP/xdg_data"
export XDG_CACHE_HOME="$JOB_TMP/xdg_cache"
export XDG_STATE_HOME="$JOB_TMP/xdg_state"

# ── Debug mode ──────────────────────────────────────────────────────
if [ "${DEBUG_MODE:-0}" = "1" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
    export VLLM_LOGGING_LEVEL=INFO
    export FI_LOG_LEVEL=warn
    export TORCH_DISTRIBUTED_DEBUG=INFO
    # Suppress per-request httpx/httpcore TRACE logging (client-side HTTP noise)
    export HTTPX_LOG_LEVEL=WARNING
    export HTTPCORE_LOG_LEVEL=WARNING
    export TORCH_FR_BUFFER_SIZE=1000
    # NCCL flight recorder: dumps in-flight collectives on timeout
    export TORCH_NCCL_ENABLE_MONITORING=1
    export TORCH_NCCL_DUMP_ON_TIMEOUT=1
    export TORCH_NCCL_TRACE_CPP_STACK=1
    DEBUG_LOG_DIR="${WORK_DIR:-$PWD}/logs/debug_${SLURM_JOB_ID}"
    mkdir -p "$DEBUG_LOG_DIR"
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DEBUG_LOG_DIR/nccl_trace"
    echo "[DEBUG] Debug mode enabled — NCCL flight recorder to $DEBUG_LOG_DIR"
fi

# ── Per-model container environment variables ────────────────────────
# Singularity passes through host env by default, so exporting here
# makes these available inside the container without --env flags.
if [ -n "${CONTAINER_ENV:-}" ]; then
    IFS=';' read -ra _ENVS <<< "$CONTAINER_ENV"
    for _env_pair in "${_ENVS[@]}"; do
        export "$_env_pair"
    done
    unset _ENVS _env_pair
fi

# ── Multi-node container support (Slingshot NCCL) ────────────────────
module load brics/apptainer-multi-node

# ── Resolve head node ─────────────────────────────────────────────────
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(dig +short "$HEAD_NODE")

# ── Per-node worker function ──────────────────────────────────────────
run_node() {
    local my_ip
    my_ip=$(dig +short "$(hostname)")
    export VLLM_HOST_IP="$my_ip"
    local my_hostname
    my_hostname=$(hostname)
    local node_rank=$SLURM_NODEID

    echo "[$my_hostname] rank=$node_rank ip=$my_ip"

    # Per-job tmp dir on host /tmp (RAM-backed tmpfs), isolated from other jobs
    HOST_TMP="/tmp/vllm_${SLURM_JOB_ID}_${SLURM_NODEID}"
    mkdir -p "$HOST_TMP"

    # Build bind mounts
    # HOST_TMP → /tmp inside container so caches are isolated per job
    # /dev/shm is critical for vLLM V1's multiprocess shm_broadcast mechanism
    # /lib64 + /usr/lib64 are needed for Triton JIT compilation - GCC's linker scripts
    # reference /lib64/libc.so.6 and /usr/lib64/libc_nonshared.a
    # Note: brics/apptainer-multi-node sets SINGULARITY_BINDPATH for Slingshot libs
    # --contain uses a minimal /dev, so bind the full host /dev for CXI + GDRCopy access
    BIND_MOUNTS="$SCRATCHDIR:$SCRATCHDIR,$HOST_TMP:/tmp"
    BIND_MOUNTS="$BIND_MOUNTS,/lib64:/lib64:ro,/usr/lib64:/usr/lib64:ro"
    BIND_MOUNTS="$BIND_MOUNTS,/etc/ssl:/etc/ssl:ro,/var/lib/ca-certificates:/var/lib/ca-certificates:ro"
    BIND_MOUNTS="$BIND_MOUNTS,/dev:/dev"
    BIND_MOUNTS="$BIND_MOUNTS,$PROJECTDIR:$PROJECTDIR"

    # Increase file descriptor limit
    ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || true

    # Singularity wrapper
    # --contain disables auto-mounting of $HOME and CWD for filesystem isolation
    # --pid creates a PID namespace so killing singularity kills all children (vLLM workers)
    # /host/adapt.sh binds required host dependencies (Slingshot/NCCL) but
    # force-sets some NCCL vars (CROSS_NIC=0, LD_LIBRARY_PATH) that are
    # suboptimal for GH200. We override these after sourcing adapt.sh.
    # See: https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/
    # Note: --contain does NOT clean env vars, so host variables from .env
    # (HF_HUB_CACHE, SCRATCHDIR, etc.) are still passed through automatically.
    #
    # JIT cache dirs (Triton, FlashInfer, torchinductor, torch extensions) stay on
    # RAM-backed /tmp — they generate many small files during compilation and would
    # bottleneck on Lustre MDS. They don't need to persist across jobs.
    srun_exec() {
        singularity exec --nv --pid --contain \
            --bind "$BIND_MOUNTS" \
            --env TRITON_CACHE_DIR="/tmp/.triton" \
            --env FLASHINFER_CACHE_DIR="/tmp/.flashinfer" \
            --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
            --env VLLM_USE_FLASHINFER_MOE_FP8=0 \
            --env TORCH_EXTENSIONS_DIR="/tmp/.torch_ext" \
            --env TORCH_HOME="/tmp/.torch" \
            --env TORCHINDUCTOR_CACHE_DIR="/tmp/.torchinductor" \
            --env DG_JIT_CACHE_DIR="/tmp/deepgemm_jit" \
            "$CONTAINER" \
            bash -c 'source /host/adapt.sh; exec "$@"' _ \
            bash -c '
                # ── GH200 overrides — applied AFTER adapt.sh ──
                # adapt.sh ends in `exec "$@"`, so it execs THIS wrapper
                export NCCL_CROSS_NIC=1
                # Use container-built aws-ofi-nccl (matched to our CUDA) over host version
                export LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH"
                # Disable GDRCopy for libfabric CUDA HMEM registration: its finite
                # BAR1 mapping pool exhausts under sustained multi-node P2P traffic,
                # after which gdr_map fails with EINVAL (RC:22) and NCCL tears the
                # job down.
                export FI_HMEM_CUDA_USE_GDRCOPY=0
                exec "$@"
            ' _ "$@"
    }

    # Build vLLM args (runtime args + vllm_args from Python)
    VLLM_CMD_ARGS=(
        --distributed-executor-backend mp
        --nnodes "$SLURM_NNODES"
        --node-rank "$node_rank"
        --master-addr "$HEAD_IP"
        --master-port "$MASTER_PORT"
    )
    # VLLM_ARGS is built by dict_to_cli_args with shlex.quote(); eval is needed
    # so bash processes the quotes rather than treating them as literal characters.
    eval "VLLM_CMD_ARGS+=($VLLM_ARGS)"

    # Rank 0 is the API server, others are headless workers
    if [ "$node_rank" -eq 0 ]; then
        VLLM_CMD_ARGS+=(--host "${LISTEN_HOST:-0.0.0.0}" --port "$SERVE_PORT")
    else
        VLLM_CMD_ARGS+=(--headless)
    fi

    # Start vLLM in background
    srun_exec vllm serve "$MODEL" "${VLLM_CMD_ARGS[@]}" &
    VLLM_PID=$!

    # ── Debug: per-node resource monitor ─────────────────────────────
    MONITOR_PID=""
    if [ "${DEBUG_MODE:-0}" = "1" ]; then
        DEBUG_LOG_DIR="${WORK_DIR:-$PWD}/logs/debug_${SLURM_JOB_ID}"
        mkdir -p "$DEBUG_LOG_DIR"
        RESOURCE_LOG="$DEBUG_LOG_DIR/${my_hostname}_resources.log"
        echo "[DEBUG] Resource monitoring → $RESOURCE_LOG"
        (
            while kill -0 $VLLM_PID 2>/dev/null; do
                echo "=== $(date -Iseconds) ===" >> "$RESOURCE_LOG"
                # GPU utilisation and memory
                nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                    --format=csv,noheader,nounits >> "$RESOURCE_LOG" 2>&1 || true
                # Host memory and CPU
                echo "MEM: $(free -m | awk 'NR==2{printf "used=%sMB total=%sMB (%.1f%%)", $3, $2, $3*100/$2}')" >> "$RESOURCE_LOG"
                echo "CPU: $(uptime | sed 's/.*load average: /load=/')" >> "$RESOURCE_LOG"
                echo "" >> "$RESOURCE_LOG"
                sleep 10
            done
        ) &
        MONITOR_PID=$!
    fi

    # Shared markers for coordinated shutdown (all ranks need these paths)
    MARKER_DIR="$JOB_TMP/markers"
    mkdir -p "$MARKER_DIR" 2>/dev/null || true
    DONE_MARKER="$MARKER_DIR/done"

    if [ -n "$USER_SCRIPT" ]; then
        # ── Script mode: run user script then signal shutdown ──

        if [ "$node_rank" -eq 0 ]; then
            # Safety-net trap (only fires if script crashes or job gets killed)
            trap 'kill -TERM $VLLM_PID 2>/dev/null; sleep 5; kill -KILL $VLLM_PID 2>/dev/null || true' EXIT

            # Wait for vLLM health (up to 30 min)
            echo "Waiting for vLLM to become healthy on port $SERVE_PORT..."
            for _attempt in $(seq 1 360); do
                if curl -sf "http://localhost:$SERVE_PORT/health" > /dev/null 2>&1; then
                    echo "vLLM is ready!"
                    break
                fi
                sleep 5
            done

            # Export for Inspect AI vLLM provider (https://inspect.aisi.org.uk/providers.html#vllm)
            export VLLM_BASE_URL="http://localhost:$SERVE_PORT/v1"
            export VLLM_API_KEY="token"  # vLLM doesn't require auth, but Inspect expects this

            # Run user script on host (not in container)
            if [ -n "${WORK_DIR:-}" ]; then
                cd "$WORK_DIR"
                source .venv/bin/activate
            fi
            # shellcheck disable=SC2086
            python "$USER_SCRIPT" $SCRIPT_ARGS
            SCRIPT_EXIT=$?

            # Signal workers to shut down first (they poll every 1s)
            echo "$SCRIPT_EXIT" > "$DONE_MARKER"

            # Wait for workers to acknowledge shutdown before killing head's vLLM.
            # Workers must disconnect from TCPStore before the head (which hosts it)
            # exits, otherwise PyTorch distributed prints noisy teardown errors.
            NUM_WORKERS=$((SLURM_NNODES - 1))
            if [ "$NUM_WORKERS" -gt 0 ]; then
                echo "Waiting for $NUM_WORKERS worker(s) to shut down..."
                for _i in $(seq 1 30); do
                    ACKS=$(ls "$MARKER_DIR"/worker_done_* 2>/dev/null | wc -l)
                    if [ "$ACKS" -ge "$NUM_WORKERS" ]; then
                        echo "All workers confirmed shutdown."
                        break
                    fi
                    sleep 1
                done
            fi

            # Now safe to kill head's vLLM (workers already disconnected from TCPStore)
            kill -TERM $VLLM_PID 2>/dev/null || true
            wait $VLLM_PID 2>/dev/null || true

            # Disable safety-net trap since we already cleaned up
            trap - EXIT
            exit $SCRIPT_EXIT
        else
            # Worker: wait for done marker then exit cleanly
            while kill -0 $VLLM_PID 2>/dev/null; do
                if [ -f "$DONE_MARKER" ]; then
                    echo "[$my_hostname] Head node completed, shutting down worker..."
                    kill -TERM $VLLM_PID 2>/dev/null || true
                    wait $VLLM_PID 2>/dev/null || true
                    # Acknowledge shutdown so head knows we've disconnected from TCPStore
                    touch "$MARKER_DIR/worker_done_${node_rank}"
                    exit 0
                fi
                sleep 1
            done
            # vLLM exited - check if head completed
            [ -f "$DONE_MARKER" ] && exit 0
            exit 1
        fi
    else
        # ── Serve mode: wait for vLLM, with optional gimlet tunnel ──
        if [ "$node_rank" -eq 0 ] && [ -n "${GIMLET_TOKEN_FILE:-}" ] && [ -f "$GIMLET_TOKEN_FILE" ]; then
            GIMLET_AGENT="${HOME}/.local/bin/gimlet-agent"
            if [ -f "$GIMLET_AGENT" ]; then
                echo "Waiting for vLLM health check before starting gimlet..."
                for _ in $(seq 1 360); do
                    curl -sf "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null && break
                    sleep 5
                done
                if curl -sf "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null; then
                    "$GIMLET_AGENT" \
                        --server-url "${GIMLET_SERVER_URL}" \
                        --target-url "http://${LISTEN_HOST:-0.0.0.0}:${SERVE_PORT}" \
                        --token "$(cat "$GIMLET_TOKEN_FILE")" &
                    GIMLET_PID=$!
                    echo "Gimlet tunnel active: ${GIMLET_URL}/services/${SERVICE_NAME}/v1"
                else
                    echo "WARNING: vLLM health check timed out, skipping gimlet"
                fi
            else
                echo "WARNING: gimlet-agent not found at $GIMLET_AGENT, skipping tunnel"
            fi
        fi
        wait $VLLM_PID
        [ -n "${GIMLET_PID:-}" ] && kill $GIMLET_PID 2>/dev/null || true
    fi
}

export -f run_node
export MODEL CONTAINER HEAD_IP VLLM_ARGS SERVE_PORT MASTER_PORT VLLM_HOST_IP LISTEN_HOST
export USER_SCRIPT SCRIPT_ARGS WORK_DIR CONTAINER_ENV DEBUG_MODE JOB_TMP
export GIMLET_TOKEN_FILE GIMLET_URL GIMLET_SERVER_URL SERVICE_NAME

# Top-level cleanup: ensure all child processes are killed on job termination
cleanup_job() {
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 2
    pkill -KILL -P $$ 2>/dev/null || true
    rm -rf "$JOB_TMP" 2>/dev/null || true
    # Clean per-node RAM-backed tmp dirs (HOST_TMP = /tmp/vllm_<JOBID>_<NODEID>)
    rm -rf /tmp/vllm_${SLURM_JOB_ID}_* 2>/dev/null || true
}
trap cleanup_job EXIT TERM INT

srun --kill-on-bad-exit=1 --gpus="$((SLURM_NNODES * SLURM_GPUS_PER_NODE))" bash -c run_node
