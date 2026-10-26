#!/usr/bin/env bash
# Reproducible LBMSim benchmark.
#
# Builds the simulator in Release mode, records the hardware/software
# configuration, and times the solver on several grid sizes:
#   * GPU, compute only   (headless, no window)
#   * GPU, compute+render (window, vsync off; skipped without a display)
#   * CPU prototype       (branch alpha-version, single thread; see cpu_reference/)
#
# Usage:  benchmarks/run_benchmarks.sh [--no-render] [--no-cpu] [--quick]
# Tunables (environment): SIZES, STEPS, WARMUP, REPEATS, CPU_STEPS, CUDA_ARCH_LIST
#
# Results go to benchmarks/results/<date>-<host>/:
#   hardware.txt  configuration of the machine
#   raw.csv       every run
#   summary.md    medians, ready to paste into the README
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIZES=${SIZES:-"640x360 960x540 1280x720 1920x1080"}
STEPS=${STEPS:-2000}
WARMUP=${WARMUP:-200}
REPEATS=${REPEATS:-3}
CPU_STEPS=${CPU_STEPS:-30}
RENDER=1
CPU=1

for arg in "$@"; do
    case "$arg" in
        --no-render) RENDER=0 ;;
        --no-cpu)    CPU=0 ;;
        --quick)     STEPS=300; WARMUP=50; REPEATS=1; CPU_STEPS=5 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$RENDER" = 1 ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "No display found: skipping the render benchmark (compute-only and CPU still run)." >&2
    RENDER=0
fi

OUT="$ROOT/benchmarks/results/$(date +%Y-%m-%d)-$(hostname -s)"
BUILD="$ROOT/build-bench"
mkdir -p "$OUT"

echo "==> Building LBMSim (Release) in $BUILD"
cmake -S "$ROOT/ProjectSource" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      ${CUDA_ARCH_LIST:+-DCUDA_ARCH_LIST="$CUDA_ARCH_LIST"} > "$OUT/cmake.log"
cmake --build "$BUILD" -j "$(nproc)" >> "$OUT/cmake.log"
BIN="$BUILD/executables"

if [ "$CPU" = 1 ]; then
    echo "==> Building CPU prototype reference"
    g++ -O3 -march=native -std=c++17 -w "$ROOT/benchmarks/cpu_reference/alpha_cpu_bench.cpp" \
        -o "$BUILD/alpha_cpu_bench"
fi

echo "==> Recording configuration to $OUT/hardware.txt"
{
    echo "date:       $(date -Iseconds)"
    echo "commit:     $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo ' (dirty)')"
    echo "os:         $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
    echo "cpu:        $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
    echo "cpu cores:  $(nproc) logical"
    echo "ram:        $(free -g | awk '/^Mem:/ {print $2 " GiB"}')"
    echo "gpu:        $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
    echo "gpu clocks: $(nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader | head -1) (max SM, max mem, power limit)"
    echo "gpu bus:    PCIe gen $(nvidia-smi --query-gpu=pcie.link.gen.max --format=csv,noheader | head -1) x$(nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader | head -1)"
    echo "cuda arch:  $(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:STRING=//p' "$BUILD/CMakeCache.txt")"
    echo "nvcc:       $(nvcc --version 2>/dev/null | tail -1 || echo n/a)"
    echo "host cxx:   $(g++ --version | head -1)"
    echo "cmake:      $(cmake --version | head -1)"
    echo "on battery: $(cat /sys/class/power_supply/AC*/online 2>/dev/null | grep -q 0 && echo yes || echo no/unknown)"
    echo "settings:   STEPS=$STEPS WARMUP=$WARMUP REPEATS=$REPEATS CPU_STEPS=$CPU_STEPS SIZES='$SIZES'"
    echo
    "$BIN" --benchmark --size 64x64 --steps 1 --warmup 0 | grep '^#' || true
} > "$OUT/hardware.txt"
cat "$OUT/hardware.txt"

RAW="$OUT/raw.csv"
echo "repeat,width,height,steps,mode,total_s,ms_per_step,mlups,steps_per_s,finite" > "$RAW"

# Run one measurement; append "<repeat>,<csv>" only if it produced a CSV line.
# Exit code 3 (NaN in the field) still yields a line marked "nan".
run_case() {
    local repeat=$1; shift
    local line
    line=$("$@" 2>>"$OUT/errors.log" | tail -1) || true
    if [[ "$line" =~ ^[0-9]+,[0-9]+, ]]; then
        echo "$repeat,$line" >> "$RAW"
        return 0
    fi
    return 1
}

for size in $SIZES; do
    w=${size%x*}; h=${size#*x}
    for r in $(seq 1 "$REPEATS"); do
        echo "==> GPU compute  $size  run $r/$REPEATS"
        run_case "$r" "$BIN" --benchmark --csv --size "$size" --steps "$STEPS" --warmup "$WARMUP" ||
            { echo "GPU compute run failed, see $OUT/errors.log" >&2; exit 1; }
        if [ "$RENDER" = 1 ]; then
            echo "==> GPU render   $size  run $r/$REPEATS"
            if ! run_case "$r" "$BIN" --benchmark --render --csv --size "$size" --steps "$STEPS" --warmup "$WARMUP"; then
                echo "Render benchmark failed (no CUDA-OpenGL interop here? WSL does not support it);" \
                     "skipping render runs. Details: $OUT/errors.log" >&2
                RENDER=0
            fi
        fi
    done
    if [ "$CPU" = 1 ]; then
        echo "==> CPU prototype $size ($CPU_STEPS steps)"
        run_case 1 "$BUILD/alpha_cpu_bench" "$w" "$h" "$CPU_STEPS" 2 ||
            echo "CPU prototype run failed for $size, see $OUT/errors.log" >&2
    fi
done

python3 "$ROOT/benchmarks/summarize.py" "$RAW" "$OUT/hardware.txt" > "$OUT/summary.md"
echo
cat "$OUT/summary.md"
echo
echo "Done. Results: $OUT"
