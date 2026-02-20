#!/usr/bin/env bash
# Run all sweep configs sequentially on the current machine.
# Usage (on the pod):  ./scripts/sweep.sh
# Usage (from Mac):    ./scripts/run_remote.sh '' scripts/sweep.sh
set -euo pipefail

CONFIGS=(configs/sweep_*.yaml)

if [ "${#CONFIGS[@]}" -eq 0 ] || [ "${CONFIGS[0]}" = "configs/sweep_*.yaml" ]; then
    echo "ERROR: No configs/sweep_*.yaml files found."
    exit 1
fi

echo "=== Sweep: ${#CONFIGS[@]} configs ==="
for cfg in "${CONFIGS[@]}"; do
    echo "$cfg"
done
echo ""

FAILED=()
for cfg in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "=== Starting: $cfg ==="
    echo "========================================"
    if uv run python src/train.py "$cfg"; then
        echo "=== Completed: $cfg ==="
    else
        echo "=== FAILED: $cfg ==="
        FAILED+=("$cfg")
    fi
    echo ""
done

echo "========================================"
echo "=== Sweep complete ==="
echo "  Total:  ${#CONFIGS[@]}"
echo "  Failed: ${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
    exit 1
fi
