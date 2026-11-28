#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  gpu-unlock Verification & Tensor Core Diagnostics"
echo "============================================================"
echo

echo "[1/4] Kernel command line (/proc/cmdline):"
cat /proc/cmdline
echo

echo "[2/4] Kernel unlock log (gpu_unlock):"
if command -v journalctl >/dev/null 2>&1 && journalctl -k -b --no-pager 2>/dev/null | grep -q gpu_unlock; then
    journalctl -k -b --no-pager 2>/dev/null | grep gpu_unlock
elif dmesg 2>/dev/null | grep -q gpu_unlock; then
    dmesg 2>/dev/null | grep gpu_unlock
else
    echo "(No gpu_unlock lines readable without root; run 'sudo dmesg | grep gpu_unlock' to inspect)"
fi
echo

echo "[3/4] NVIDIA driver status (nvidia-smi):"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,pstate,power.draw,temperature.gpu --format=csv
echo

echo "[4/4] Tensor Core MMA benchmark (FP16 -> FP16 vs FP16 -> FP32):"
python3 "${SCRIPT_DIR}/bench_tensor.py"
