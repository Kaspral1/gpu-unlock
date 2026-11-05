#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Diagnostyka odblokowania NVIDIA GeForce RTX 2060 Mobile"
echo "============================================================"
echo

echo "[1/3] Parametry startowe jądra (/proc/cmdline):"
cat /proc/cmdline
echo

echo "[2/3] Status sterownika NVIDIA (nvidia-smi):"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,pstate,power.draw,temperature.gpu --format=csv
echo

echo "[3/3] Benchmark rdzeni Tensor Core (FP16 -> FP16 vs FP16 -> FP32):"
python3 "${SCRIPT_DIR}/bench_tensor.py"
