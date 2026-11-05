# How NVIDIA Turing (TU102 / TU104 / TU106 / TU106M) GPUs Are Unlocked

While NVIDIA Ampere (RTX 30) and Turing (RTX 20) share the same high-level SEC2 Booter stack-overflow vulnerability (V67), the hardware register layout, Falcon engine behavior, DMA addressing constraints, and post-unlock handoff requirements on **Turing (`TU102`/`TU104`/`TU106`, including mobile `TU106M` RTX 2060 `10de:1f15`)** differ significantly from Ampere.

This document details all architectural differences and fixes required to make `gpu_unlock` work reliably on Turing GPUs and seamlessly hand off the unlocked card to the stock NVIDIA Linux driver (`nvidia.ko`).

> [!WARNING]
> **Use at your own risk:** Turing support is experimental and interacts with low-level GPU registers via a kernel module. This software is provided **"as is"**, without warranty of any kind, under **GPL-2.0**. You use it entirely at your own risk.

---

## 1. Turing SM Speed Selectors and Privilege Level Mask (PLM)

Unlike Ampere (where the speed selectors live in `0x008238xx`), Turing's SM instruction issue-rate overrides live in the `PGRAPH` broadcast space:

| Register | BAR0 Offset | Stock GeForce Value | Target Unlocked Value | Role |
| :--- | :--- | :--- | :--- | :--- |
| `TU_FEAT_PLM` (`NV_PGRAPH_PRI_GPCS_TPCS_SM_SCH_MACRO_SCHED_LOCK`) | `0x00409650` | `0xffffff8f` | `0xffffffff` (during write) → `0xffffff8f` (restored) | Privilege Level Mask governing writes to `SS0`/`SS1` |
| `SS0` (`FEAT_OVR_SM_SPD`) | `0x0040960c` | `0x00004010` | `0x88888888` | Eight 4-bit speed selector nibbles (FMLA16, FMLA32, etc.) |
| `SS1` (`FEAT_OVR_SM_SPD_1`) | `0x00409634` | `0x00000004` / `0x5` | `0x00000008` | Additional speed selector nibble (`IMLA4`) |
| `READOUT` | `0x00409968` | non-zero at `[25:24]` | `0x00000000` at `[25:24]` | Hardware readout confirming full-rate Tensor Core operation |

---

## 2. Why PCIe Function Level Reset (`pcie_flr`) Breaks Turing Unlock

On Ampere (`GA102`/`GA104`), `gpu_unlock` triggers a PCIe Function Level Reset (`pcie_flr()`) at the end of the unlock transaction to tear down the `WPR2` protected memory region before `nvidia.ko` probes the device; on Ampere, `SS0` and `SS1` survive `pcie_flr()`.

**On Turing, `pcie_flr()` triggers VBIOS `devinit`, which immediately resets `SS0` (`0x4010`), `SS1` (`0x4`), and `TU_FEAT_PLM` (`0xffffff8f`) back to their factory-capped state.**

### Solution: Dual-Write V67 ROP Payload + No-FLR Handoff
Instead of resetting the GPU via `pcie_flr()`, Turing's V67 ROP chain performs **two** Heavy-Secure (L3) writes in a single SEC2 Booter execution:
1. **`0x00409650 = 0xffffffff`** — Opens `TU_FEAT_PLM` so the host driver can write `SS0` and `SS1` over BAR0 MMIO.
2. **`0x001fa828 = 0x00000000`** — Clears `NV_PFB_PRI_MMU_WPR2_ADDR_HI` directly from Heavy-Secure mode, disabling `WPR2` without needing a PCIe reset!

Furthermore, the Turing V67 ROP chain:
- Releases hardware ACR mutex 3 (`SEMA = 0`, `RELEASE = 0x08`),
- Clears `{ 0xf9f8, 0x00000000 }` (`main_sp + 0x08 = 0`) so that the Booter epilogue at `0x7ea4` writes `0x0` (success) to `SEC2_MAILBOX0` instead of the residual signature verification error code `0x31`,
- Scrubs DMEM and halts `SEC2` cleanly.

After the host writes `SS0 = 0x88888888` and `SS1 = 0x8`, it relocks `TU_FEAT_PLM` (`0x00409650 = 0xffffff8f`, readback `0xffffffef`) and hands the device off to `nvidia.ko` **without** `pcie_flr()`.

---

## 3. GSP Falcon `CPUCTL.HALTED` Behavior on Turing (`TU106M`)

On `TU106M` (`10de:1f15`), resetting the GSP Falcon engine (`0x110000`) leaves `GSP_CPUCTL` (`0x110100`) reading `0x10` (`HALTED == 1`) until `STARTCPU` (`0x2`) is written.
This caused two failure modes in the stock Ampere-oriented Falcon loader:
1. Checking `(ctl & 16)` right after `gu_falcon_reset()` rejected the engine with `-ENODEV` (`-19`).
2. Polling `GSP_CPUCTL` for `HALTED == 1` immediately after writing `STARTCPU` could return before the Falcon actually cleared the pre-start `HALTED` bit.

### Solution
- Wait for `DMACTL` scrubbing bits (`0x6`) to clear after reset and do not reject `CPUCTL & 0x10` prior to `STARTCPU`.
- Configure `FBIF_CTL` (`bit 7`) and `TRANSCFG(0)` (`physical coherent sysmem`) before starting the Falcon.
- After issuing `STARTCPU` for `FWSEC-FRTS`, poll `WPR2_ADDR_HI` (`0x1fa828`) to become non-zero first (proving `FWSEC-FRTS` has booted and provisioned WPR2), and only then poll `GSP_CPUCTL` for the post-execution `HALTED` state.
- Constrain `FWSEC` DMA allocations on Turing to a 32-bit coherent mask (`DMA_BIT_MASK(32)`), then widen back to 47-bit (`DMA_BIT_MASK(47)`) for the SEC2 Booter DMA buffers.

---

## 4. Benchmark Verification (RTX 2060 Mobile `10de:1f15`)

Tested on **Acer Nitro AN515-54** (NVIDIA GeForce RTX 2060 Mobile, 30 SMs, `TU106M`, Linux kernel `7.0.0-42-generic`, driver `615.71.09`) using `scripts/bench_tensor.py` (`mma.sync.aligned.m16n8k8`):

- **FP16 → FP16 accumulate (unthrottled reference):** `58.82 TFLOPS`
- **FP16 → FP32 accumulate (stock capped at ~28.6 TFLOPS):** `57.17 TFLOPS` (**97.2% of full hardware speed, ~2× speedup**)

---

## 5. Support

If this Turing unlock and documentation helped you, you can support the author of this fork:

- **Buy Me a Coffee:** [https://buymeacoffee.com/kaspral](https://buymeacoffee.com/kaspral)
- **BTC:** `bc1pdc8clusv5hpgz7laywwzhjjcklk6x2cgj2xu5uc9xes9fukxaegsn7cs20`
- **ETH:** `0x006dAcBA1344144AB395D02637045Dd19aB54C2E`
- **SOL:** `96Lc4X6hieoCkUFHtDW4wvbWXgZvebqTktjRNiCNe1ex`

