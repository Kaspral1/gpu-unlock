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

## 4. Benchmark Verification & Boot Log (RTX 2060 Mobile `10de:1f15`)

Tested on **Acer Nitro AN515-54** (NVIDIA GeForce RTX 2060 Mobile, 30 SMs, `TU106M`, Linux kernel `7.0.0-42-generic`, driver `615.71.09`) using `scripts/bench_tensor.py` (`mma.sync.aligned.m16n8k8`):

- **FP16 → FP16 accumulate (unthrottled reference):** `58.82 TFLOPS`
- **FP16 → FP32 accumulate (stock capped at ~28.6 TFLOPS):** `57.17 TFLOPS` (**97.2% of full hardware speed, ~2× speedup**)

### Expected `dmesg` Boot Sequence (`sudo dmesg | grep gpu_unlock`)

```text
[    0.980499] pci 0000:01:00.0: gpu_unlock: stage=claim device=0x1f15 family=TU102/TU104/TU106 cmd=0x102 original=0x102 bar_len=0x1000000 region=1 enabled=0 master=0
[    0.980510] pci 0000:01:00.0: gpu_unlock: stage=layout fb=0x180000000 vga=0x17ff00000 frts=0x100000 wpr=0x177800000..0x17ff00000 boot=0x17fdff000 gsp=0x17e200000 heap=0x177900000
[    0.991917] pci 0000:01:00.0: gpu_unlock: stage=dma booter=59136 payload=0xfa00 pages=7179 meta=0xfffcf000 sig=0xfffd0000 stock=0xfffce000 boot=0xfffcd000
[    0.991930] pci 0000:01:00.0: gpu_unlock: stage=gfw plm=0xffffff8f ss0=0x4010 ss1=0x4 wpr2_hi=0x0
[    1.353047] pci 0000:01:00.0: gpu_unlock: stage=fwsec version=2 fuse=0x0 ver=0 scratch=0x0 wpr2_raw=0x17fe000:0x17fee00 frts=0x17fe000:0x17fee00
[    1.353071] pci 0000:01:00.0: gpu_unlock: stage=publish meta=0xfffcf000 sig=0xfffd0000 size=0xfa00 radix=0xfffcc000
[    1.390038] pci 0000:01:00.0: gpu_unlock: pass=1/2 attempt=1 target=0x409650 value=0xffffffff before=0xffffff8f after=0xffffffff wpr2=0x1778000:0x0 reset_plm=0xff mbox=0x0:0x0 ret=0
[    1.390062] pci 0000:01:00.0: gpu_unlock: pass=2/2 attempt=1 target=0x1fa828 value=0x0 before=0xffffff8f after=0x0 wpr2=0x1778000:0x0 reset_plm=0xff mbox=0x0:0x0 ret=0
[    1.390080] pci 0000:01:00.0: gpu_unlock: stage=host ss0=0x88888888 ss1=0x8 plm_feat=0xffffffff plm_wpr_cfg=0xbadf5040
[    1.390103] pci 0000:01:00.0: gpu_unlock: stage=quiesce-turing wpr2=0x1778000:0x0 ss0=0x88888888 ss1=0x8 readout=0x0 plm=0xffffffef cmd=0x102 devsta=0x19 chip=0x166000a1
[    1.406221] pci 0000:01:00.0: gpu_unlock: stage=handoff ret=0 state=4 retained=0 pages=0/7179 cleanup=0
[    1.406233] pci 0000:01:00.0: gpu_unlock: unlocked
```

### Laptop Note (Preventing `D3cold` Power-Off Reset)

On Optimus laptops, runtime power management can power down the dGPU into `D3cold` when idle, which silently resets `SS0`/`SS1` back to factory defaults until the next reboot. To prevent this:

```sh
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```

---

## 5. Support

If this Turing unlock and documentation helped you, you can support the author of this fork:

- **Buy Me a Coffee:** [https://buymeacoffee.com/kaspral](https://buymeacoffee.com/kaspral)
- **BTC:** `bc1pdc8clusv5hpgz7laywwzhjjcklk6x2cgj2xu5uc9xes9fukxaegsn7cs20`
- **ETH:** `0x006dAcBA1344144AB395D02637045Dd19aB54C2E`
- **SOL:** `96Lc4X6hieoCkUFHtDW4wvbWXgZvebqTktjRNiCNe1ex`

