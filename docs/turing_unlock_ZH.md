# NVIDIA Turing (TU102 / TU104 / TU106 / TU106M) Tensor Core 解锁技术说明

NVIDIA GeForce RTX 20 系列 (Turing) 与 Quadro RTX / Tesla T4 使用相同的第 2 代 Tensor Core 硬件，但在出厂固件中被限制为 **FP16 输入 + FP32 累加** (`HMMA.1688.F32`) 半速运行。

`gpu_unlock` 在系统启动早期 (`initramfs`)、官方 `nvidia.ko` 绑定显卡之前解除该限制，无需修改驱动或刷写 VBIOS。

> [!WARNING]
> **风险提示：** Turing 支持仍处于实验阶段，会在启动时通过内核模块修改底层 GPU 寄存器。本软件按 GPL-2.0 协议按原样提供，使用风险自负。

---

## 1. Turing 寄存器布局

与 Ampere (`0x008238xx`) 不同，Turing 的 SM 速度选择器位于 `PGRAPH` 广播空间：

| 寄存器 | BAR0 偏移 | 出厂默认值 | 解锁目标值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `TU_FEAT_PLM` | `0x00409650` | `0xffffff8f` | `0xffffffff` (写入期间) -> `0xffffff8f` (恢复) | 控制 `SS0`/`SS1` 写入权限的 PLM |
| `SS0` (`FEAT_OVR_SM_SPD`) | `0x0040960c` | `0x00004010` | `0x88888888` | 8 组 4-bit 速度选择器 |
| `SS1` (`FEAT_OVR_SM_SPD_1`) | `0x00409634` | `0x00000004` / `0x5` | `0x00000008` | 附加速度选择器 (`IMLA4`) |
| `READOUT` | `0x00409968` | `[25:24]` 非零 | `[25:24]` 为 `00` | 硬件状态回读 |

---

## 2. 为什么 Turing 不能在解锁后执行 PCIe FLR

在 Ampere 上，解锁后调用 `pcie_flr()` 可清除 `WPR2` 且 `SS0`/`SS1` 会保留。但在 **Turing** 上，`pcie_flr()` 会重新触发 VBIOS `devinit`，立即把 `SS0`、`SS1` 和 `TU_FEAT_PLM` 恢复为出厂限速状态。

### 解决方案：双写 V67 ROP + 无 FLR 交接

Turing 的 V67 ROP 链在单次 SEC2 Booter 重安全 (L3) 执行中完成两次写入：
1. `0x00409650 = 0xffffffff`：打开 `TU_FEAT_PLM`，允许主机通过 BAR0 MMIO 写入 `SS0`/`SS1`。
2. `0x001fa828 = 0x00000000`：直接从 L3 清零 `NV_PFB_PRI_MMU_WPR2_ADDR_HI`，无需 PCIe 复位即可关闭 `WPR2`。

同时 ROP 链会释放硬件 ACR 互斥锁 3、清零 `{ 0xf9f8, 0x00000000 }` 使 `SEC2_MAILBOX0` 返回 `0x0`、清理 DMEM 并正常停机。主机写入 `SS0 = 0x88888888` 与 `SS1 = 0x8` 后重新锁定 `TU_FEAT_PLM`，不调用 `pcie_flr()` 直接交接给 `nvidia.ko`。

---

## 3. GSP Falcon 与 DMA 掩码修复

- 在 `TU106M` (`10de:1f15`) 上，复位 GSP Falcon (`0x110000`) 后 `GSP_CPUCTL` (`0x110100`) 在写入 `STARTCPU` (`0x2`) 前保持 `0x10` (`HALTED`)。加载器需等待 `DMACTL` 清理位 (`0x6`) 清零，且在启动前不将 `CPUCTL & 0x10` 视为错误。
- 启动 `FWSEC-FRTS` 后，先轮询 `WPR2_ADDR_HI` (`0x1fa828`) 变为非零，再等待 `GSP_CPUCTL` 进入 `HALTED`。
- Turing 的 `FWSEC-FRTS` 使用 32-bit 一致性 DMA 掩码 (`DMA_BIT_MASK(32)`)，`SEC2 Booter` 使用 47-bit 掩码 (`DMA_BIT_MASK(47)`)。

---

## 4. 实测结果 (RTX 2060 Mobile `10de:1f15`)

测试平台：Acer Nitro AN515-54 (RTX 2060 Mobile, 30 SMs, `TU106M`, Linux `7.0.0-42-generic`, NVIDIA `615.71.09`)，`python3 scripts/bench_tensor.py`：

- **FP16 -> FP16 累加 (100% 参考):** `58.82 TFLOPS`
- **FP16 -> FP32 累加 (出厂限速 ~28.6 TFLOPS):** `57.17 TFLOPS` (**97.2% 满速，约 2 倍提升**)

### 笔记本 `D3cold` 注意事项

双显卡笔记本在空闲时若将独显断电进入 `D3cold`，硬件会重置 `SS0`/`SS1`。可禁用运行时电源管理以保持解锁状态：

```sh
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```
