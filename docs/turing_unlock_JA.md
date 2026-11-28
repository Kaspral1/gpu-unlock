# NVIDIA Turing (TU102 / TU104 / TU106 / TU106M) Tensor Core アンロック技術解説

GeForce RTX 20 シリーズ (Turing) は Quadro RTX や Tesla T4 と同じ第 2 世代 Tensor Core を搭載していますが、出荷時ファームウェアにより **FP16 入力 / FP32 累積** (`HMMA.1688.F32`) のスループットが 50% に制限されています。

`gpu_unlock` は `nvidia.ko` がデバイスをバインドする前の `initramfs` 段階でロードされる Linux カーネルモジュールです。ドライバの改変や VBIOS の書き換えなしに制限を解除し、標準の NVIDIA ドライバへ引き継ぎます。

> [!WARNING]
> **自己責任で使用してください:** Turing 対応は実験的であり、起動時に GPU の低レベルレジスタを書き換えます。GPL-2.0 ライセンスの下で無保証で提供されます。

---

## 1. Turing の SM 速度セレクタレジスタ

Ampere (`0x008238xx`) とは異なり、Turing のレジスタは `PGRAPH` ブロードキャスト空間に配置されています。

| レジスタ | BAR0 オフセット | 初期値 | 解除後の値 | 役割 |
| :--- | :--- | :--- | :--- | :--- |
| `TU_FEAT_PLM` | `0x00409650` | `0xffffff8f` | `0xffffffff` (書込時) -> `0xffffff8f` (復元) | `SS0`/`SS1` 書込権限マスク |
| `SS0` (`FEAT_OVR_SM_SPD`) | `0x0040960c` | `0x00004010` | `0x88888888` | 4-bit 速度セレクタ x 8 |
| `SS1` (`FEAT_OVR_SM_SPD_1`) | `0x00409634` | `0x00000004` / `0x5` | `0x00000008` | 追加セレクタ (`IMLA4`) |
| `READOUT` | `0x00409968` | `[25:24]` 非ゼロ | `[25:24]` = `00` | ハードウェア確認用リードアウト |

---

## 2. Turing で PCIe FLR を使わない理由

Ampere では `pcie_flr()` を実行しても `SS0`/`SS1` が保持されますが、**Turing** では `pcie_flr()` により VBIOS `devinit` が再実行され、`SS0` / `SS1` / `TU_FEAT_PLM` が即座に初期制限値へ戻ってしまいます。

### 解決策: デュアルライト V67 ROP + No-FLR ハンドオフ

Turing 用 V67 ROP チェーンは、1 回の SEC2 Booter Heavy-Secure (L3) 実行で 2 つの書き込みを行います。
1. `0x00409650 = 0xffffffff`: `TU_FEAT_PLM` を開放し、ホスト MMIO から `SS0`/`SS1` を書き込めるようにします。
2. `0x001fa828 = 0x00000000`: L3 から直接 `NV_PFB_PRI_MMU_WPR2_ADDR_HI` を 0 クリアし、PCIe リセットなしで `WPR2` を無効化します。

その後 ACR ミューテックス 3 を解放し、`{ 0xf9f8, 0x00000000 }` をクリアして `SEC2_MAILBOX0 = 0x0` で正常停止します。ホスト側で `SS0 = 0x88888888` と `SS1 = 0x8` を書き込み、`TU_FEAT_PLM` を再ロックしたあと `pcie_flr()` なしで `nvidia.ko` に引き継ぎます。

---

## 3. GSP Falcon および DMA マスクの修正

- `TU106M` (`10de:1f15`) では GSP Falcon (`0x110000`) リセット直後に `GSP_CPUCTL` (`0x110100`) が `0x10` (`HALTED`) を返すため、起動前にエラー扱いせず `DMACTL` スクラブ完了 (`0x6`) を待ちます。
- `FWSEC-FRTS` 起動後は先に `WPR2_ADDR_HI` (`0x1fa828`) が非ゼロになるのをポーリングしてから `GSP_CPUCTL.HALTED` を確認します。
- Turing では `FWSEC-FRTS` に 32-bit (`DMA_BIT_MASK(32)`)、SEC2 Booter に 47-bit (`DMA_BIT_MASK(47)`) のコヒーレント DMA マスクを使用します。

---

## 4. ベンチマーク結果 (RTX 2060 Mobile `10de:1f15`)

検証環境: Acer Nitro AN515-54 (RTX 2060 Mobile, 30 SMs, `TU106M`, Linux `7.0.0-42-generic`, NVIDIA `615.71.09`), `python3 scripts/bench_tensor.py`:

- **FP16 -> FP16 累積 (100% 基準):** `58.82 TFLOPS`
- **FP16 -> FP32 累積 (制限時 ~28.6 TFLOPS):** `57.17 TFLOPS` (**フル速度の 97.2%、約 2 倍**)

### ノート PC 向け `D3cold` 注意事項

Optimus 環境でアイドル時に dGPU の電源が完全に落ちる (`D3cold`) と、次回再起動まで制限状態に戻ります。ランタイム電源管理を無効化してください。

```sh
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```
