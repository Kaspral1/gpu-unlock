# Technische Dokumentation: Tensor-Core-Freischaltung auf NVIDIA Turing (TU102 / TU104 / TU106 / TU106M)

Consumer-Karten der Serie GeForce RTX 20 (Turing) besitzen dieselben Tensor Cores der 2. Generation wie Quadro RTX und Tesla T4, werden jedoch per Firmware bei **FP16-Eingabe mit FP32-Akkumulation** (`HMMA.1688.F32`) auf 50 % Durchsatz gedrosselt.

`gpu_unlock` ist ein Linux-Kernelmodul, das früh im `initramfs` vor `nvidia.ko` lädt, die Drosselung aufhebt und die Karte an den unveränderten NVIDIA-Treiber übergibt.

> [!WARNING]
> **Nutzung auf eigene Gefahr:** Die Turing-Unterstützung ist experimentell und schreibt beim Systemstart in GPU-Register. Bereitstellung ohne Gewährleistung unter GPL-2.0.

---

## 1. Turing-Register für SM-Geschwindigkeitsselektoren

Anders als bei Ampere (`0x008238xx`) liegen die Turing-Register im `PGRAPH`-Broadcast-Bereich:

| Register | BAR0-Offset | Standardwert | Zielwert (freigeschaltet) | Funktion |
| :--- | :--- | :--- | :--- | :--- |
| `TU_FEAT_PLM` | `0x00409650` | `0xffffff8f` | `0xffffffff` (Schreibfenster) -> `0xffffff8f` (wiederhergestellt) | Privilege Level Mask für `SS0`/`SS1` |
| `SS0` (`FEAT_OVR_SM_SPD`) | `0x0040960c` | `0x00004010` | `0x88888888` | Acht 4-Bit-Geschwindigkeitsselektoren |
| `SS1` (`FEAT_OVR_SM_SPD_1`) | `0x00409634` | `0x00000004` / `0x5` | `0x00000008` | Zusätzlicher Selektor (`IMLA4`) |
| `READOUT` | `0x00409968` | `[25:24]` ungleich 0 | `[25:24]` = `00` | Hardware-Rücklesewert |

---

## 2. Warum PCIe FLR die Freischaltung auf Turing zurücksetzt

Bei Ampere überleben `SS0` und `SS1` ein `pcie_flr()`, das `WPR2` abbaut. Auf **Turing** löst `pcie_flr()` jedoch erneut VBIOS `devinit` aus und setzt `SS0`, `SS1` und `TU_FEAT_PLM` sofort auf die gedrosselten Werkswerte zurück.

### Lösung: Dual-Write V67 ROP + Übergabe ohne FLR

Die Turing-V67-ROP-Kette führt zwei Heavy-Secure-(L3)-Schreibvorgänge in einem SEC2-Booter-Durchlauf aus:
1. `0x00409650 = 0xffffffff`: Öffnet `TU_FEAT_PLM` für Host-MMIO-Zugriffe auf `SS0`/`SS1`.
2. `0x001fa828 = 0x00000000`: Löscht `NV_PFB_PRI_MMU_WPR2_ADDR_HI` direkt aus L3 und deaktiviert `WPR2` ohne PCIe-Reset.

Zusätzlich gibt die ROP-Kette ACR-Mutex 3 frei, setzt `{ 0xf9f8, 0x00000000 }` für `SEC2_MAILBOX0 = 0x0`, löscht DMEM und stoppt SEC2 sauber. Danach schreibt der Host `SS0 = 0x88888888` und `SS1 = 0x8`, sperrt `TU_FEAT_PLM` wieder und übergibt ohne `pcie_flr()` an `nvidia.ko`.

---

## 3. GSP-Falcon- und DMA-Korrekturen

- Auf `TU106M` (`10de:1f15`) bleibt `GSP_CPUCTL` (`0x110100`) nach einem Reset auf `0x10` (`HALTED`), bis `STARTCPU` (`0x2`) geschrieben wird. Der Loader wartet auf das Löschen der `DMACTL`-Scrubbing-Bits (`0x6`) und lehnt `CPUCTL & 0x10` vor dem Start nicht ab.
- Nach dem Start von `FWSEC-FRTS` wird zuerst auf `WPR2_ADDR_HI != 0` gewartet und erst danach auf `GSP_CPUCTL.HALTED`.
- `FWSEC-FRTS` nutzt auf Turing `DMA_BIT_MASK(32)`, der SEC2-Booter `DMA_BIT_MASK(47)`.

---

## 4. Benchmark-Ergebnisse (RTX 2060 Mobile `10de:1f15`)

Getestet auf Acer Nitro AN515-54 (RTX 2060 Mobile, 30 SMs, `TU106M`, Linux `7.0.0-42-generic`, Treiber `615.71.09`) mit `python3 scripts/bench_tensor.py`:

- **FP16 -> FP16 Akkumulation (100 % Referenz):** `58.82 TFLOPS`
- **FP16 -> FP32 Akkumulation (gedrosselt ~28.6 TFLOPS):** `57.17 TFLOPS` (**97,2 % volle Leistung, ca. 2x schneller**)

### Hinweis für Laptops (`D3cold`)

Schaltet Optimus die dGPU im Leerlauf in `D3cold` ab, gehen die Registerwerte bis zum nächsten Neustart verloren:

```sh
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```
