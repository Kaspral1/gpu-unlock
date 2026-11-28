# Documentación técnica: Desbloqueo de Tensor Cores en NVIDIA Turing (TU102 / TU104 / TU106 / TU106M)

Las tarjetas GeForce RTX 20 (Turing) integran los mismos Tensor Cores de 2.ª generación que las Quadro RTX y Tesla T4, pero NVIDIA limita de fábrica el rendimiento de **FP16 con acumulación FP32** (`HMMA.1688.F32`) al 50 %.

`gpu_unlock` es un módulo del kernel de Linux cargado en `initramfs` antes de `nvidia.ko`. Elimina esa limitación en hardware y entrega la GPU al controlador oficial de NVIDIA sin modificarlo ni flashear la VBIOS.

> [!WARNING]
> **Úsalo bajo tu propia responsabilidad:** El soporte para Turing es experimental y modifica registros de bajo nivel al arrancar. Se entrega sin garantías bajo licencia GPL-2.0.

---

## 1. Registros de selectores de velocidad en Turing

A diferencia de Ampere (`0x008238xx`), en Turing estos registros residen en el espacio broadcast de `PGRAPH`:

| Registro | Offset BAR0 | Valor de fábrica | Valor desbloqueado | Función |
| :--- | :--- | :--- | :--- | :--- |
| `TU_FEAT_PLM` | `0x00409650` | `0xffffff8f` | `0xffffffff` (escritura) -> `0xffffff8f` (restaurado) | Máscara de privilegios para `SS0`/`SS1` |
| `SS0` (`FEAT_OVR_SM_SPD`) | `0x0040960c` | `0x00004010` | `0x88888888` | Ocho selectores de velocidad de 4 bits |
| `SS1` (`FEAT_OVR_SM_SPD_1`) | `0x00409634` | `0x00000004` / `0x5` | `0x00000008` | Selector adicional (`IMLA4`) |
| `READOUT` | `0x00409968` | `[25:24]` distinto de 0 | `[25:24]` = `00` | Lectura de verificación en hardware |

---

## 2. Por qué PCIe FLR rompe el desbloqueo en Turing

En Ampere, `SS0` y `SS1` sobreviven a `pcie_flr()`. En **Turing**, ejecutar `pcie_flr()` vuelve a lanzar `devinit` de la VBIOS y restaura inmediatamente `SS0`, `SS1` y `TU_FEAT_PLM` a los valores limitados.

### Solución: ROP V67 de doble escritura y entrega sin FLR

La cadena ROP V67 de Turing realiza dos escrituras Heavy-Secure (L3) en una sola ejecución del SEC2 Booter:
1. `0x00409650 = 0xffffffff`: Abre `TU_FEAT_PLM` para permitir escritura MMIO en `SS0`/`SS1`.
2. `0x001fa828 = 0x00000000`: Pone a cero `NV_PFB_PRI_MMU_WPR2_ADDR_HI` directamente desde L3, desactivando `WPR2` sin reinicio PCIe.

Además libera el mutex ACR 3, limpia `{ 0xf9f8, 0x00000000 }` para dejar `SEC2_MAILBOX0 = 0x0`, borra DMEM y detiene SEC2. El host escribe `SS0 = 0x88888888` y `SS1 = 0x8`, vuelve a cerrar `TU_FEAT_PLM` y entrega el dispositivo a `nvidia.ko` sin `pcie_flr()`.

---

## 3. Correcciones de GSP Falcon y DMA

- En `TU106M` (`10de:1f15`), tras reiniciar GSP Falcon (`0x110000`), `GSP_CPUCTL` (`0x110100`) devuelve `0x10` (`HALTED`) hasta recibir `STARTCPU` (`0x2`). El cargador espera la limpieza de `DMACTL` (`0x6`) y no rechaza `CPUCTL & 0x10` antes del arranque.
- Tras iniciar `FWSEC-FRTS`, se comprueba primero que `WPR2_ADDR_HI` (`0x1fa828`) sea distinto de cero antes de esperar a `GSP_CPUCTL.HALTED`.
- `FWSEC-FRTS` usa máscara DMA coherente de 32 bits (`DMA_BIT_MASK(32)`) y SEC2 Booter usa 47 bits (`DMA_BIT_MASK(47)`).

---

## 4. Verificación en RTX 2060 Mobile (`10de:1f15`)

Probado en Acer Nitro AN515-54 (RTX 2060 Mobile, 30 SMs, `TU106M`, Linux `7.0.0-42-generic`, driver `615.71.09`) con `python3 scripts/bench_tensor.py`:

- **FP16 -> FP16 acumulación (referencia 100 %):** `58.82 TFLOPS`
- **FP16 -> FP32 acumulación (limitado ~28.6 TFLOPS):** `57.17 TFLOPS` (**97,2 % de velocidad completa, ~2x**)

### Nota para portátiles (`D3cold`)

Si el portátil apaga la GPU dedicada en reposo (`D3cold`), el desbloqueo se pierde hasta el siguiente reinicio:

```sh
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```
