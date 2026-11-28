# Documentation technique : Déblocage des Tensor Cores sur NVIDIA Turing (TU102 / TU104 / TU106 / TU106M)

Les cartes GeForce RTX 20 (Turing) utilisent les mêmes Tensor Cores de 2e génération que les Quadro RTX et Tesla T4, mais leur débit en **FP16 avec accumulation FP32** (`HMMA.1688.F32`) est bridé à 50 % par le firmware.

`gpu_unlock` est un module noyau Linux chargé tôt dans l'`initramfs`, avant que `nvidia.ko` ne prenne la carte. Il lève la limitation matérielle puis passe la main au pilote officiel NVIDIA sans patch ni flash VBIOS.

> [!WARNING]
> **Utilisation à vos risques et périls :** Le support Turing est expérimental et modifie des registres GPU bas niveau au démarrage. Fourni tel quel, sans garantie, sous licence GPL-2.0.

---

## 1. Registres de sélection de vitesse sur Turing

Contrairement à Ampere (`0x008238xx`), les registres Turing se trouvent dans l'espace broadcast `PGRAPH` :

| Registre | Offset BAR0 | Valeur d'origine | Valeur débloquée | Rôle |
| :--- | :--- | :--- | :--- | :--- |
| `TU_FEAT_PLM` | `0x00409650` | `0xffffff8f` | `0xffffffff` (écriture) -> `0xffffff8f` (restauré) | Masque de privilège pour `SS0`/`SS1` |
| `SS0` (`FEAT_OVR_SM_SPD`) | `0x0040960c` | `0x00004010` | `0x88888888` | Huit sélecteurs de vitesse sur 4 bits |
| `SS1` (`FEAT_OVR_SM_SPD_1`) | `0x00409634` | `0x00000004` / `0x5` | `0x00000008` | Sélecteur supplémentaire (`IMLA4`) |
| `READOUT` | `0x00409968` | `[25:24]` non nul | `[25:24]` = `00` | Lecture matérielle de vérification |

---

## 2. Pourquoi le PCIe FLR annule le déblocage sur Turing

Sur Ampere, `SS0` et `SS1` survivent à `pcie_flr()`. Sur **Turing**, `pcie_flr()` relance `devinit` du VBIOS, ce qui réinitialise immédiatement `SS0`, `SS1` et `TU_FEAT_PLM` aux valeurs bridées.

### Solution : ROP V67 à double écriture et passage sans FLR

La chaîne ROP V67 pour Turing effectue deux écritures Heavy-Secure (L3) en une seule exécution du SEC2 Booter :
1. `0x00409650 = 0xffffffff` : Ouvre `TU_FEAT_PLM` pour autoriser l'écriture MMIO de `SS0`/`SS1`.
2. `0x001fa828 = 0x00000000` : Efface `NV_PFB_PRI_MMU_WPR2_ADDR_HI` directement en L3 pour désactiver `WPR2` sans reset PCIe.

La chaîne libère aussi le mutex ACR 3, remet `{ 0xf9f8, 0x00000000 }` à zéro (`SEC2_MAILBOX0 = 0x0`), nettoie la DMEM et arrête SEC2 proprement. L'hôte écrit ensuite `SS0 = 0x88888888` et `SS1 = 0x8`, reverrouille `TU_FEAT_PLM` et remet la carte à `nvidia.ko` sans `pcie_flr()`.

---

## 3. Correctifs GSP Falcon et masques DMA

- Sur `TU106M` (`10de:1f15`), après réinitialisation du GSP Falcon (`0x110000`), `GSP_CPUCTL` (`0x110100`) reste à `0x10` (`HALTED`) jusqu'à l'écriture de `STARTCPU` (`0x2`). Le chargeur attend l'effacement des bits `DMACTL` (`0x6`) sans rejeter `CPUCTL & 0x10` avant le démarrage.
- Après lancement de `FWSEC-FRTS`, le module attend d'abord `WPR2_ADDR_HI != 0` puis l'état `HALTED` dans `GSP_CPUCTL`.
- `FWSEC-FRTS` utilise un masque DMA cohérent 32 bits (`DMA_BIT_MASK(32)`), tandis que SEC2 Booter utilise 47 bits (`DMA_BIT_MASK(47)`).

---

## 4. Résultats sur RTX 2060 Mobile (`10de:1f15`)

Testé sur Acer Nitro AN515-54 (RTX 2060 Mobile, 30 SMs, `TU106M`, Linux `7.0.0-42-generic`, pilote `615.71.09`) avec `python3 scripts/bench_tensor.py` :

- **Accumulation FP16 -> FP16 (référence 100 %) :** `58.82 TFLOPS`
- **Accumulation FP16 -> FP32 (bridé ~28.6 TFLOPS) :** `57.17 TFLOPS` (**97,2 % du débit maximal, ~2x**)

### Note pour ordinateurs portables (`D3cold`)

Si Optimus coupe complètement l'alimentation du GPU dédié (`D3cold`), le déblocage est perdu jusqu'au redémarrage suivant :

```sh
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```
