# Raport i Dokumentacja Techniczna: Odblokowanie Pełnej Wydajności Tensor Core w NVIDIA GeForce RTX 2060 Mobile (Turing TU106M)

**Data opracowania:** 5 listopada 2026 r.  
**Sprzęt testowy:** Acer Nitro AN515-54  
**Karta graficzna:** NVIDIA GeForce RTX 2060 Mobile (`10de:1f15`, układ `TU106M`, identyfikator `0x166000a1`, 6 GiB GDDR6)  
**System operacyjny:** Linux Mint 22.3 (jądro `7.0.0-42-generic`, sterownik NVIDIA `615.71.09`)  
**Bazowe projekty:** [`Zagitta/gpu-unlock`](https://github.com/Zagitta/gpu-unlock), [`nlzy/turing-unlock`](https://github.com/nlzy/turing-unlock), [`physicsdolphin/nv-uefi-unlocker#82`](https://github.com/physicsdolphin/nv-uefi-unlocker/issues/82)

---

## 1. Streszczenie i Wyniki (Przed vs Po)

Karty konsumenckie z serii **GeForce RTX 20 (Turing)** posiadają fizycznie te same rdzenie **Tensor Core (2. generacji)** co profesjonalne akceleratory **Quadro RTX** oraz **Tesla T4**. Firma NVIDIA nakłada jednak w oprogramowaniu układowym (vBIOS / `devinit`) sztuczną blokadę zmniejszającą o **50%** przepustowość instrukcji macierzowych **FP16 z akumulacją FP32** (`HMMA.1688.F32` / PTX `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`).

Dzięki zmodyfikowanemu modułowi jądra `gpu_unlock`, ładowanemu we wczesnej fazie startu systemu (`initramfs`), blokada zostaje zdjęta przed załadowaniem oficjalnego sterownika `nvidia.ko`.

### Wyniki pomiaru wydajności Tensor Core (`bench_tensor.py`)

| Tryb obliczeń (instrukcja PTX `sm_75`) | Przed odblokowaniem (fabryczny GeForce) | Po odblokowaniu (`gpu_unlock`) | Przyrost |
| :--- | :---: | :---: | :---: |
| **FP16 → FP16 accumulate** (`f16.f16.f16.f16`, referencja) | ~58.8 TFLOPS (100%) | **58.82 TFLOPS** (100%) | — (brak blokady) |
| **FP16 → FP32 accumulate** (`f32.f16.f16.f32`, AI / PyTorch / LLM) | ~28.6 TFLOPS (~50%) | **57.17 TFLOPS** (**97.2%**) | **+99.9% (~2× szybciej)** |

---

## 2. Co daje odblokowanie w praktyce (a gdzie nie ma wpływu)?

### Gdzie zyskujesz od +25% do +100% wydajności:
1. **Lokalne modele językowe (LLM – vLLM, PyTorch, HuggingFace Transformers, Ollama / llama.cpp przy długim kontekście):**
   - Szybsze przetwarzanie promptu (*Prompt Evaluation / Prefill*) oraz krótszy czas do pierwszego tokenu (*TTFT*).
2. **Generowanie obrazów i wideo (Stable Diffusion 1.5 / SDXL, ComfyUI, Flux FP16, Wan):**
   - Operacje konwolucji i uwagi (*FlashAttention / SDPA*) w precyzji FP16 z akumulacją FP32 przyspieszają o **30–55%**.
3. **Trenowanie i douczanie modeli (LoRA / QLoRA / Mixed Precision AMP w PyTorch):**
   - `torch.cuda.amp.autocast(dtype=torch.float16)` domyślnie używa instrukcji Tensor Core z akumulacją FP32 — tutaj zysk jest bezpośredni.

### Gdzie wynik się NIE zmienia:
- **Gry komputerowe i benchmarki graficzne 3D (np. FurMark, 3DMark, Unigine):**
  - FurMark obciąża klasyczne jednostki **CUDA Core (FP32)** oraz rasteryzator OpenGL/Vulkan, które fabrycznie działają na 100% możliwości układu `TU106`. Odblokowanie nie zmienia zegarów ani limitu mocy (TGP), więc wynik w FurMarku pozostaje taki sam.

---

## 3. Analiza Techniczna: Dlaczego standardowy `gpu-unlock` nie działał na RTX 2060 Mobile i jak to naprawiliśmy?

Oryginalne repozytorium `Zagitta/gpu-unlock` zostało zaprojektowane głównie pod architekturę **Ampere (RTX 30)**, a wsparcie dla architektury **Turing (RTX 20)** było niekompletne i nie uwzględniało specyfiki układu mobilnego `TU106M` (`10de:1f15`).

W trakcie diagnostyki rozwiązaliśmy **6 kluczowych problemów sprzętowo-programowych**:

### Problem 1: Brak identyfikatora PCI `0x1f15` (RTX 2060 Mobile)
- **Przyczyna:** W pliku `kernel/gpu_unlock/recipes.c` tablica `gu_cards[]` zawierała wyłącznie identyfikatory desktopowego RTX 2060 (`0x1f08`, `0x1f09`, `0x1f0a`, `0x1f12`) oraz wariant Max-Q (`0x1f55`), pomijając standardowy układ mobilny `0x1f15`.
- **Rozwiązanie:** Dodano wpis `GU_CARD(0x1f15, tu102)` w `recipes.c`.

### Problem 2: Różnice w rejestrach selektorów prędkości SM na architekturze Turing (`TU102`/`TU104`/`TU106`)
- **Przyczyna:** Na architekturze Turing rejestry sterujące przepustowością Tensor Core znajdują się pod innymi adresami BAR0 niż na Ampere:
  - `NV_PGRAPH_PRI_GPCS_TPCS_SM_SCH_MACRO_SCHED_LOCK` (`TU_FEAT_PLM`) = `0x00409650` (domyślnie `0xffffff8f` — zapis tylko z poziomu Heavy-Secure L3)
  - `SS0` (`FEAT_OVR_SM_SPD`) = `0x0040960c` (fabrycznie `0x00004010` — przycięcie FMLA32 do połowy prędkości)
  - `SS1` (`FEAT_OVR_SM_SPD_1`) = `0x00409634` (fabrycznie `0x00000004` lub `0x00000005`)
  - Odczyt weryfikacyjny (`READOUT`) = `0x00409968` (bity `[25:24]` muszą wynosić `00` po odblokowaniu).

### Problem 3: Błąd `-19` (`-ENODEV`) w `unlock_transaction_fwsec` (`last_mmio=0x110100:0x10`)
- **Objaw w `dmesg`:**
  ```text
  gpu_unlock: failure unlock_transaction_fwsec:951 error=-19 family=TU102/TU104/TU106 ... last_mmio=0x110100:0x10
  ```
- **Przyczyna:** Na układzie `TU106M` po resecie silnika GSP Falcon (`0x110000`) rejestr `GSP_CPUCTL` (`0x110100`) zwraca wartość `0x10` (`HALTED == 1`), dopóki procesor nie otrzyma komendy `STARTCPU` (`0x2`). Funkcja `gu_falcon_loader()` w `falcon.c` zawierała warunek `|| (ctl & 16)`, który błędnie odrzucał ten stan jako awarię urządzenia (`-ENODEV`). Dodatkowo natychmiastowe odpytywanie `GSP_CPUCTL` o bit `HALTED` po zapisie `STARTCPU` mogło odczytać stary bit `0x10` zanim mikrokod `FWSEC-FRTS` zdążył wystartować.
- **Rozwiązanie:**
  1. Usunięto błędny warunek `|| (ctl & 16)` po resecie silnika w `gu_falcon_loader()` oraz dodano oczekiwanie na wyzerowanie bitów scrubbing pamięci (`DMACTL & 0x6 == 0`) i konfigurację `FBIF_CTL` / `TRANSCFG(0)`.
  2. Po wystartowaniu `FWSEC-FRTS` najpierw oczekujemy, aż rejestr `WPR2_ADDR_HI` (`0x1fa828`) stanie się niezerowy (co dowodzi, że `FWSEC` faktycznie ruszył i skonfigurował region WPR2), a dopiero potem czekamy na ponowne ustawienie bitu `HALTED` (`0x10`) w `GSP_CPUCTL`.
  3. Dla `FWSEC` na Turingu wymuszono 32-bitową maskę DMA (`DMA_BIT_MASK(32)`), a dla `Booter` 47-bitową (`DMA_BIT_MASK(47)`).

### Problem 4: Błąd `-5` (`-EIO`) w `unlock_transaction_boot` (`mbox=0x0:0x31`)
- **Objaw w `dmesg`:**
  ```text
  gpu_unlock: failure unlock_transaction_boot:1157 error=-5 ... target=0x409650 value=0xffffffff ... mbox=0x0:0x31
  ```
- **Przyczyna:** Łańcuch ROP V67 na procesorze `SEC2` pomyślnie wykonał oba uprzywilejowane zapisy (`0x409650 = 0xffffffff` oraz `0x1fa828 = 0x0`), ale na końcu epilog funkcji `0x7ea4` w mikrokodzie Booter odczytał zmienną ze stosu pod adresem `main_sp + 0x08` (`0xf9f8`) i zapisał ją do rejestru `SEC2_MAILBOX0`. Ponieważ w `tu102_words` brakowało jawnego wyzerowania `{ 0xf9f8, 0x00000000 }`, w `MAILBOX0` pozostał kod błędu weryfikacji podpisu `0x31` (`49`), co sterownik uznał za błąd wykonania.
- **Rozwiązanie:** Zsynchronizowano tablicę `tu102_words` w `recipes.c` słowo w słowo z referencyjnym generatorem `generate_v67.py` z `nlzy/turing-unlock` (suma kontrolna SHA-256 bufora `0xfa00`: `b0ec90294d1de3c67e024631c0a11000f281470d76eed806fc24c66b7fd1ef41`), w tym:
  - Zapis #1: `0x00409650 = 0xffffffff` (otwarcie `TU_FEAT_PLM`),
  - Zapis #2: `0x001fa828 = 0x00000000` (wyłączenie `WPR2_ADDR_HI` z poziomu L3),
  - Zwolnienie sprzętowego muteksu ACR nr 3 (`SEMA = 0`, `RELEASE = 0x08`),
  - Wyzerowanie `{ 0xf9f8, 0x00000000 }`, dzięki czemu `SEC2_MAILBOX0` po czystym zatrzymaniu (`HALT`) wynosi `0x0`.

### Problem 5: Reset PCIe FLR na architekturze Turing kasuje odblokowanie!
- **Przyczyna:** Na kartach Ampere (RTX 30) po zapisaniu rejestrów `SS0`/`SS1` wykonuje się reset PCIe Function Level Reset (`pcie_flr()`), aby wyczyścić region `WPR2` przed startem sterownika `nvidia.ko` (na Ampere rejestry `SS0`/`SS1` przeżywają FLR). Na architekturze **Turing** wywołanie `pcie_flr()` uruchamia ponownie inicjalizację vBIOS (`devinit`), która **natychmiast przywraca blokadę** (`SS0 = 0x4010`, `SS1 = 0x4`, `PLM = 0xffffff8f`)!
- **Rozwiązanie:**
  - Na architekturze Turing **nie wykonujemy `pcie_flr()`** po udanym odblokowaniu.
  - Zamiast tego sam ładunek ROP V67 na `SEC2` zeruje rejestr `WPR2_ADDR_HI` (`0x1fa828 = 0x0`), zwalnia muteks ACR i czyści pamięć DMEM.
  - Sterownik `gpu_unlock` z poziomu hosta zapisuje `SS0 = 0x88888888` oraz `SS1 = 0x8`, a następnie zamyka z powrotem maskę uprawnień `TU_FEAT_PLM` (`0x409650 = 0xffffff8f`, odczyt `0xffffffef`).
  - Dzięki temu nadchodzący sterownik `nvidia.ko` widzi kartę z wyłączonym `WPR2` (jak po zimnym starcie) i bez problemu przeprowadza własny start GSP-RM, zachowując odblokowane rejestry `SS0`/`SS1`!

### Problem 6: Odzyskiwanie sprawności po błędzie (Unwind Path)
- W funkcji `unlock_transaction_unwind()` dodano awaryjne wywołanie `pcie_flr(work->pdev)`, jeśli transakcja przerwie się w połowie (gdy `WPR2` zdążył zostać włączony). Zapobiega to sytuacji, w której po nieudanej próbie odblokowania sterownik `nvidia.ko` odmawia startu.

---

## 4. Prawidłowy Log Startowy z `dmesg` (Wzorzec Sukcesu)

Po uruchomieniu systemu polecenie `sudo dmesg | grep gpu_unlock` zwraca poniższą sekwencję potwierdzającą pełne odblokowanie:

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

---

## 5. Przydatne Komendy i Konfiguracja na Przyszłość

### 1. Szybkie sprawdzenie statusu i benchmark Tensor Core
```bash
# Sprawdzenie logu odblokowania z jądra:
sudo dmesg | grep gpu_unlock

# Uruchomienie testu TFLOPS rdzeni Tensor Core:
python3 scripts/bench_tensor.py

# Lub za pomocą skryptu zbiorczego:
./scripts/check_unlock.sh
```

### 2. Ochrona przed resetem odblokowania przez uśpienie karty (D3cold) w laptopie
W laptopach z grafiką hybrydową (NVIDIA Optimus) sterownik może całkowicie odciąć zasilanie od karty RTX 2060, gdy nie jest używana (stan `D3cold`). Po ponownym włączeniu zasilania karta startuje z ustawieniami fabrycznymi (~50% Tensor Core) aż do kolejnego restartu systemu.  
Aby temu zapobiec, dodaj poniższą opcję modułu `nvidia` i zaktualizuj initramfs:

```bash
echo "options nvidia NVreg_DynamicPowerManagement=0x00" | sudo tee /etc/modprobe.d/99-nvidia-no-d3cold.conf
sudo update-initramfs -u
```

### 3. Jak tymczasowo wyłączyć `gpu_unlock` (np. do porównania wyników przed/po)?
- Podczas uruchamiania laptopa w menu **GRUB** naciśnij klawisz `e`, usuń parametr `gpu_unlock.targets=all` z linii `linux ...` i naciśnij `F10`. System uruchomi się bez odblokowania (wtedy `bench_tensor.py` pokaże ~28.5 TFLOPS zamiast ~57.2 TFLOPS).

### 4. Reinstalacja po przebudowie źródeł
```bash
cd gpu-unlock
make install
```
*(Moduł jest zarejestrowany w **DKMS**, więc przy zwykłych aktualizacjach jądra przebuduje się automatycznie).*

> [!WARNING]
> **Zastrzeżenie (używasz na własne ryzyko):** Wsparcie dla kart Turing (seria RTX 20) ma charakter eksperymentalny. Oprogramowanie jest udostępniane w stanie „takim, jakie jest” (*AS IS*), bez jakiejkolwiek gwarancji zgodnie z licencją **GPL-2.0**. Korzystasz z niego wyłącznie na własną odpowiedzialność.

---

## 6. Wsparcie

Jeśli odblokowanie kart Turing / RTX 2060 oraz ta dokumentacja okazały się pomocne, możesz wesprzeć rozwój projektu:

- **Buy Me a Coffee:** [https://buymeacoffee.com/kaspral](https://buymeacoffee.com/kaspral)
- **BTC:** `bc1pdc8clusv5hpgz7laywwzhjjcklk6x2cgj2xu5uc9xes9fukxaegsn7cs20`
- **ETH:** `0x006dAcBA1344144AB395D02637045Dd19aB54C2E`
- **SOL:** `96Lc4X6hieoCkUFHtDW4wvbWXgZvebqTktjRNiCNe1ex`

