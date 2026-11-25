# lsdp — DP multiobjektif eksak untuk model lot sizing Kania dkk. (CIE 2022)

## Build (CMake + Ninja)
```bash
cd ~/riset/lsdp
cmake --preset release          # konfigurasi: Ninja, -O3 -march=native, OpenMP
cmake --build --preset release  # hasil: build/release/lsdp
./build/release/lsdp verify --seeds 50
```
Tidak ada dependensi eksternal selain OpenMP (bawaan GCC), jadi Conan tidak diperlukan.
Untuk Python (pembanding): `pip install numpy scipy pymoo` di dalam virtualenv.

## Isi folder
| Berkas | Fungsi |
|---|---|
| `src/lsdp.cpp` | DP eksak (C++20/OpenMP), mode `verify`, `verify-cont`, `front` |
| `CMakeLists.txt`, `CMakePresets.json` | Build Release/Debug dengan Ninja |
| `run_suite.sh` | Protokol eksperimen E1–E5 (memakai `build/release/lsdp`) |
| `summarize.py` | Rangkuman tabel dari semua log suite |
| `lsmodel.py` | Evaluator Python (identik dengan lsdp) + operator repair |
| `nsga3_baseline.py` | Pembanding NSGA-III (pymoo), opsi `--repair 1` |
| `milp_baseline.py` | Pembanding eksak umum: ε-constraint MILP (HiGHS via SciPy) |
| `compare.py` | Indikator kualitas (toleransi sesuai presisi cetak) |
| `results_backup/` | Ringkasan hasil lama (laptop sebelumnya, 18 Nov 2026) |

## Metode (ringkas)
1. Reformulasi kumulatif X(τ); HC, ITO, POC aditif atas pasangan status berurutan.
2. DP label-setting Pareto 3-kriteria, label per status disimpan sebagai tangga 2-D per k.
3. Lemma batas atas X(T′) < β_max + max(moq, r); Lemma D (pesan hanya jika perlu).
4. Memoisasi eksak per vektor kebutuhan yang dibulatkan ke kisi.
5. SS/SOT kontinu: enumerasi sel arrangement; front kontinu umumnya tidak tertutup
   (kolom `attained` = 0 untuk titik yang hanya bisa didekati).
6. Proposisi invariansi biaya pesan (moq = r; c·T′ < p·r).

## Mode
```bash
B=./build/release/lsdp
$B verify --seeds 50
$B verify-cont --sku 1 --seed 1 --samples 1000000
$B front --sku 1 --T 48 --seed 42 --outer continuous --threads 12 --out f.csv
#  opsi: --mu --cv --c --hfrac --MS --r --L ; --SS x --SOT y (satu titik) ; --noito 1 ; --ref 1
```

## Protokol eksperimen
```bash
bash run_suite.sh E1    # kebenaran (~5–10 menit)
bash run_suite.sh E2    # studi utama T=48 (~5 menit)
bash run_suite.sh E4    # sensitivitas (~10 menit)
bash run_suite.sh E3    # skala T (~1 jam)
bash run_suite.sh E5b   # NSGA-III tanpa/dengan repair (~2 jam; perlu pymoo)
bash run_suite.sh E5a   # MILP vs DP (~1 jam; perlu scipy)
bash run_suite.sh E3x   # opsional, berat
```

## Parameter instans (direkonstruksi dari desdeo-dash/data)
| | SKU 1 | SKU 2 |
|---|---|---|
| p | 134,00 | 483,85 |
| c | 200 | 200 |
| h per minggu | 0,1·p/52 | 0,1·p/52 |
| L (minggu) | 5 | 4 |
| moq = r | 70 | 3 |
| μ, σ per minggu | 163, 41 | 12,2, 3,6 |
| MS (minggu) | 1,0 | 1,4 |

Permintaan N(μ, σ) dibulatkan dan dipotong di 0; I0 = kumulatif permintaan periode 1..L + μ.
Semua eksperimen deterministik per seed: hasil bisa dibangkitkan ulang persis sama.
