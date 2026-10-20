# IF3170_G41_OpenBobi

Tugas Besar 1 IF3170 Inteligensi Artifisial 2026/2027, pengisian peti kemas truk (Mejiro Express) dengan algoritma local search.

Laporan lengkap (PDF) ada di folder `docs/`. Tautan versi daring: https://docs.google.com/document/d/1wB1Sk5W6sjFdFmU7AT5GnXjemQ8IlfAt0DRazotKbvo/edit?usp=sharing

## Deskripsi Singkat

Program mencari konfigurasi penempatan paket di dalam peti kemas truk yang memaksimalkan total value, dengan batasan: paket tidak melayang atau menembus dinding, barang pecah-belah tidak boleh berada di bawah paket lain, dan total berat tidak melebihi kapasitas.

Algoritma yang diimplementasikan:
- Hill-climbing: Steepest Ascent, Sideways Move, Stochastic, Random Restart
- Simulated Annealing
- Genetic Algorithm

Objective function tersedia: `default` (total value), `density`, dan `eta_urgency`.

Fungsi `state_resolution(state, objective_type)` di `src/algorithms/state_space.py` memeriksa seluruh constraint sebuah state dan mengembalikan pasangan (valid, skor). Variabel `state_score_trig` di modul yang sama menyimpan skor hasil pemanggilan terakhir fungsi tersebut.

## Struktur Repository

```
src/        kode program (algorithms, models, experiments, utils, visualization)
tests/      pengujian otomatis
data/       instance masalah (JSON)
docs/       laporan (PDF)
output/     hasil run algoritma (JSON state, ringkasan eksperimen) dan dashboard.html
```

## Setup

Membutuhkan Python 3.9 atau lebih baru.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Di Linux atau macOS, aktivasi dengan `source venv/bin/activate`.

## Menjalankan Program

Satu kali run untuk algoritma tertentu dengan opsi `--algorithm` (`steepest`, `sideways`, `stochastic`, `restart`, `sa`, `ga`; bawaan `ga`):

```
python -m src.main -a steepest --seed 1
python -m src.main -a sideways --max-sideways 25 --seed 1
python -m src.main -a stochastic --seed 1
python -m src.main -a restart --max-restarts 5 --seed 1
python -m src.main -a sa --sa-iterations 3000 --seed 1
python -m src.main -a ga -p 50 -n 100 --seed 1
```

Hasil satu run disimpan ke `output/latest_solution.json`.

Seluruh eksperimen Hill-climbing dan Simulated Annealing (masing-masing 3 kali):

```
python -m src.main --run-local-search
```

Seluruh eksperimen Genetic Algorithm (variasi iterasi dan variasi populasi, masing-masing 3 kali):

```
python -m src.main --run-experiments
```

Semua eksperimen sekaligus:

```
python -m src.main --run-all
```

Objective function alternatif, simpan ke folder terpisah:

```
python -m src.main --run-all --objective density -o output/density
```

## Instance Tambahan

Instance acak yang lebih sulit (40 paket) dibuat dan dijalankan ke folder terpisah:

```
python -m src.main --generate 40 --generate-output data/hard.json --run-all -o output/hard
python -m src.visualization -i output/hard --open
```

Batas atas value (relaksasi berat dan volume, hanya berlaku untuk objective `default`):

```
python -m src.experiments.upper_bound data/hard.json
```

## Visualisasi

```
python -m src.visualization -i output --open
```

Perintah ini membuat `output/dashboard.html` yang menampilkan state awal dan akhir (3D), plot objective terhadap iterasi (dengan penanda restart untuk Random Restart), plot e^(ΔE/T) untuk Simulated Annealing, plot maksimum dan rata-rata populasi untuk Genetic Algorithm (warna `#386641`), serta tabel ringkasan eksperimen beserta gap ke batas atas value. Dashboard juga memverifikasi ulang seluruh constraint dari file solusi.

Folder `output/` berisi `experiments/` (ringkasan JSON tiap algoritma), `solutions/` (state awal dan akhir tiap run), dan `dashboard.html`. Buka `dashboard.html` langsung di browser, tanpa server.

## Pengujian

```
python -m unittest discover -s tests
```

## Catatan Implementasi

Beberapa keputusan implementasi yang perlu diketahui untuk membaca hasil eksperimen:

- **Initial state bersifat random-greedy.** `StateSpace.generate_initial_state()` mengacak urutan paket dan orientasi, lalu menempatkan setiap paket pada titik kandidat pertama yang valid. Artinya initial state cenderung sudah cukup padat, sehingga ruang perbaikan untuk Hill-climbing relatif kecil pada instance kecil. Pada instance yang lebih padat (`data/hard.json`, 40 paket), initial state bisa lebih buruk dan perbedaan antar-algoritma lebih terlihat.

- **Metrik "stuck" pada Simulated Annealing.** Tersedia dua metrik yang berbeda. `stuck_frequency` menghitung proporsi iterasi ketika nilai terbaik (best-so-far) tidak membaik, sedangkan `trapped_frequency` menghitung proporsi iterasi ketika state saat ini benar-benar tidak berubah (tidak ada neighbor yang diterima). Spesifikasi tidak mengaitkan `stuck` dengan local optima. Kelompok 41 mendefinisikan `stuck` sebagai proporsi iterasi di mana state tidak berubah sama sekali (tidak ada neighbor yang diterima), yang dilaporkan sebagai trapped_frequency. Metrik ini dipilih karena lebih mencerminkan kondisi terperangkap tanpa pergerakan, bukan sekadar best-so-far yang stagnan. stuck_frequency (best tidak membaik) tetap dilaporkan sebagai metrik pendukung.

- **Interpretasi "eET" pada spesifikasi Simulated Annealing.** Kelompok 41 menginterpretasikan `eET` sebagai `e^(ΔE/T)`, yaitu peluang penerimaan solusi yang lebih buruk pada suhu T. Nilai ini dicatat per iterasi pada `history_acceptance_probability` dan divisualisasikan pada dashboard. Untuk iterasi dengan ΔE ≥ 0, peluang dipotong menjadi 1.0 karena solusi tersebut selalu diterima.

- **Heuristik decode Genetic Algorithm.** Chromosome merepresentasikan urutan paket, indeks rotasi, dan flag inklusi. Fungsi `decode()` menempatkan paket mengikuti urutan tersebut secara greedy: untuk setiap paket, dicoba seluruh rotasi pada titik kandidat yang tersedia dan dipilih titik valid pertama dengan urutan (z, y, x) terkecil. Heuristik ini menjamin seluruh hasil decode valid terhadap seluruh constraint, tetapi tidak menjamin optimal secara geometris.

- **Kinerja enumerasi neighbor Hill-climbing.** Varian Steepest Ascent dan Sideways Move mengevaluasi seluruh neighbor lengkap per iterasi (enumerasi swap, move, rotate, exchange). Untuk instance 20 paket seperti `mejiro_express.json`, ini menghasilkan ribuan neighbor per iterasi sehingga lebih lambat dari Stochastic HC dan SA yang memakai `random_neighbor`. Perbedaan durasi ini tercatat otomatis pada output eksperimen dan divisualisasikan di dashboard.

## Format Input dan Syarat Validitas

Program menerima satu file JSON per instance dengan struktur berikut.

| Field | Tipe | Aturan validitas |
|---|---|---|
| `truck.label` | string | bebas, tidak kosong |
| `truck.dimensions` | `[int, int, int]` | masing-masing ≥ 1, bilangan bulat (w, l, h) |
| `truck.max_capacity` | int | ≥ 1, dalam satuan berat |
| `packages` | list | tidak kosong |
| `packages[].id` | string | unik dalam satu instance |
| `packages[].dimensions` | `[int, int, int]` | masing-masing ≥ 1 |
| `packages[].value` | int | ≥ 0 |
| `packages[].weight` | int | ≥ 0 |
| `packages[].is_fragile` | bool | wajib `true`/`false` (bukan string) |
| `packages[].eta` | int | ≥ 0, opsional (default 0 jika tidak ada) |

Pelanggaran aturan di atas menghasilkan `ValueError` saat `load_instance` dipanggil. Koordinat posisi paket pada state direpresentasikan sebagai `[x, y, z]` bilangan bulat, dan dimensi paket saat ditempatkan harus merupakan permutasi dari `dimensions` aslinya (divalidasi oleh `validate_state`).

## Pembagian Tugas

| Anggota | NIM | Tanggung jawab |
|---|---|---|
| Aufa Rienaldifaza Ahmad | 13524027 | State, constraint, objective, neighbor, Hill-climbing, Simulated Annealing beserta eksperimennya |
| Yavie Azka Putra Araly | 13524077 | Genetic Algorithm beserta seluruh eksperimen GA |
| Kalyca Nathania Benedicta Manullang | 13524071 | Visualisasi dan tampilan hasil eksperimen |