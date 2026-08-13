# Baseline pada skor yang SAMA — hasil pertama, dan target PCC yang dipatok di muka

**Run:** notebook 05, 2026-08-12, `git_commit a5f480a`.
**Dump:** CCC ImageNet, 1.153.051 baris × 1000 kelas, subsample terstratifikasi **21,7%**.
**Split:** DESC 99.985 / CAL 74.960 / EVAL 75.068. `n_cal` per kelas: median 76, min 43.
**α = 0,10.** Semua angka dari **array skor yang identik** dengan yang dipakai seluruh gerbang.

Implementasi baseline adalah **kode penulisnya sendiri** (`utils/conformal_utils.py`, repo
`class-conditional-conformal`), dipanggil apa adanya — bukan reimplementasi kita. Signature tiap
fungsi dicetak sebelum dipanggil dan cocok kata-per-kata dengan `release_audit.md`.

## Grid Tier-1 (fungsi skor) × Tier-2 (metode kelas)

| skor | metode | mean class-cov gap | **max gap** | very undercov. | marginal cov | avg set size |
|---|---|---|---|---|---|---|
| THR | standard | 0,0580 | **0,420** | 0,109 | 0,8980 | **1,911** |
| THR | classwise | **0,0367** | 0,173 | 0,012 | 0,9087 | 4,101 |
| THR | **clustered** | 0,0380 | 0,233 | 0,030 | 0,9012 | **2,568** |
| APS | standard | 0,0362 | 0,225 | 0,032 | 0,8996 | 25,57 |
| APS | classwise | 0,0374 | 0,156 | 0,020 | 0,9076 | 34,96 |
| APS | clustered | **0,0348** | 0,147 | 0,017 | 0,9005 | 29,10 |
| RAPS | standard | 0,0403 | 0,255 | 0,042 | 0,8986 | 5,108 |
| RAPS | classwise | 0,0380 | 0,193 | 0,018 | 0,9069 | 6,846 |
| RAPS | clustered | 0,0363 | **0,167** | 0,019 | 0,9002 | 5,391 |

RAPS memakai `lmbda=0,01`, `kreg=5` — default `get_RAPS_scores` di repo mereka, bukan pilihan kita.

## Tiga hal yang dibaca dari tabel ini

**1. Clustered CP mereproduksi klaim sentralnya, dan itu titik operasi yang harus dilampaui.**
Pada THR: gap 0,0380 dengan set **2,568** — nyaris menyamai gap classwise (0,0367) di **63% ukuran
set**-nya (4,101). Jadi Clustered CP bukan strawman; ia memang membeli ekuitas dengan murah. Setiap
klaim PCC harus diukur terhadap baris ini, bukan terhadap `standard`.

**2. Pilihan fungsi skor mendominasi sumbu ukuran set, jadi perbandingan wajib menahannya tetap.**
Set APS **10–13× lebih besar** dari THR untuk gap yang praktis sama. Membandingkan metode lintas
fungsi skor akan mencampur dua efek yang tidak sebanding. PCC memakai THR/LAC, jadi baris
pembandingnya adalah tiga baris THR — dan hanya itu.

**3. Kolom yang sebanding dengan §6.4 kita adalah `max gap`, bukan `mean gap`.**
Objektif kita worst-class; `mean class-cov gap` bukan mata uangnya. Pada THR: standard 0,420,
clustered 0,233, classwise 0,173. Itu angka yang harus dikalahkan, dan `very undercovered`
(fraksi kelas yang jauh di bawah target) adalah pembanding kedua yang sah.

## Target PCC — DIPATOK SEBELUM `g_θ` DITULIS

Ditulis sekarang justru supaya tidak bisa disesuaikan belakangan.

**Tabel 1 — kelas ber-data (§7 "seen"), THR, α=0,10, array yang sama.** PCC **minimal seri** dengan
Clustered CP:

- `max gap` ≤ **0,233** pada `avg set size` ≤ **2,568**; dan
- `mean class-cov gap` ≤ **0,0380** pada ukuran set yang sama.

Kalau PCC lebih baik pada satu sumbu dengan mengorbankan sumbu lain, yang dilaporkan adalah
**frontier**-nya, bukan satu titik pilihan.

**Tabel 2 — kelas held-out (`n_y = 0`), tempat klaimnya hidup.** Per
[`fallback_policy.md`](fallback_policy.md) yang beku sejak 2026-07-24, setiap baseline kelas jatuh ke
ambang marginal global di `n_y = 0`, yaitu baris **THR|standard: max gap 0,420, very undercovered
0,109, size 1,911**. PCC harus mengalahkan itu **pada ukuran set tercocokkan**.

> **Peringatan strawman, dinyatakan di muka.** Menang di Tabel 2 saja **tidak cukup**, karena di
> sana setiap pesaing memang tak terdefinisi. Itu sebabnya §7 mewajibkan dua tabel dan Tabel 1 harus
> minimal seri. Kalau PCC menang di Tabel 2 tetapi kalah di Tabel 1, kesimpulannya adalah metode ini
> menukar kelas ber-data demi kelas tanpa data — dan itu yang harus ditulis.

## Status §7: BELUM terpenuhi

§7 melarang baseline masuk tabel paper sebelum **direproduksi lawan angka terbit penulisnya** pada
≥1 setting. Angka di atas belum bisa dibandingkan dengan angka terbit karena:

1. **Subsample 21,7%**, bukan dump penuh → jalankan ulang dengan `MAX_ROWS = None`;
2. **Protokol split kita** (DESC/CAL/EVAL 40/30/30) berbeda dari `n_totalcal` per kelas yang dipakai
   papernya — mereka memakai `split_X_and_y(..., n_k, split='balanced')`;
3. **α dan `n_cal`** kita dipilih untuk gerbang, bukan untuk mencocokkan tabel mereka.

Sampai ketiganya diselaraskan, angka di atas sahih untuk **perbandingan internal** (metode kita vs
baseline, array identik) dan **tidak** sahih sebagai klaim reproduksi. Keduanya berbeda dan tidak
boleh dicampur dalam satu kalimat di paper.

## Sisa baseline yang belum tersentuh

| Metode | Kode | Status |
|---|---|---|
| standard / classwise / Clustered CP | repo CCC | **dipanggil ✅** |
| APS / RAPS (fungsi skor) | repo CCC | **dipanggil ✅** |
| PAS + Interp-Q | repo LTC | API belum didaftar |
| Fuzzy Classwise CP | repo LTC | API belum didaftar — §7: **jalankan awal** |
| Macro-coverage / label-weighted | rilis | belum di-fetch |
| Class similarity (Fargion) | github | belum di-fetch — §7: **paling berbahaya** |
| RC3P | github | belum |
| TACP / sTACP | — | reimplementasi |
| CFCP | — | reimplementasi |

Dua yang ditandai §7 sebagai paling berbahaya — Fuzzy Classwise dan Class Similarity — keduanya
memakai kemiripan kelas yang ditentukan tangan, jadi keduanya adalah pesaing terdekat konseptual
PCC. Keduanya harus dijalankan **sebelum** metode kita dianggap punya keunggulan.
