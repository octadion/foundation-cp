# Pra-registrasi: metrik mana yang boleh dilaporkan di dataset mana

**Ditulis 2026-08-13, SEBELUM satu pun run Phase 2 dijalankan.** Alasan dokumen ini ada:
metrik utama kita adalah **worst-class coverage**, dan di dataset ekor-panjang statistik itu
**tidak terukur** — bukan karena metodenya, tetapi karena tidak ada cukup sampel evaluasi per
kelas. Kalau pilihan metrik dibuat setelah melihat hasil, itu p-hacking dengan nama lain.
Jadi aturannya ditetapkan di sini, dalam bentuk yang dieksekusi kode, bukan diingat.

## Fakta pengukuran, dari survei dump (bukan ingatan)

| Dataset | sumber | K | baris test | median/kelas | granularitas coverage per kelas |
|---|---|---|---|---|---|
| ImageNet | CCC | 1000 | 1.153.051 | **1168** | ~0,001 |
| CIFAR-100 | milik sendiri | 100 | 10.000 | 100 | 0,01 |
| Pl@ntNet | LTC | 1081 | 31.112 | **3** | **0,33** |
| iNat-2018 | LTC | 8142 | 46.227 | **2** | **0,50** |

Baris terakhir adalah masalahnya. Dengan 2 sampel evaluasi, coverage sebuah kelas hanya bisa
bernilai 0, 0,5, atau 1. Atas 8142 kelas, `min` atas nilai-nilai itu **hampir pasti 0** untuk
metode apa pun — termasuk oracle. Jadi `worst` dan `max_gap` di sana mengukur derau sampling,
bukan metode, dan **selisih antar metode di kolom itu tidak punya arti.**

## Aturan, dieksekusi oleh `pcc/experiments/phase2_pcc.py`

**Ambang keterukuran per-kelas: median ≥ 30 sampel evaluasi per kelas.** Di bawah itu, sebuah
estimasi coverage per kelas tidak dapat membedakan 0,90 dari 0,80 pada tingkat kelas, sehingga
statistik per-kelas (`worst`, `max_gap`, `p05`, `p10`) **ditahan** — dihitung dan dilaporkan
di bawah kunci `withheld_unmeasurable`, tetapi **tidak boleh masuk tabel paper dan tidak boleh
menentukan verdict.**

**Statistik primer, per rezim:**

| Rezim | median eval/kelas | statistik primer | unit |
|---|---|---|---|
| **A** | ≥ 30 | `worst` — coverage kelas terburuk | kelas |
| **B** | < 30 | `bin_worst` — coverage bin prevalensi terburuk | bin |

Rezim B mempertahankan sifat *worst-case*-nya dan hanya **memperkasar unitnya** sampai
terukur. Kelas diurutkan menurut prevalensi lalu diakumulasi serakah menjadi bin yang
masing-masing memuat **≥ 200 baris evaluasi**; coverage dihitung terpooling di dalam bin.
Bin terakhir digabung ke bin sebelumnya bila kurang dari 200, sehingga tidak ada bin
setengah-terisi yang mendominasi minimum.

Yang **selalu** dilaporkan di kedua rezim, karena selalu terukur:

- **macro coverage** — rata-rata coverage per kelas. Tak bias meski n kecil per kelas, hanya
  berderau; galatnya dilaporkan.
- **avg set size** — tidak butuh label per kelas sama sekali, jadi presisinya penuh.
- **marginal coverage.**

## Konsekuensi yang diterima di muka

1. **ImageNet dan Pl@ntNet tidak akan punya kolom utama yang sama.** Itu bukan kelemahan yang
   disembunyikan; itu batas data mereka. Tabel paper memuat dua blok dengan header berbeda,
   bukan satu tabel yang menyamarkan perbedaan unit.
2. **`bin_worst` lebih lemah dari `worst`.** Ia bisa menyembunyikan satu kelas yang runtuh di
   dalam bin yang sehat. Itu dinyatakan sebagai batas, dan alasan mengapa klaim ekor-panjang
   disandarkan pada ImageNet untuk daya uji dan pada Pl@ntNet/iNat untuk relevansi — masing-
   masing pada apa yang bisa diukur di sana.
3. **Kalau nanti muncul dorongan melaporkan `worst` di Pl@ntNet karena angkanya bagus, itu
   dilarang oleh dokumen ini.** Begitu pula sebaliknya: kalau `bin_worst` di ImageNet lebih
   bagus dari `worst`, `worst` yang tetap primer di sana.

## Yang membuat aturan ini bisa gagal, dan bagaimana kalau iya

Ambang 30 dan 200 adalah pilihan, bukan turunan. Keduanya ditetapkan di sini agar tidak bisa
digeser belakangan. Kalau setelah run ternyata sebuah dataset jatuh persis di sekitar ambang
(median 25–35), **kedua rezim dilaporkan berdampingan** untuk dataset itu, bukan dipilih
salah satu — dan fakta bahwa ia di perbatasan dinyatakan.
