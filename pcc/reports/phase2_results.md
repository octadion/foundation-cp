# Phase 2 — hasil run pertama (2026-08-13)

**Run:** notebook 06, 30 konfigurasi, 30/30 selesai, 2202 s, `git_commit f4c0503`.
**Artefak:** `06_phase2_summary.json` + satu laporan per konfigurasi.
**Baseline Tabel 1 DILEWATI** (repo CCC tidak ada di `/content/ccc`), jadi target patokan
lawan Clustered CP belum diuji.

## Yang LOLOS: ImageNet + φ kepala

| besaran | nilai | CI antar-seed (n=5) |
|---|---|---|
| λ | 0,080 | [0,041, 0,119] |
| **Tabel 2 — kelas `n_y = 0`, worst** | **+0,0249** | **[+0,0054, +0,0444]** |
| **Tabel 1 — kelas terlihat, worst** | **+0,0495** | **[+0,0059, +0,0930]** |
| ukuran set tercocokkan | 5/5 seed | — |

Inilah klaim inti proyek, terukur: **δ̂ dari geometri kepala yang eksogen memperbaiki
coverage kelas terburuk pada kelas yang tidak punya satu pun sampel kalibrasi**, dan tidak
dibayar dengan kelas yang punya data. Kriteria pra-registrasi terpenuhi pada kedua tabel.

### Harga yang harus ikut dilaporkan

**Macro coverage TURUN** di kedua tabel: −0,0041 [−0,0066, −0,0016] (Tabel 1) dan
−0,0036 [−0,0056, −0,0016] (Tabel 2). CI-nya **tidak** memuat nol.

Pada ukuran set tercocokkan, memperbaiki kelas terburuk berarti **memindahkan** anggaran
dari kelas lain — jadi ini redistribusi, bukan perbaikan gratis. Itu memang perilaku yang
diharapkan dari metode ekuitas, tetapi **harus ditulis sebagai trade-off**, bukan
disembunyikan di balik angka worst-class. Besarannya: worst naik ~0,025–0,050 sementara
macro turun ~0,004.

## Yang GAGAL — dan ketiganya karena λ = 0,000

| konfigurasi | λ | Tabel 2 | verdict |
|---|---|---|---|
| ImageNet, φ ruang-output | **0,000** | +0,0000 | GAGAL |
| Pl@ntNet, φ ruang-output | **0,000** | +0,0000 | GAGAL |
| iNat-2018, φ ruang-output | **0,000** | +0,0000 | GAGAL |

λ = 0 berarti `select_shrinkage` **tidak menemukan satu pun λ > 0 yang memperbaiki ekuitas
worst-class pada kelas TRAIN.** Jadi ini bukan koreksi yang salah arah — ini koreksi yang
ditolak seluruhnya oleh pemilihan λ. Metodenya menahan diri, dan itu perilaku yang benar.

### Temuan terkuat run ini: peringkat R² TERBALIK di tingkat metode

| keluarga φ | gate B R² (nb 05) | λ terpilih | Tabel 2 |
|---|---|---|---|
| ruang-output | **+0,4975** | 0,000 | +0,0000 |
| kepala `w_y` | +0,3880 | 0,080 | **+0,0249** |

φ dengan R² **lebih tinggi** ternyata **tidak bernilai sama sekali**, dan φ dengan R² lebih
rendah yang bekerja. Ini pelajaran Amandemen 8 muncul untuk **ketiga** kalinya, sekarang di
tingkat metode: objektif worst-class dikuasai oleh **galat ekor**, bukan galat kuadrat
rata-rata, dan R² hanya mengendalikan yang kedua.

Mekanisme yang konsisten dengan datanya: deskriptor kepala **eksak secara konstruksi** —
diturunkan dari matriks bobot, nol derau sampling. Deskriptor ruang-output **diestimasi dari
sampel** split DESC, jadi δ̂-nya memikul derau sampling DESC **di atas** galat model. Atas
700 kelas train, yang menentukan minimum adalah galat terbesar, dan itulah yang dibayar
keluarga ruang-output.

Konsekuensi untuk paper: **jangan pakai R² untuk memilih keluarga deskriptor.** Yang
memilih adalah §6.4/Tabel 2, dan urutannya bisa berkebalikan.

### Temuan kedua: keunggulan classwise CP sebagian dibeli dengan ukuran set

Seed 4 pada ImageNet/ruang-output memakai δ_y **terukur** untuk kelas dengan ≥50 baris CAL
(`n_star = 50`, λ = 0) dan Tabel 1 worst-nya **−0,5132**. Itu classwise CP, pada ukuran set
tercocokkan.

Bandingkan `baseline_reproduction.md`: classwise CP di sana `max_gap 0,173` lawan standard
`0,420` — tampak jauh lebih baik. Tetapi ukuran setnya **4,10 lawan 1,91**. Jadi sebagian
besar keunggulan itu dibeli dengan set yang dua kali lebih besar; **pada ukuran tercocokkan
ia bisa berbalik negatif.** Keduanya tidak bertentangan — yang kedua yang menjawab
pertanyaan yang benar, dan itu justru alasan Amandemen 8 mewajibkan pencocokan sumber daya.

## Ketegangan ekor-panjang: sekarang TERUKUR, bukan diperdebatkan

Klaim ekstrapolasi **tidak** tertegakkan di Pl@ntNet maupun iNat: Tabel 2 tepat +0,0000 di
keduanya, semua seed, semua α.

Tetapi diagnosisnya **bukan** "dataset ekor-panjang tidak cocok". Yang dijalankan di sana
**hanya keluarga ruang-output** — dan keluarga itu juga gagal di ImageNet, di mana daya
ujinya paling besar. Keluarga yang berhasil (kepala) **tidak pernah diuji** di sana, karena
kepala torchvision punya 1000 kelas sementara Pl@ntNet 1081 dan iNat 8142, jadi grid
melewatinya secara otomatis.

**Jadi ini kegagalan KELUARGA φ, bukan kegagalan dataset** — dan itu bisa diuji, karena LTC
**merilis checkpoint ResNet-50** untuk kedua dataset (`release_audit.md`). Mengambil
`fc.weight` dari checkpoint itu tidak butuh forward pass, tidak butuh GPU, dan langsung
memberi keluarga kepala pada dua dataset ekor-panjang. Itu langkah berikutnya yang bernilai
paling tinggi.

Satu perbedaan yang harus dinyatakan bila itu dijalankan: di ImageNet, kepala berasal dari
model yang **berbeda** dari penghasil skor (eksogen sepenuhnya). Di Pl@ntNet/iNat kepalanya
dari model yang **sama**, jadi kurang eksogen — tetapi ia tetap parameter, bukan estimasi
sampel, dan tetap tersedia untuk kelas tanpa satu pun label. Keduanya layak dilaporkan
sebagai dua tingkat klaim yang berbeda, bukan digabung.

## Anomali yang belum terselesaikan

1. **`nb05 §6.4` dan `nb06` tidak sepakat untuk ruang-output.** nb05 melaporkan +0,1173,
   nb06 melaporkan λ=0 dan +0,0000. Perbedaan mekanisnya sudah teridentifikasi: nb05
   memilih λ pada **slice EVAL** yang dibatasi kelas train (`setsize_translation_shrunk`),
   nb06 memilih pada **slice CAL** (`fit_pcc`). Keduanya sah — tidak ada yang menyentuh
   kelas yang dilaporkan — tetapi menghasilkan λ yang berbeda. Yang mana yang benar untuk
   deployment perlu diputuskan **dan dinyatakan**, bukan dipilih berdasarkan mana yang
   angkanya lebih bagus. Sampai itu diputuskan, angka §6.4 nb05 untuk ruang-output tidak
   boleh dilaporkan berdampingan dengan angka nb06 seolah sebanding.
2. **iNat: ukuran set tidak cocok pada 1 dari 5 seed** (`size_matched` 0,8). Angka Tabel 1
   iNat untuk seed itu tidak boleh dibaca. Perlu diperiksa apakah `shift_to_size` gagal
   konvergen pada 8142 kelas dengan median 2 baris eval.
3. **Baseline belum ikut.** Target patokan (`max_gap ≤ 0,233` pada `size ≤ 2,568` lawan
   Clustered CP) belum diuji karena repo CCC tidak ada di runtime. Tabel 1 saat ini hanya
   dibandingkan dengan ambang marginal global, yang lebih lemah dari yang dipra-registrasi.
4. **α = 0,05 pada φ kepala GAGAL** (Tabel 2 −0,0217, `n_star = 50`). Jadi hasil yang lolos
   spesifik untuk α = 0,10 sejauh ini. Itu batas cakupan, dan harus dinyatakan begitu.

## KENAPA λ = 0 DI EKOR PANJANG — terjawab (2026-08-15)

Tiga run berturut-turut melaporkan λ = 0 di Pl@ntNet dan iNat, dengan baris `head` dan
`output` **identik sampai 16 digit** — bukti bahwa keluarga deskriptor tidak pernah masuk
ke hasil sama sekali. Instrumentasi `zero_lambda_reason` dipasang, dan kurva λ dari
`phase2_ltc_plantnet_head_a0.05_nc25_ho0.3_s0` menjawabnya:

```
λ:   0,0 → 0,5     0,05 → 0,0     0,1 → 0,0     ...     1,0 → 0,0
```

**Bukan datar, bukan menurun — sebuah tebing.** Statistiknya jatuh dari 0,5 ke **tepat
0,0** begitu koreksi sekecil apa pun diterapkan, lalu diam di sana untuk setiap λ.

Nilai 0,5 dan 0,0 adalah tanda pengenal kelas dengan **dua baris kalibrasi**: coverage-nya
hanya bisa 0, 0,5, atau 1. Jadi objektif seleksi ditentukan oleh segelintir kelas yang
perturbasi ambang sekecil apa pun langsung menjatuhkannya ke nol. `n_star` mengonfirmasi
dari sisi lain: `curve_observed {5: 0,0, 10: 0,0}`, dan kandidat 20/30/50/75 **tidak bisa
dievaluasi** — kurang dari 75 kelas sanggup menyisihkan barisnya.

**Dua dugaan sebelumnya, keduanya salah, dicatat:**

1. *"δ̂ degenerate"* — diuji pada dunia Zipf sintetis di mana hanya 7% kelas train punya
   δ_obs; λ tetap 0,7, kurva tidak datar, sd(δ̂) 0,0024 vs sd(δ_obs) 0,0022. **Terbantah.**
2. *"turun ke p25 cukup"* — tidak. Tebingnya bertahan bahkan pada kuantil ekor-bawah.

Yang degenerate **bukan δ̂, melainkan OBJEKTIFNYA.** Metodenya menolak bertindak, dan
penolakan itu benar: tidak ada λ yang bisa memperbaiki statistik yang sudah nol.

### Satu bug yang tersingkap dari kurva yang sama

`select_n_star_oos` dipanggil dengan `stat=stat`, bukan `stat=sel_stat` — jadi penurunan
slice-tipis mencapai λ tetapi **tidak** mencapai `n_star`. Di dump ekor-panjang, dua
parameter yang saling bergantung dipilih oleh **dua objektif berbeda** (`p25` untuk λ,
`worst` untuk `n_star`). Terlihat langsung di laporan: `n_star` mencetak `stat: worst`
padahal λ sudah diturunkan. Diperbaiki.

### Perbaikan yang benar, dan mengapa bukan p25

`prereg_metrics_per_dataset.md` sudah menetapkan aturannya untuk **pelaporan**: di rezim B
statistik primernya `bin_worst` — kelas dikelompokkan menurut prevalensi sampai tiap bin
memuat ≥200 baris evaluasi. Kurva ini menunjukkan aturan yang sama berlaku untuk
**seleksi**, dan lebih keras dari yang diasumsikan: p25 pun tidak cukup, karena
diskretisasi kelas 2-baris menembusnya.

Jadi seleksi λ dan `n_star` di rezim B harus memakai **statistik tingkat-bin**, bukan
kuantil per-kelas apa pun. Itu perbaikan berikutnya, dan ia bukan tambalan — ia
menyelaraskan seleksi dengan aturan pelaporan yang sudah dipra-registrasi.

**Sampai itu dikerjakan, "PCC gagal di ekor panjang" TIDAK BOLEH ditulis.** Yang benar:
objektif seleksinya tidak terukur di sana, jadi metodenya tidak pernah bertindak — dan
klaim ekor-panjang masih **belum diuji**, bukan terbantah.

## HASIL FINAL EKOR PANJANG (2026-08-16) — terjawab, dan jawabannya negatif

Seleksi tingkat-bin dipasang, dan **ia bekerja secara mekanis**. Untuk pertama kalinya
dalam empat run, `head` dan `output` menghasilkan angka **berbeda** di Pl@ntNet — artinya
keluarga deskriptor akhirnya masuk ke hasil, bukan ditolak sebelum dipakai.

| Pl@ntNet | λ (CI antar-seed) | Tabel 2 (`n_y = 0`) |
|---|---|---|
| φ kepala | 0,040 [−0,008, +0,088] | **+0,0269** [−0,0063, +0,0602] |
| φ ruang-output | 0,040 [+0,003, +0,077] | +0,0030 [−0,0009, +0,0069] |

Per-seed, φ kepala bergerak di 2 dari 5 seed; satu **LULUS** (s3: T1 +0,0450, T2 +0,0547),
satu **MENUKAR** (s1: T1 −0,2550, T2 +0,0800). iNat-2018 **tidak bergerak sama sekali** —
λ = 0 di setiap seed, semuanya tepat +0,0000, meski bin aktif.

**Kesimpulan, dan ia negatif:** setelah objektif seleksinya dibuat terukur, PCC **tetap
tidak menunjukkan manfaat yang tertegakkan** di dump ekor-panjang yang dirilis. CI memuat
nol untuk kedua keluarga. Ini berbeda secara mendasar dari tiga run sebelumnya — di sana
metodenya **tidak pernah bertindak**; sekarang ia bertindak dan efeknya tidak signifikan.

**Satu cacat baru yang menutup sebagian angkanya:** `size_matched` gagal di sebagian seed
Pl@ntNet/head begitu λ > 0 (sebelumnya 1,0 di semua seed). Tanpa ukuran set tercocokkan,
perbandingannya tidak bermakna, jadi angka seed tersebut **tidak boleh dibaca**. Itu
melemahkan lagi agregat +0,0269 yang sudah tidak signifikan.

### Mengapa berhenti di sini

Batas keras yang disepakati sebelum perbaikan ini dijalankan: satu percobaan, lalu tulis
apa pun hasilnya. Percobaan itu sudah dilakukan dan menjawab pertanyaannya. Melanjutkan
dari titik ini — memperbaiki size-match, menambah seed, menyetel lebar bin — berarti
mengutak-atik sampai CI-nya bergerak, dan itu p-hacking dengan nama lain.

**Yang ditulis di paper:** klaim ekstrapolasi tertegakkan di ImageNet dengan deskriptor
eksogen, dan **tidak** tertegakkan di dua dump ekor-panjang yang dirilis — dengan mekanisme
terukur, bukan sekadar disebut: kelas berkalibrasi dua baris membuat objektif per-kelas
degenerate, dan bahkan setelah dikelompokkan ke bin, dayanya tidak cukup.

## BACKBONE KEDUA (2026-08-16) — dan batas yang akhirnya terlihat

Notebook 08 menjalankan PCC di atas skor ResNet-50 yang kita hitung sendiri (cache logit
UMTTA, suhu Platt mereka), tiga keluarga φ × empat metode agregasi × 3 seed. 48/48 jalan.

**Tidak satu pun signifikan.** Setiap CI antar-seed memuat nol, dan besarannya ~100×
lebih kecil dari hasil CCC:

| φ | Tabel 2 (`n_y = 0`) |
|---|---|
| `head_rn50` (model sama) | −0,0046 … +0,0015 |
| `head_vit` (**eksogen**) | **tepat +0,0000** — λ=0 di 11 dari 12 konfigurasi |
| ruang-output | +0,0011 … +0,0086 |

### Pola yang muncul setelah empat setting, dan ia monoton

Yang membedakan bukan dataset, bukan backbone, bukan keluarga φ. **Kedalaman kalibrasi
per kelas:**

| Setting | baris CAL / kelas | Tabel 2 (`n_y = 0`) |
|---|---|---|
| **ImageNet CCC** | **76** | **+0,0249** [+0,0054, +0,0444] ✓ |
| ImageNet UMTTA | ~20 | ≈ 0, CI memuat nol |
| Pl@ntNet | ~12 | +0,0269, CI memuat nol |
| iNat-2018 | ~3 | tepat +0,0000 |

Satu setting yang berhasil punya **tiga sampai dua puluh lima kali** lebih banyak data
kalibrasi per kelas daripada yang lain. Itu penjelasan tunggal yang konsisten dengan
keempatnya, dan ia terukur — bukan dugaan.

Mekanismenya masuk akal: g_θ dilatih pada δ_y kelas TRAIN, dan δ_y itu sendiri kuantil
empiris dari `n_cal` sampel. Kalau `n_cal` tipis, targetnya berderau, `r_δ` turun,
plafonnya turun, dan seleksi λ menolak bertindak. Rantainya sama dengan yang menjelaskan
kegagalan α kecil (`r_δ` 0,829 → 0,738 → 0,620): **δ_y yang berderau, apa pun sebabnya,
menutup ruang gerak metodenya.**

### Konsekuensi jujur untuk paper

Klaim ekstrapolasi paling menarik justru di setting **label langka** — dan di sanalah ia
**tidak bekerja**. Setting yang berhasil punya 76 sampel kalibrasi per kelas, yang bukan
setting langka-label.

Itu ketegangan inti proyek ini, sekarang **terukur pada empat setting**, bukan
diperdebatkan. Ia harus ditulis sebagai temuan utama, bukan sebagai batasan di kaki
halaman:

> Koreksi kelas terprediksi memperbaiki ekuitas worst-class pada kelas tanpa sampel
> kalibrasi **bila kelas-kelas lain punya kalibrasi yang cukup dalam** (≈76 baris/kelas
> pada ImageNet). Di bawah ~26 baris/kelas efeknya hilang ke dalam derau di empat dataset
> dan dua backbone. Prasyaratnya bukan jumlah kelas, bukan arsitektur, dan bukan pilihan
> deskriptor — melainkan **kedalaman kalibrasi kelas yang dipakai melatih prediktornya.**

### Yang TIDAK boleh disimpulkan dari run ini

`head_vit` memberi tepat nol di hampir semua konfigurasi, jadi run ini **tidak menjawab**
pertanyaan eksogenitas (apakah φ harus dari model lain). Objektifnya tidak bergerak sama
sekali di sini, persis seperti Pl@ntNet sebelum perbaikan bin. Pertanyaan itu tetap hanya
terjawab oleh CCC ImageNet, di mana φ kepala memang eksogen dan hasilnya positif.

## Ringkasan status klaim

| Klaim | Status |
|---|---|
| δ_y terprediksi dari geometri kelas | **tertegakkan** (nb 05, dua keluarga φ) |
| Koreksinya membeli ekuitas pada `n_y = 0` | **tertegakkan untuk φ kepala di ImageNet, α=0,10** |
| Tidak dibayar oleh kelas ber-data | **tertegakkan** (Tabel 1 juga positif) |
| Berlaku di dataset ekor-panjang | **TIDAK** — diuji 2026-08-16 setelah objektifnya dibuat terukur; CI memuat nol di Pl@ntNet, nol mutlak di iNat |
| Mengalahkan Clustered CP pada ukuran tercocokkan | **belum diuji** |
| Berlaku lintas α | **tidak** pada α=0,05 sejauh ini |
