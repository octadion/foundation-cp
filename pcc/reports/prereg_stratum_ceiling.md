# PRE-DEKLARASI — plafon reliabilitas per-stratum (Amandemen 9)

**Ditulis:** 2026-08-07, **SEBELUM** angkanya dilihat. Penulis: coding agent.
**Disetujui manusia:** ya — "coba pada scope yang masih secara saintifik benar, bukan sampai p-hacking".

Dokumen ini ada supaya uji berikutnya bisa **gagal**. Tanpanya, ia hanya menambah satu lihatan lagi
ke tumpukan yang sudah banyak.

---

## 1. Berapa kali gate C sudah dilihat sebelum ini — dinyatakan terlebih dahulu

Menyembunyikan jumlah lihatan lebih merusak kredibilitas daripada lihatannya sendiri. Sampai
2026-08-06, gate B/C sudah dievaluasi pada:

- **4 sel**: {primer, sensitivitas} × {stable, full}
- **3 pandangan**: pooled, 4-stratum, 2-stratum
- **2 baseline jarak**: bersarang (`cos_knn_5`) dan pre-registered (`cos_knn_1`)

Itu banyak perbandingan simultan. **§8.6 Holm–Bonferroni sudah terimplementasi di repo tetapi belum
pernah dipakai.** Ia WAJIB dipakai pada uji di bawah.

## 2. Cacat yang diperbaiki, dan mengapa perbaikannya bukan penggeseran gawang

§6.3 sudah mensyaratkan R² held-out dilaporkan **ter-normalisasi oleh plafon reliabilitas**, dan
menyebut R² mentah tanpa plafon sebagai kesalahan analisis. Persyaratan itu dipenuhi — tetapi pada
**tingkat agregasi yang salah**: plafon `r_δ = 0,754` dihitung *pooled*, lalu dipakai menormalkan
R² di setiap stratum.

Reliabilitas adalah **rasio varians**. Di stratum yang δ_y-nya nyaris tidak bervariasi, plafonnya
jauh lebih rendah dari 0,754. Menormalkan stratum seperti itu dengan plafon pooled **secara
sistematis merendahkan** performanya.

Perbaikannya menerapkan koreksi yang **sudah disepakati** pada level yang benar. Ia tidak mengubah
kriteria lulus, tidak menambah fitur, tidak mengubah jumlah stratum, dan tidak menyentuh definisi
δ_y.

## 3. Pengamatan yang memicu ini

Estimasi titik R² pada `primary|stable`, 4 stratum pre-registered, **menurun monoton** dengan
prevalensi:

| Stratum | sampel kal./kelas | R² |
|---|---|---|
| q0 | 25–40 | +0,453 |
| q1 | 41–70 | +0,261 |
| q2 | 72–151 | +0,126 |
| q3 | 151–616 | −0,037 |

Ini **tidak konsisten** dengan penjelasan berbasis derau deskriptor, karena stabilitas deskriptor
bergerak ke arah **sebaliknya**: 0,684 di kuartil terlangka versus 0,922 di terpadat. Kalau derau
deskriptor penyebabnya, deskriptor yang lebih baik di kepala seharusnya memberi R² lebih baik.

Penjelasan alternatif yang lebih hemat: **`SS_tot` menyusut di kepala.** Karena
`R² = 1 − SS_res/SS_tot`, galat absolut yang sama menghasilkan R² lebih kecil di tempat target
kurang bervariasi. Kelas berprevalensi tinggi diperkirakan punya sebaran δ_y sejati yang jauh lebih
sempit — semuanya "mudah", berkerumun di sekitar δ ≈ 0.

Ini adalah kelas kesalahan yang **sama** dengan empat artefak yang sudah tercatat di proyek ini:
R² mencampur skala dengan akurasi.

## 4. PREDIKSI — dinyatakan sebelum angkanya dilihat

Diukur: `sd(δ_y)` per stratum, dan reliabilitas split-half δ_y **dibatasi ke kelas stratum itu**
(`split_half_reliability(..., class_subset=...)`, dengan `q̂_global` pooled tetap utuh).

**Jika penjelasan skala BENAR:**

- `sd(δ_y)` menurun monoton dari q0 ke q3;
- reliabilitas per-stratum menurun mengikuti pola yang sama;
- **R² TER-NORMALISASI mendatar** lintas stratum (tidak lagi menurun monoton).

**Jika penjelasan skala SALAH:**

- reliabilitas per-stratum kurang lebih **datar** sementara R² mentah tetap menurun;
- maka ini **batasan sejati**: geometri memang gagal memprediksi δ_y di kelas berprevalensi tinggi.

Kedua hasil dapat diterima dan keduanya akan dicatat apa adanya.

## 5. Uji PRIMER — satu, dinyatakan sekarang

> **Uji primer:** set fitur `stable`, δ_y pada n_cal = 25 (PRIMER), **4 stratum pre-registered**,
> baseline jarak **pre-registered** (`cos_knn_1`, independen — bukan yang bersarang),
> **R² ter-normalisasi oleh plafon per-stratum**, dengan **koreksi Holm–Bonferroni lintas 4 stratum**.

Semua yang lain — pandangan pooled, 2-stratum, set `full`, n_cal = 10, baseline bersarang — adalah
**SEKUNDER dan eksplisit**, dan tidak boleh dipakai sebagai verdict.

Kriteria lulus tidak berubah dari yang sudah tertulis: CI bawah R² ter-normalisasi > 0 di
esensialnya setiap stratum, dan `full` mengalahkan setiap ablasi.

## 6. Apa yang TIDAK akan dilakukan

- **Tidak menambah atau mengganti fitur.** Ide deskriptor bobot-`fc` (`w_y`) valid tetapi menaikkan
  **plafon**, bukan memperbaiki pola kegagalan ini. Ia dijalankan sebagai eksperimen terpisah, bukan
  dicampur ke run ini.
- **Tidak mengubah jumlah stratum.** 4 tetap primer.
- **Tidak mengulang** kalau hasilnya tidak diinginkan. Run ini satu kali; apa pun keluarannya, itu
  yang dicatat, dan dokumen ini yang menjadi buktinya.
- **Tidak mengubah kriteria lulus** setelah melihat hasilnya.

## 7. Konsekuensi ke depan — keputusannya tidak bergantung pada hasil ini

Phase 2 berjalan **baik cara**. Yang berubah hanya **klaim yang bisa dipertahankan**:

| Hasil | Klaim paper |
|---|---|
| R² ter-normalisasi mendatar | δ_y terprediksi dari geometri **seragam lintas prevalensi**; R² mentah menyesatkan karena varians target menyempit di kepala |
| tidak mendatar | terprediksi **di rezim data-sedikit** (≲70 sampel/kelas); di kepala tidak, dan di sana ekstrapolasi tidak dibutuhkan |

Yang pertama lebih kuat. Yang kedua sudah cukup untuk melanjutkan.

---

## HASIL (2026-08-07) — PREDIKSI TERFALSIFIKASI

Dijalankan sekali, sesuai komitmen. Prediksi di §4 **SALAH**.

| stratum | sampel kal. | sd(δ_y) | plafon per-stratum | R² mentah | R² ter-normalisasi |
|---|---|---|---|---|---|
| q0 | 25–40 | 0,2973 | 0,775 | +0,453 | +0,585 |
| q1 | 41–70 | 0,2730 | 0,516 | +0,261 | +0,505 |
| q2 | 72–151 | 0,3250 | 0,890 | +0,126 | +0,142 |
| q3 | 151–616 | 0,2522 | 0,908 | −0,037 | −0,041 |

- `sd(δ_y)` **tidak** menurun monoton (q2 justru tertinggi).
- Plafon per-stratum **NAIK** dengan prevalensi (0,775 → 0,908) — **berlawanan arah** dengan yang
  diprediksi.
- Sebaran R² **melebar** setelah normalisasi (0,490 → 0,625), bukan mendatar.

**Bacaan yang menentukan:** di q3, δ_y terukur paling reliabel (0,908) namun geometri
memprediksinya paling buruk (−0,037). Penjelasan skala mensyaratkan yang sebaliknya. Karena itu
verdict-nya **BATASAN SEJATI**: penurunan R² dengan prevalensi bukan artefak metrik.

Hipotesis di §3 salah. Dicatat, bukan direvisi.

### Koreksi multiplisitas (§8.6) — 0 dari 4 stratum lolos

Baseline jarak pre-registered (`cos_knn_1`), satu-sisi, Holm step-down:

| stratum | p | ambang Holm | tolak H₀ |
|---|---|---|---|
| q0 | 0,0133 | 0,0125 | **tidak** |
| q1 | 0,0151 | 0,0167 | tidak (diblokir step-down) |
| q2 | 0,0715 | 0,0250 | tidak |
| q3 | 0,2850 | 0,0500 | tidak |

q0 gagal terpaut **0,0008**. Holm bersifat step-down, jadi kegagalan pada p terkecil memblokir
sisanya.

**Batasan yang harus dinyatakan:** p ini adalah **aproksimasi normal** dari lebar CI
(`z = mean/(ciW/3,92)`), bukan p eksak. Pada kasus batas seperti q0 ia tidak cukup presisi untuk
memisahkan 0,0125 dari 0,0133. Uji permutasi akan lebih tepat — **dan sengaja TIDAK dijalankan
sekarang**, karena menjalankannya setelah melihat hasil ini adalah p-hacking yang dokumen ini ada
untuk mencegah. Uji permutasi di-pre-deklarasi untuk dataset berikutnya.

### Konsekuensi

Sesuai §7, keputusan tidak berubah: Phase 2 berjalan. Yang berubah adalah klaimnya, dan ia sekarang
**lebih lemah** dari kedua opsi yang diantisipasi:

> **§6.4 (outcome) LULUS** — koreksi terprediksi menaikkan worst-class coverage +0,0748 pada ukuran
> set tercocokkan, mengalahkan null teracak. **Gate C (kebaruan) TIDAK TERBUKTI** pada tingkat
> signifikansi yang di-pre-deklarasi: 0/4 stratum lolos koreksi multiplisitas.

Secara arah, geometri memprediksi δ_y di stratum terlangka (R² +0,453, ter-normalisasi +0,585,
mengalahkan prevalensi dan baseline jarak pre-registered). Itu menjanjikan tetapi **tidak
tertegakkan** pada level yang sudah dikomitmenkan.

**Gate C tidak dapat ditegakkan di Pl@ntNet pada daya berapa pun** — 38 kelas per stratum adalah
batas keras dari 152 kelas yang punya δ_y. Ini memindahkan jalur kritis: **dataset kedua bukan lagi
opsi validitas eksternal, ia satu-satunya cara menguji gate C dengan benar.** iNaturalist-2018
punya 8.142 kelas, sehingga daya berhenti menjadi faktor pembatas.

**Status:** dijalankan, prediksi terfalsifikasi, tercatat. Tidak ada run ulang.

---

## ADENDUM (2026-08-07, lebih lambat) — p-value Holm di atas TIDAK BERLANDAS

Ditemukan saat mengkalibrasi uji permutasi untuk dataset berikutnya. Dicatat di sini karena ia
melemahkan angka yang baru saja dilaporkan di dokumen ini.

**Cacat 1 — pembaginya salah.** p dihitung `z = mean/(ciW/3,92)`. Tetapi `_percentile_ci`
mengembalikan **persentil sebaran nilai per-split**, bukan CI atas rata-rata. Jadi pembaginya adalah
sd sebaran, bukan standard error. Rekonstruksi kasus q0 memberi p sign-flip **0,130** versus 0,0133
yang dilaporkan.

**Cacat 2 — dan memperbaiki Cacat 1 akan lebih buruk.** Mengganti pembagi menjadi
`sd/√n_splits` akan memperlakukan 100 split sebagai 100 observasi independen. Split-split itu
**pemakaian ulang 38 kelas yang sama**; sebaran antar-split adalah derau resampling, bukan derau
sampling dari populasi kelas. Membaginya dengan √100 akan melambungkan signifikansi ~10×.

**Akibatnya kedua uji itu keliru untuk desain ini**, termasuk uji sign-flip atas selisih per-split:
unit yang bisa ditukar adalah **KELAS**, bukan split. `pcc/eval/tail.py` sudah benar soal ini —
bootstrap-nya meresample kelas, bukan sampel. Gate C tidak.

**Temuan tambahan.** Null dari selisih berpasangan **bukan nol**. Terukur pada 38 kelas tanpa sinyal:
selisih teramati +0,109 sementara null permutasi berpusat di −0,102. Jadi kriteria gate C yang
berlaku, *"CI selisih mengecualikan 0"*, **bukan uji yang valid** — nol bukan nilai null-nya.

**Status angka Holm 0/4 di atas:** tidak berlandas. Ini **tidak** berarti gate C lulus; berarti gate
C **belum diuji dengan benar**. Verdict "BATASAN SEJATI" pada bagian sebelumnya berdiri pada
perbandingan reliabilitas-versus-R², yang tidak bergantung pada p-value ini, sehingga tetap berlaku.

**Tidak dijalankan ulang di Pl@ntNet.** Uji permutasi tingkat-kelas
(`predictability.class_permutation_p`) dipre-deklarasi untuk run berikutnya, bukan dipakai menambal
run ini. Terkalibrasi: 1.000 kelas tanpa sinyal → p 0,384; 1.000 kelas sinyal lemah → p 0,0066.
