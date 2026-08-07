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

**Status:** ditulis, menunggu run tunggal notebook 04.
