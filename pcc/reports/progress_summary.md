# Predicted Class Correction (PCC) — Ringkasan Progres Riset

**Diperbarui:** 2026-08-14 · **Dataset:** ImageNet-1k (CCC) · Pl@ntNet-300K · iNat-2018 (LTC)
**Status:** Phase 0 LULUS · Phase 1 LULUS di ImageNet (4 gerbang) · **Phase 2 LULUS di ImageNet + φ kepala**
**Amandemen protokol:** 11 · **Tes regresi:** 101 lolos · **Metode `g_θ`: sudah ada**

Dokumen ini menjelaskan **apa yang diukur setiap gerbang, bagaimana cara mengukurnya, dan mengapa
dirancang begitu** — bukan hanya angka hasilnya. Angka tanpa prosedurnya tidak bisa direview.

> **BACA INI DULU — §1–§9 ditulis 2026-08-06 dan berbasis Pl@ntNet.** Sejak itu banyak yang
> berubah: gerbangnya dijalankan ulang di **ImageNet** (1.000 kelas, 1,15 juta baris), keluarga
> deskriptor **kedua** ditambahkan, metodenya dibangun, dan Phase 2 dijalankan. Bagian **§10–§14**
> memuat keadaan terkini dan **menggantikan** angka Pl@ntNet di bagian awal. Kalau membuat slide,
> ambil dari §10 ke bawah.

---

## 0. Ringkasan eksekutif — untuk slide

**Pertanyaannya:** bisakah koreksi konformal per-kelas δ_y **diekstrapolasi** ke kelas yang punya
**nol** sampel kalibrasi, dari deskriptor geometrik φ(y)? Itu satu-satunya klaim yang memisahkan ini
dari Clustered CP, Fuzzy Classwise CP, RC3P, dan sejenisnya — mereka semua **mengestimasi** δ_y dari
sampel kelas itu sendiri.

**Jawaban sejauh ini: ya, di ImageNet, dengan deskriptor yang eksogen.**

| Klaim | Status | Bukti |
|---|---|---|
| δ_y punya struktur di tingkat kelas | **tertegakkan** | Phase 0, divalidasi 4/4 di dunia tertanam |
| δ_y **terprediksi** dari geometri kelas | **tertegakkan** | gate B/C, 1.000 kelas, **dua** keluarga φ |
| bukan sekadar prevalensi atau jarak | **tertegakkan** | gate C, p ≤ 0,001, 73σ–102σ |
| koreksinya **membeli ekuitas** di kelas `n_y = 0` | **tertegakkan** | Tabel 2 **+0,0249** [+0,0054, +0,0444] |
| tidak dibayar oleh kelas ber-data | **tertegakkan** | Tabel 1 **+0,0495** [+0,0059, +0,0930] |
| berlaku di dataset ekor-panjang | **belum** | λ=0 — sebab teridentifikasi, sudah diperbaiki |
| mengalahkan Clustered CP pada ukuran tercocokkan | **belum diuji** | baseline belum masuk tabel |
| berlaku lintas α | **tidak** pada α=0,05 | batas cakupan, dinyatakan |

**Tiga temuan yang berdiri sendiri sebagai kontribusi**, terlepas dari nasib metodenya:

1. **R² adalah mata uang yang salah untuk memilih deskriptor.** Keluarga dengan R² **lebih tinggi**
   (0,4975) ternyata bernilai **nol** di tingkat metode; keluarga dengan R² lebih rendah (0,3880)
   yang bekerja. Objektif worst-class dikuasai galat **ekor**, bukan galat kuadrat rata-rata.
2. **Keunggulan classwise CP sebagian dibeli dengan ukuran set.** Tampak jauh lebih baik
   (`max_gap` 0,173 vs 0,420) — tetapi dengan set 4,10 vs 1,91. Pada ukuran tercocokkan ia bisa
   **berbalik negatif** (−0,51 pada satu seed).
3. **Dataset ekor-panjang tidak bisa menopang uji prediktabilitas tingkat-kelas yang berdaya.**
   Kelas yang layak: ImageNet 1.000, Pl@ntNet 90, iNat-2018 63. Prior art memakai dataset kedua dan
   ketiga itu.

---

## 1. Masalah dan pertanyaan riset

### 1.1 Mengapa conformal prediction saja tidak cukup

Split conformal prediction memberi jaminan yang tepat namun **marginal**: dengan threshold
`q̂` yang dikalibrasi pada n sampel exchangeable,

```
P( y_test ∈ C(x_test) ) ≥ 1 − α
```

Probabilitas itu diambil atas **seluruh distribusi bersama** (x, y). Ia tidak menjanjikan apa pun
untuk kelas tertentu. Pada distribusi ekor-panjang, konsekuensinya bukan teoretis:

| Kuartil prevalensi | sampel kal./kelas | macro-coverage | ukuran set rata-rata |
|---|---|---|---|
| q0 (terlangka) | 1 | **0,270** | 2,31 |
| q1 | 1–3 | 0,414 | 2,25 |
| q2 | 3–12 | 0,658 | 2,00 |
| q3 (terpadat) | 12–616 | 0,877 | 1,51 |

Coverage marginalnya **0,900** — tepat pada target. Namun kelas terlangka hanya tercakup 27%.
Rata-rata yang sehat menyembunyikan kegagalan sistematis.

### 1.2 Perbaikan yang dikenal, dan mengapa ia mentok

Solusi standarnya classwise (Mondrian) CP: beri setiap kelas thresholdnya sendiri, atau ekuivalen
sebuah offset

```
δ_y = q̂_y − q̂_global
```

di mana `q̂_y` adalah kuantil conformal yang dihitung **hanya dari sampel kalibrasi kelas y**.
Di sinilah masalahnya: `q̂_y` butuh sampel berlabel untuk kelas y, dan kelas yang paling butuh
koreksi justru yang paling sedikit sampelnya.

Distribusi sampel kalibrasi Pl@ntNet:

| persentil | p0 | p25 | p50 | p75 | p90 | p100 |
|---|---|---|---|---|---|---|
| sampel/kelas | 1 | 1 | **3** | 12 | 50 | 616 |

**105 kelas punya nol sampel kalibrasi.** Untuk kelas-kelas itu `δ_y` tidak terdefinisi, dan
classwise CP jatuh ke fallback (threshold tak hingga → set memuat semua label, atau threshold
global → coverage kelas itu tak terkendali).

### 1.3 Pertanyaan riset

> **Bisakah δ_y diekstrapolasi ke kelas dengan NOL sampel kalibrasi berlabel, dari deskriptor
> geometrik φ(y) yang dihitung di ruang embedding?**

Distingsi yang memisahkan ini dari prior art bukan "koreksi per-kelas" — itu sudah ada
(classwise CP, Clustered CP, Fuzzy Classwise CP, PAS/Interp-Q, RC3P, TACP, CFCP). Distingsinya
adalah **estimasi versus ekstrapolasi**: semua metode itu *mengestimasi* δ dari label kelas
tersebut; kami *memprediksinya* untuk kelas yang tidak punya label sama sekali.

---

## 2. Metode

### 2.1 Skor nonconformity

Dipakai THR/LAC: `s(x, y) = 1 − p(y | x)`, dengan p dari softmax. Skor lebih tinggi = lebih
nonconforming. Set prediksi memuat label k bila `s(x, k) ≤ threshold`.

### 2.2 Alur lengkap

```
                     ┌─ split TRAIN ──────────────────────────────┐
                     │  embedding penultimate + logits            │
                     │  → φ(y) untuk SETIAP kelas y               │
                     │    (termasuk kelas tanpa data kalibrasi)   │
                     └────────────────────┬───────────────────────┘
                                          │
  ┌─ split KALIBRASI ─────────────┐       │
  │ q̂_global = kuantil(1−α)       │       │
  │ δ_y      = q̂_y − q̂_global     │       │
  │   (hanya untuk kelas yang     │       │
  │    punya cukup sampel)        │       │
  └──────────────┬────────────────┘       │
                 │  target latih          │  fitur latih
                 └────────────┬───────────┘
                              ▼
                      g_θ : φ(y) → δ̂_y
                              │
                              ▼
                    δ̃_y = λ · δ̂_y          λ dipilih pada kelas 𝒴_train
                              │
                              ▼
        threshold_k = q̂_global + δ̃_k     untuk SEMUA k, termasuk kelas tak-berlabel
```

### 2.3 Sifat keamanan — mengapa ini bisa dipertahankan tanpa asumsi pada g_θ

Koreksinya bekerja pada skor label-benar: `s'(x, y) = s(x, y) − δ̂_y`. Karena `δ̂_y` adalah fungsi
dari label `y` saja (bukan dari `x`), `s'` **tetap fungsi terukur dari (x, y)**. Split conformal
tidak mengasumsikan apa pun tentang bentuk fungsi skor — hanya exchangeability antara kalibrasi
dan test. Maka:

> **Validitas coverage marginal split-conformal tetap berlaku untuk δ̂ APA PUN**, seburuk apa pun
> prediktornya, tanpa asumsi apa pun tentang g_θ.

Kalau `g_θ` buruk, yang rusak adalah **efisiensi** (ukuran set), bukan jaminannya. Ini diuji
sebagai regression test terpisah (`tests/test_coverage_validity.py`) yang membangun set dari skor
mentah maupun skor terkoreksi lewat primitif yang sama.

Sifat ini penting untuk framing paper: metodenya **tidak bisa merusak jaminan**, hanya bisa gagal
memperbaiki efisiensi.

### 2.4 Deskriptor φ(y)

15 deskriptor per kelas, semuanya dari embedding penultimate ResNet-50 (2048-d) dan logits pada
**split TRAIN saja**:

| Kelompok | Deskriptor | Maksud |
|---|---|---|
| Kedekatan antar-kelas | `cos_knn_1`, `cos_knn_5`, `cos_knn_10`, `cos_knn_50` | jarak cosine centroid kelas y ke k centroid terdekat — seberapa "berkerumun" kelas ini |
| Sebaran dalam-kelas | `cov_trace`, `cov_eig_0`, `cov_eig_1`, `cov_eig_2` | trace dan tiga eigenvalue teratas kovarians dalam-kelas — seberapa menyebar |
| Skala | `mean_norm` | norma rata-rata embedding |
| Kepercayaan model | `logit_margin`, `softmax_entropy`, `frac_top1`, `mean_log1p_rank` | seberapa yakin model pada kelas ini |
| Ukuran | `n_eff`, `log_prevalence` | jumlah citra efektif (ditentukan kuota) dan log prevalensi |

**Leak guard.** Deskriptor HARUS dari TRAIN, karena kalau dihitung dari split kalibrasi ia akan
melihat data yang sama yang membentuk target δ_y, dan gate B jadi sirkular. Di Pl@ntNet:
deskriptor dari direktori `train` (45.756 citra ber-kuota), kalibrasi dari `val`, evaluasi dari
`test` — tiga sumber terpisah. Ditegakkan `assert` di notebook, bukan sekadar konvensi.

**Trik komputasi.** Eigenvalue kovarians dalam-kelas dihitung lewat matriks Gram: eigenvalue tak-nol
dari `XᶜᵀXᶜ` (2048×2048) sama dengan milik `XᶜXᶜᵀ` (q×q) dengan q = jumlah citra kelas. Untuk q=50
itu 185× lebih cepat, eksak sampai 5,7e-14. Tanpa ini, perhitungan stabilitas butuh berjam-jam.

### 2.5 Penyusutan (Amandemen 8)

`δ̃_y = λ · δ̂_y`, dengan λ dipilih pada kelas 𝒴_train, bukan pada kelas held-out. Alasannya
struktural, dibahas di §6.4 di bawah: objektif worst-class diatur oleh error **terbesar** di δ̂,
sedangkan R² mengendalikan error kuadrat **rata-rata**. Menerapkan δ̂ mentah (λ=1) **merugikan**.

λ adalah parameter bebas. Memilihnya pada kelas held-out akan membuatnya jadi alat memanufaktur
hasil positif, jadi disiplinnya ditegakkan di kode dan diuji.

---

## 3. Data dan reproduksi split

| Split skor | Sumber direktori | n | Keterangan |
|---|---|---|---|
| `cal` | `val` | **21.783** | subset kalibrasi milik LTC, direproduksi |
| `val` (proper) | `val` | 9.335 | sisa split LTC, tidak dipakai di sini |
| `test` (evaluasi) | `test` | **31.112** | |
| `train_quota` | `train` | **45.756** | ber-kuota per kelas, khusus deskriptor |

Arsip Zenodo memuat `{train: 243.916, val: 31.118, test: 31.112}` citra.

**Reproduksi split cal/val LTC.** LTC membagi direktori `val` (31.118) dengan seed tetap. Split itu
memakai **legacy global RNG** numpy (`np.random.seed(0)` diikuti `np.random.shuffle`), bukan
`default_rng`. Direproduksi dengan menyimpan-dan-memulihkan state RNG global agar tidak mencemari
seed lain: n=31.118 → 9.335 proper-val / **21.783 cal**, tepat sama dengan jumlah baris skor rilis.

Akurasi top-1 pada `cal`: **0,7950**.

---

## 4. Prasyarat — gate reproduksi checkpoint

### 4.1 Mengapa gerbang ini ada, dan mengapa ia yang pertama

Seluruh pipeline mengandaikan embedding kami berasal dari **model yang sama** yang menghasilkan skor
softmax rilis LTC. Kalau tidak: ekstraksi tetap jalan, deskriptor tetap terhitung, Phase 1 tetap
mengeluarkan R² dan reliabilitas — dan **semuanya tak bermakna, tanpa satu pun tanda kerusakan**.
Kegagalan senyap seperti ini tidak bisa ditangkap review; ia butuh gerbang.

Gerbang ini ditulis 2026-07-25, **sebelum** forward pass pertama, dan dipromosikan dari "open item"
menjadi prasyarat keras.

### 4.2 Metode — permutasi-invarian

Skor rilis (`*_softmax.npy`) berada dalam **urutan teracak yang tak bisa dipulihkan** (`shuffle=True`
di loader LTC). Jadi pemeriksaannya tidak menyelaraskan baris ke citra sumber; ia memakai statistik
yang invarian terhadap urutan baris:

| Kriteria | Yang diukur | Toleransi |
|---|---|---|
| **G1** | akurasi top-1 (sepenuhnya invarian urutan) | \|Δacc\| ≤ max(0,002; 3·SE) |
| **G2** | jarak L∞ nearest-neighbour baris ke himpunan rilis | median ≤ 1e-5; ≥99% dalam 1e-4 |
| **G3** | kurva probabilitas kelas-benar terurut | max \|Δ\| ≤ 1e-3 |
| **G4** | multiset label per-kelas | eksak |

### 4.3 Tiga kesalahan spesifikasi yang ditemukan saat gerbang pertama dijalankan

Dicatat karena §12 mensyaratkan perubahan kriteria eksplisit, dan karena **dua di antaranya akan
menghasilkan KEGAGALAN PALSU pada checkpoint yang benar**.

**C1 — toleransi G1 salah diturunkan (akan false-alarm).** `|Δacc| ≤ 0,002` mengasumsikan
perbandingan setara. Padahal gerbang membandingkan akurasi **subsample** (3.000 citra) dengan
**himpunan penuh** (21.783); derau sampling binomial saja `√(p(1−p)/n)` = **0,0073** pada p≈0,8,
n=3000 — lebih besar dari seluruh toleransinya. Diperbaiki jadi `max(0,002; 3·SE)`. Diverifikasi
masih menangkap kerusakan nyata: perbandingan yang sengaja dikorupsi gagal di **43,7 σ**.

**C2 — G4 diterapkan pada himpunan yang salah (akan selalu gagal).** Dipanggil pada subsample
3.000 versus array rilis 21.783; jumlah per-kelas tak mungkin cocok. Diperbaiki: G4 kini memakai
label seluruh subset cal yang direkonstruksi, yang **tidak butuh forward pass sama sekali**.

**C3 — G2 menghabiskan RAM (session crash).** `abs(reference[None] − chunk[:, None])` membentuk
array `[256, 21783, 1081]` float64 = **48,2 GB**. Blocking-nya di aksis yang salah. Diganti skema
pruning eksak: row-max bersifat 1-Lipschitz dalam L∞, jadi `|max(a) − max(b)| ≤ ‖a−b‖_∞`, dan
mengurutkan referensi menurut row-max memungkinkan binary search membuang segala yang di luar
`[q_max ± r]` **tanpa false negative**.

### 4.4 Satu bug lagi yang sempat menghasilkan kesimpulan salah

Radius pruning awal dipasang `tol = 1e-4`, sementara selisih row-max sebenarnya bermedian
**2,9e-4** — tiga kali lebih lebar dari jendelanya. Akibatnya **kembaran sejati setiap baris
tersingkir sistematis dari pencarian**, dan median dilaporkan 0,5536. Hanya baris tersaturasi
(row-max ≈ 1,0 di kedua sisi) yang tetap masuk jendela — persis tanda tangan "19,5% cocok, semuanya
yang percaya diri".

Diperbaiki dengan pelebaran progresif `(tol, 10·tol, 100·tol, 1e-2, 1e-1, ∞)` yang berhenti begitu
`d ≤ r`. Kondisi itu adalah **sertifikat optimalitas**: baris di luar jendela berbeda row-max lebih
dari `r ≥ d`, dan karena `L∞ ≥ |Δrow-max|`, jaraknya pasti melebihi `d`. Median jatuh **370×**:
0,5536 → **0,00149**.

Klaim "DIFFERENT MODEL → blocker §2.3.4" yang sempat dibuat di sel diagnostik juga **ditarik** —
11 argmax flip dari 21.783 sama konsistennya dengan nondeterminisme numerik sebagaimana dengan
checkpoint berbeda; akurasi saja tidak bisa memisahkannya.

### 4.5 Hasil akhir gerbang: FAIL, dijalankan di bawah pengecualian tertulis

| Kriteria | Nilai | Verdict |
|---|---|---|
| **G1** akurasi | kami 0,7950236 vs rilis 0,7945187 → Δ 5,05e-4 (**0,18 σ**) | LULUS |
| **G4** multiset label | 21.783 = 21.783, jumlah per-kelas eksak | LULUS |
| **G3** kurva prob-benar | max \|Δ\| **0,0040** (tol 1e-3) | GAGAL |
| **G2** NN baris | median L∞ **0,00149** (tol 1e-5); 21% dalam 1e-4, 83% dalam 1e-2 | GAGAL |

**Yang sudah tersingkirkan sebagai penyebab:** bobot berbeda (akurasi beda 11 dari 21.783),
citra/label/indeks kelas berbeda (G4 eksak), transform berbeda (diverifikasi byte-for-byte terhadap
sumber upstream `plantnet/PlantNet-300K/utils.py`), presisi numerik (probe TF32 on/off identik
sampai empat angka penting: 1,202e-03 keduanya).

**Sisa penyebab yang masuk akal:** perbedaan implementasi decode/resize citra — versi
Pillow/libjpeg, atau arsip Zenodo memakai encoding JPEG berbeda dari salinan lokal yang LTC pakai.
Tidak dapat direproduksi tanpa environment mereka.

**Keputusan (disetujui manusia).** Skor **milik kami** dipakai di seluruh jalur — kalibrasi, δ_y,
evaluasi, dan setiap baseline. Gerbangnya **tetap FAIL, permanen dan tercatat**; toleransinya tidak
disentuh. Kriteria pengganti "konfirmasi bobot" sempat dicoba lalu **dibatalkan karena tidak bisa
diturunkan secara berprinsip** — simulasi menunjukkan `frac ≤ 1e-2` memberi 0,998 untuk bobot-sama
versus 0,984 untuk bobot-berbeda, praktis tak terpisahkan, dan ambang apa pun akan dipilih *setelah*
melihat nilai observasinya.

Ini sengaja **tidak** dilabel ulang jadi LULUS. Reviewer harus melihat bahwa reproduksi eksak gagal,
apa yang sudah disingkirkan, dan mengapa kesimpulannya tidak bergantung padanya.

---

## 5. Phase 0 — apakah strukturnya memang berada di tingkat kelas?

### 5.1 Pertanyaan yang diuji

Sebelum "memprediksi δ_y" punya arti, harus dipastikan dulu bahwa struktur threshold yang
dibutuhkan **memang berlokasi di tingkat kelas**, bukan sekadar penamaan ulang dari sesuatu yang
lebih sederhana. Dua hipotesis alternatif yang harus dibunuh:

1. **Miskalibrasi global.** Mungkin modelnya cuma terlalu percaya diri secara seragam, dan satu
   temperatur global menyelesaikan semuanya.
2. **Kesulitan per-sampel yang diagregasi per kelas.** Mungkin "kelas sulit" hanyalah "kelas yang
   sampel-sampelnya sulit", dan koreksi berbasis kepercayaan per-sampel sudah cukup — tanpa perlu
   indeks kelas sama sekali.

### 5.2 Bagaimana cara mengukurnya (Amandemen 6)

Target: **`q*_y`**, kuantil (1−α) dari skor label-benar kelas y **pada split EVALUASI**. Setiap
mekanisme di-fit pada split KALIBRASI lalu memprediksi `q*_y`:

| Mekanisme | Prediksi tingkat-kelas | Parameter bebas |
|---|---|---|
| `global` | satu konstanta `q̂_global` | 1 |
| `energy_bK` | rata-rata (atas sampel eval kelas y) dari threshold per-bin yang ia berikan | K |
| `class` | `q̂_y` dari split kalibrasi | jumlah kelas |

Detail mekanisme energy: hitung free energy `E(x) = −logsumexp(logits)`, bagi sampel kalibrasi ke K
bin kuantil (tepi bin **hanya dari energi kalibrasi**, tanpa kebocoran eval), hitung threshold
per-bin, lalu prediksi tingkat-kelas = rata-rata threshold yang mekanisme itu berikan pada sampel
eval kelas tersebut.

**R² diambil lintas KELAS, out-of-sample.** Konsekuensinya penting: mekanisme yang parameter
tambahannya berisi derau mendapat R² **negatif**. Terukur: `energy_b50 → −35,8` di dunia sintetik
tanpa struktur. **Kapasitas dihukum, bukan dihadiahi.**

**Temperatur diuji dengan pertanyaan yang bisa ia jawab.** Temperatur global secara struktural tidak
bisa menghasilkan varians tingkat-kelas, jadi menilainya lewat R² tingkat-kelas akan mencurangi
perbandingan. Yang ditanyakan justru: **apakah temperatur MENGHAPUS struktur tingkat-kelas?**
Diukur sebagai split-half reliability `q̂_y` pada skor yang sudah diskalakan temperatur. Dan bukan
hanya pada satu temperatur — di-scan pada grid geometris `[0,2 … 5,0]` dan diambil reliabilitas
**terburuk**, karena hipotesis alternatifnya adalah "ada temperatur yang menjelaskan ini", bukan
"temperatur optimal-efisiensi menjelaskan ini".

**Kriteria lulus (pre-registered).** Ketiganya harus terpenuhi:
1. struktur tingkat-kelas ada di atas derau: reliabilitas > 0,30
2. bertahan di SETIAP temperatur: reliabilitas terburuk sepanjang scan > 0,30
3. `class` R² melampaui setiap rival energy, dengan CI tak beririsan

### 5.3 Rezim "data berlimpah" yang §5 minta

§5 menyatakan offset per-kelas di-fit pada *data berlimpah, bukan budget realistis*. Karena median
Pl@ntNet 3 sampel/kelas, pada ruang label penuh mekanisme kelas hanya akan mengukur derau estimasi.
Jadi analisis **PRIMER** dibatasi ke kelas dengan ≥50 sampel kalibrasi: **98 dari 1.081 kelas**,
mencakup 16.021 dari 21.783 sampel (73,5%).

Pembatasannya diterapkan ke **sampel DAN kolom skor** (`restrict_to_classes`), sehingga tidak ada
asimetri "terukur versus tak-terukur" yang bisa dieksploitasi. Pandangan tak-terbatas dilaporkan
sebagai SEKUNDER dan tidak pernah digabung.

50 split cal/eval acak; kelas perlu ≥20 sampel eval agar kuantilnya bermakna (rata-rata 97,88 kelas
tersekor).

### 5.4 Hasil — α = 0,10

| Mekanisme | R² | 95% CI | korelasi | plafon jika diskalakan optimal (corr²) |
|---|---|---|---|---|
| `global` | −0,173 | [−0,195; −0,151] | +0,000 | 0,000 |
| `energy_b2` | −0,164 | [−0,182; −0,145] | +0,037 | 0,001 |
| `energy_b10` | −0,073 | [−0,087; −0,059] | +0,071 | 0,005 |
| `energy_b50` | −0,050 | [−0,062; −0,038] | +0,090 | 0,008 |
| **`class`** | **+0,583** | **[+0,563; +0,604]** | **+0,797** | **0,635** |

Reliabilitas `q̂_y`: **0,828** pada skala identitas, **0,688** pada temperatur terburuk sepanjang
scan. Pencarian temperatur **tidak** mentok di tepi grid (`best_T_at_boundary_frac = 0,0`), jadi
nilainya benar-benar ter-fit.

α = 0,05: `class` +0,367 [+0,342; +0,392] versus rival terbaik −0,116; reliabilitas 0,726 / 0,547.

### 5.5 Mengapa kolom korelasi ada, dan apa yang ia selesaikan

R² mencampur **arah** dengan **amplitudo**: mekanisme yang mengurutkan kelas dengan benar tetapi
meremehkan sebarannya akan mendapat R² negatif. Celah itu penting — pada data sintetik dengan
struktur kelas tertanam, `energy_b10` mencapai **korelasi +0,979** meski R²-nya +0,523. Di bawah
bacaan itu, energy yang diskalakan-ulang optimal (plafon corr² = 0,958) akan nyaris menyamai `class`
(0,976), dan hipotesis alternatif §5 sebagian besar **benar** — klaimnya tak bisa dipertahankan.

**Data nyata menyelesaikannya ke arah sebaliknya.** Korelasi energy pada Pl@ntNet hanya **0,090**,
jadi bahkan dengan penskalaan afin optimal ia menjelaskan **0,8%** varians, lawan **63,5%** untuk
identitas kelas.

Diskrepansi sintetik-versus-nyata itu instruktif, dan merupakan batasan generatornya, bukan datanya:
generator memberi setiap sampel dalam satu kelas kesulitan yang sama, sehingga energi rata-rata
per-kelas jadi fungsi nyaris deterministik dari kesulitan kelas. Kelas nyata punya sebaran internal.
**Dunia tanam memvalidasi bahwa metrik bisa MEMISAHKAN; ia tidak mengkalibrasi seberapa kuat rival
di data nyata.**

### 5.6 Diskriminan yang paling langsung terbaca

`sd(q*_y)` = sebaran kuantil per-kelas pada data evaluasi:

| Dunia | sd(q*_y) |
|---|---|
| Pl@ntNet nyata (α=0,10) | **0,348** |
| sintetik: struktur kelas ditanam | 0,084 |
| sintetik: hanya miskalibrasi global | **0,002** |

### 5.7 Kesimpulan Phase 0

**LULUS di kedua α.** Struktur threshold yang dibutuhkan benar-benar berlokasi di tingkat kelas.
Bukan miskalibrasi global (bertahan di setiap temperatur), dan bukan kesulitan per-sampel yang
dinamai ulang (free energy praktis tidak membawa informasi kuantil tingkat-kelas).

Verdict ini hanya bisa dibaca karena metriknya divalidasi **4/4** pada dunia tanam lebih dulu —
lihat §8.2.

---

## 6. Phase 1 — gate A, B, C

### 6.1 Gate A — reliabilitas δ_y, dan mengapa ia adalah PLAFON

**Yang ditanyakan.** Sebelum menanyakan "apakah δ_y bisa diprediksi", harus dipastikan **δ_y itu
sendiri bukan derau**. Kalau δ_y yang diukur dari satu setengah data tidak berkorelasi dengan δ_y
dari setengah lainnya, maka tidak ada yang bisa diprediksi — dan R² prediktif berapa pun adalah
artefak.

**Prosedurnya (100 repetisi).** Untuk setiap repetisi:

1. Bagi sampel **setiap kelas** jadi dua setengah acak (kelas dengan <4 sampel dilewati).
2. Hitung δ_y **independen** pada masing-masing setengah — termasuk `q̂_global`-nya sendiri per
   setengah, sehingga tidak ada konstanta bersama yang menaikkan korelasi secara palsu.
3. Korelasikan dua vektor δ_y **lintas kelas** (kelas yang hadir di kedua setengah).
4. Koreksi Spearman-Brown: `r_sb = 2r / (1 + r)` — karena tiap setengah hanya memakai ~separuh
   sampel per kelas, SB mengestimasi reliabilitas pengukuran panjang-penuh.

**Mengapa estimator level-matched.** Kuantil conformal punya level `⌈(n+1)(1−α)⌉/n` yang
**bergantung pada n**, dan mengembalikan `+inf` outright bila `n < ⌈1/α⌉ − 1`. Pada setengah-kelas
Pl@ntNet, itu membuat gate A mengembalikan **NaN di setiap split** meski 523 kelas nominal
memenuhi syarat. Diganti kuantil empiris (1−α) yang menargetkan persentil sama untuk setiap grup.
Dampaknya terukur: **523 kelas berkontribusi versus 204**, dengan reliabilitas praktis identik
(0,945 vs 0,949).

**Hasil.**

| | |
|---|---|
| Reliabilitas (SB-corrected) | **0,754** |
| 95% CI | [0,750; 0,757] |
| Ambang | 0,30 |
| Kelas memenuhi syarat (≥4 sampel kal.) | 453 |
| Split yang menghasilkan nilai | 100/100 |
| Kelas berkontribusi per split | 453 |

**LULUS.** Perlu disebut apa artinya secara jujur: dengan median 3 sampel/kelas, pada n_y=2 kuantil
empiris di α=0,1 mendekati *maksimum dari dua titik*. Jadi yang diukur lebih dekat ke **level
kesulitan kelas** daripada ke ekor kuantil-90 yang presisi. Itu tetap struktur tingkat-kelas yang
sah dan tetap yang dibutuhkan prediktor, tapi harus dinyatakan.

### 6.2 Dua plafon

Ada **dua** sumber atenuasi, dan keduanya harus dilaporkan bersama R² mentah:

| Plafon | Nilai | Cara ukur |
|---|---|---|
| `r_δ` — reliabilitas target | **0,754** | gate A di atas |
| `r_φ` — reliabilitas deskriptor | **0,805** | hitung ulang φ(y) pada resample bootstrap citra kelas, korelasikan lintas kelas |
| **gabungan `r_δ × r_φ`** | **0,607** | **model sempurna pun tidak bisa melampaui ini** |

Melaporkan R² mentah tanpa plafon ini adalah kesalahan analisis. R² pooled yang dicapai adalah
0,303 — sekitar **setengah** plafon. Artinya ruang perbaikannya bukan pada konsepnya.

### 6.3 δ_y pada n_cal tercocokkan (Amandemen 2)

**Masalah yang diperbaiki.** Bias estimator kuantil bergantung pada ukuran grup n, dan `n_y`
sebanding prevalensi. Maka δ_y berkorelasi **secara mekanis** dengan prevalensi bahkan ketika tidak
ada struktur kelas sama sekali. Diukur pada kontrol di mana setiap kelas punya distribusi skor
identik (δ_y sejati = 0 untuk semua):

| Estimator | corr(δ_y, log n_y) |
|---|---|
| conformal, n tak tercocokkan | −0,287 |
| empiris, n tak tercocokkan | +0,227 |
| **matched n_cal = 20** | **+0,016** |

Ini penting karena gate C menanyakan apakah geometri mengalahkan log-prevalensi, dan §6.5 akan
menghentikan proyek bila prevalensi menjelaskan sebagian besar δ_y. Kaitan prevalensi–δ_y yang
palsu akan mematikan proyek gara-gara artefak estimator.

**Prosedurnya.** Ambil tepat `n_cal` sampel acak per kelas, gabungkan pool itu untuk `q̂_global`,
hitung δ_y per kelas dari pool yang sama. Kelas dengan <n_cal sampel dikembalikan NaN — dan
jumlahnya dilaporkan, karena membuangnya sendiri merupakan seleksi terkait-prevalensi.

Kenapa default estimatornya empiris juga di sini: mencocokkan n menghilangkan ketergantungan bias
pada n_y, tapi kuantil conformal masih membawa **ketidakcocokan level** antara grup kelas kecil dan
pool besar. Pada n_cal=25, α=0,1: kelas di level 0,96, pool di 0,9004. Pada kontrol δ_y sejati nol:

| Estimator | mean δ_y | % kelas positif |
|---|---|---|
| conformal | **+0,1158** | 90% |
| empiris | −0,0093 | 39% |

Pada skala skor [0,1], offset palsu +0,116 menaikkan setiap threshold dan pernah membuat §6.4
melaporkan set **bertambah** ~12,7 label.

**Pengaturan dan hasil.**

| Setting | n_cal | kelas dengan δ_y | corr(δ_y, log n_y) | null permutasi |
|---|---|---|---|---|
| PRIMER | 25 | 152 / 1.081 | **−0,478** | mean +0,018, sd 0,061, \|p95\| 0,122 |
| Sensitivitas | 10 | 283 / 1.081 | −0,379 | mean +0,008, sd 0,061, \|p95\| 0,098 |

**Null permutasi wajib** untuk gate C: label kelas dipermutasi sehingga struktur kelas hancur
**tetapi jumlah sampel per kelas dipertahankan**, 30 repetisi. Efek prevalensi yang tidak melampaui
null ini adalah artefak estimator. Observasi −0,478 **melampaui** null — jadi hubungan
prevalensi–δ_y di Pl@ntNet itu **nyata**, bukan artefak. Implikasinya untuk gate C dibahas di §6.6.

### 6.4 Gate B — apakah δ_y terprediksi dari geometri, di kelas held-out?

**Prosedurnya (100 split).** Untuk setiap split:

1. Partisi **kelas** (bukan sampel) jadi 𝒴_train / 𝒴_held-out, 50/50. Split di tingkat kelas
   adalah inti persoalannya: split tingkat sampel akan mengukur interpolasi, bukan ekstrapolasi.
2. Fit **ridge tetap** (λ=1,0, fitur distandardisasi, intercept tak-dipenalti) pada 𝒴_train.
3. Evaluasi R² held-out pada 𝒴_held-out.

**Ridge-nya sengaja sederhana dan tetap.** Ia adalah *instrumen pengukuran*, bukan g_θ dari Phase 2.
Pencarian arsitektur pada g_θ dilarang sebelum gerbang lulus (§10); modul ini hanya mengukur apakah
sinyalnya ada.

**Kebijakan set fitur (pre-registered).** Set PRIMER = fitur dengan stabilitas ≥0,90; set `full`
(15 fitur) adalah pandangan sensitivitas. Di Pl@ntNet hanya **2 dari 15** yang lolos:
`cos_knn_5` dan `logit_margin`. Yang dibuang: `mean_norm`, `cos_knn_1`, `cos_knn_10`, `cos_knn_50`,
`cov_trace`, `cov_eig_0/1/2`, `softmax_entropy`, `frac_top1`, `mean_log1p_rank`.

Kebijakan ini memvalidasi dirinya sendiri di CIFAR-100: set `stable` LULUS sementara `full` GAGAL.
Dan di Pl@ntNet set 15-fitur memang tidak terpakai — 15 parameter pada 19 kelas latih overfit dan
memberi R² held-out negatif, sementara set 2-fitur mencapai +0,453 di stratum yang sama. Ditandai
otomatis dengan flag `underpowered` (n_train < 3p).

### 6.5 Gate C — apakah geometri mengalahkan prediktor trivial?

Ini kriteria yang **memisahkan proyek ini dari prior art**, dan yang paling mungkin gagal.
Prediktor pembanding (ablasi), semuanya dihitung berpasangan di dalam split yang sama:

| Ablasi | Isi | Mewakili |
|---|---|---|
| `log_prevalence_only` | log prevalensi saja | "prevalensi sudah cukup" (§6.5C) |
| `distance_only` | satu fitur jarak | baseline gaya Fargion, jarak kelas-terdekat |
| `prevalence+distance` | keduanya | gabungan prediktor trivial |

Selisih diambil **berpasangan di dalam split**, jadi varians antar-split saling batal. Ini sebabnya
gate C bisa lulus di tempat gate B gagal: uji berpasangan lebih berdaya daripada uji satu-sampel
atas R². Itu diharapkan, bukan anomali.

**Mengapa dinilai DI DALAM stratum prevalensi (Amandemen 5).** Stabilitas deskriptor naik monoton
dengan prevalensi — **0,684** di kuartil terlangka versus **0,922** di terpadat, selisih **0,238**
yang tak bisa dihilangkan pilihan kuota apa pun, karena kelas langka hanya punya 2–7 citra. Jadi
deskriptor geometri **paling akurat justru di tempat prevalensi paling tinggi**. Di-pool lintas semua
kelas, "geometri mengalahkan prevalensi" jadi terkonfound: geometri bisa tampak prediktif *karena*
kualitas deskriptor yang terkait prevalensi, bukan *terlepas dari* prevalensi.

Obat standar untuk confound adalah **mengondisikan** padanya. Di dalam satu kuartil prevalensi,
prevalensi hampir tak bervariasi (jadi tak bisa menjelaskan banyak) dan kualitas deskriptor kurang
lebih seragam (jadi gradiennya ditahan tetap).

Divalidasi pada cerita tanam: δ yang digerakkan prevalensi → geometri mengalahkan prevalensi di
**0/4** stratum; δ yang digerakkan geometri → **4/4**. R² pooled 0,935 versus 0,928 — praktis tak
berguna untuk memisahkan keduanya.

**Konsekuensi karena §3.3 tidak terpenuhi.** Karena stabilitas deskriptor tak pernah mencapai 0,90,
bacaan gate B menjadi **asimetris**, dan ini sudah disetujui: derau deskriptor hanya bisa
**MENGATENUASI** R², tidak bisa mengarangnya. Maka **LULUS bermakna; GAGAL AMBIGU**, bukan bukti
tidak ada sinyal.

### 6.6 Hasil gate B/C

**Pandangan pooled** (sekunder, karena terkonfound — tapi dilaporkan):

| Sel | R² held-out | 95% CI | R² ter-normalisasi | Gate B | Gate C |
|---|---|---|---|---|---|
| primer \| stable | +0,303 | [+0,119; +0,419] | 0,402 | LULUS | LULUS |
| primer \| full | +0,470 | [+0,319; +0,609] | 0,623 | LULUS | LULUS |
| sensitivitas \| stable | +0,292 | [+0,140; +0,377] | 0,388 | LULUS | LULUS |
| sensitivitas \| full | +0,394 | [+0,294; +0,499] | 0,523 | LULUS | LULUS |

**Pandangan bertingkat (PRIMER), primer | stable, 4 stratum pre-registered:**

| Stratum (sampel kal./kelas) | kelas | R² | lebar CI | Gate B | Gate C (prereg) |
|---|---|---|---|---|---|
| **q0: 25–40** | 38 | **+0,453** | 0,59 | **LULUS** | **LULUS** |
| q1: 41–70 | 38 | +0,261 | **1,38** | gagal | gagal |
| q2: 72–151 | 38 | +0,126 | 0,74 | gagal | gagal |
| q3: 151–616 | 38 | −0,037 | 0,86 | gagal | gagal |

**Kolom lebar CI adalah kunci membaca tabel ini.** Stratum q1 punya CI selebar **1,38 satuan R²** —
interval seperti itu tidak menguji apa pun. Dengan 19 kelas latih per stratum, uji bertingkat 4-cara
tidak bisa memisahkan efek sedang. Ini pernyataan **daya uji** yang bisa diturunkan tanpa melihat
hasilnya.

**Pandangan daya (2 stratum, dilaporkan berdampingan, bukan pengganti):**

| Sel | Stratum | kelas | R² | lebar CI | Gate B | Gate C | Gate C prereg |
|---|---|---|---|---|---|---|---|
| primer\|stable | q0: 25–70 | 76 | +0,398 | 0,36 | **LULUS** | **LULUS** | **LULUS** |
| primer\|stable | q1: 72–616 | 76 | +0,204 | 0,42 | gagal | gagal | gagal |
| sensit.\|stable | q0: 10–27 | 142 | +0,358 | 0,41 | **LULUS** | **LULUS** | **LULUS** |
| sensit.\|stable | q1: 27–616 | 141 | +0,178 | 0,32 | gagal | **LULUS** | **LULUS** |

Justifikasi 2 stratum adalah **daya**, bukan hasil: 2 stratum tetap mengondisikan pada prevalensi
dengan dua kali kelas per sel. Namun memilih 2 setelah melihat 4 gagal tetap bentuk seleksi, jadi
4-stratum tetap PRIMER dan dilaporkan meski gagal. **Argumen terkuat tidak butuh pandangan
2-stratum sama sekali:** di pandangan 4-stratum pre-registered, stratum prevalensi terendah sudah
lulus gate B DAN gate C terhadap baseline independen.

### 6.7 Pola kegagalannya, dan mengapa itu justru memperkuat klaimnya

Kegagalan terkonsentrasi di stratum **prevalensi TINGGI** — dan itu tempat yang paling tidak
konsekuensial. Kelas dengan 72–616 sampel kalibrasi **tidak butuh ekstrapolasi**; mereka bisa
memakai classwise CP secara langsung. Klaim proyek ini tentang kelas bersampel sedikit atau nol.

Klaim yang bisa dipertahankan karena itu **ber-cakupan**, dan cakupan itu membuatnya lebih kuat:

> δ_y dapat diprediksi dari geometri kelas, melampaui log-prevalensi dan baseline jarak, **untuk
> kelas dengan ≲70 sampel kalibrasi**. Di kelas berprevalensi tinggi hal ini tidak dapat ditunjukkan
> — dan di sana tidak relevan, karena kelas-kelas itu punya cukup data untuk dikalibrasi langsung.

Itu memberi aturan deployment yang bersih: **di bawah ambang data, pakai δ̂ terekstrapolasi; di
atasnya, pakai classwise CP.** Aturan itu harus jadi bagian dari metodenya, bukan tempelan.

### 6.8 Bukti terkuat untuk kebaruan, dalam satu pasang angka

| Kuantitas | Nilai |
|---|---|
| `corr(δ_y, log n_y)` | **−0,478** (melampaui null permutasi) |
| R² held-out `log_prevalence_only` | **−0,030** |
| R² held-out geometri (stable) | **+0,303** |

Prevalensi **sungguh berkorelasi** dengan δ_y — dan itu bukan artefak, karena melampaui null yang
mempertahankan jumlah sampel. Namun prevalensi **tidak punya daya prediksi lintas-kelas sama
sekali**. Korelasi monoton tanpa kemampuan generalisasi antar kelas.

Itu tepat distingsi estimasi-versus-ekstrapolasi yang jadi fondasi proyek ini, dalam bentuk yang
paling ringkas.

---

## 7. §6.4 — apakah koreksi terprediksi itu membeli sesuatu?

### 7.1 Mengapa gate B/C saja tidak cukup

δ_y hanyalah **proksi**. Yang sebenarnya dipedulikan adalah coverage dan ukuran set, dan relasi
δ_y → ukuran set tidak monoton sederhana lintas kelas. Maka gerbang ini wajib, bukan opsional:
predictability yang tidak menerjemah jadi manfaat berarti targetnya salah pilih.

### 7.2 Empat desain yang gagal, dan mengapa

Perlu lima operasionalisasi. Empat yang pertama dicatat karena kegagalannya informatif:

1. **Ukuran set pada coverage marginal nominal.** Terkonfound: coverage-nya melayang, jadi ia
   membandingkan dua level coverage berbeda. δ̂ dengan konstanta +0,116 tak terkoreksi membuatnya
   melaporkan set **bertambah 12,7 label**.
2. **Ukuran set pada coverage marginal TERCOCOKKAN.** Terkonfound secara fundamental: split-CP
   marginal **sudah optimal** untuk coverage marginal, jadi koreksi ber-indeks kelas **terbukti tidak
   bisa menang**. Oracle δ̂ +1,58 (lebih buruk), konstanta murni −0,93 (lebih baik) — peringkatnya
   tak bermakna. **Uji yang tidak bisa mengembalikan hasil positif bukan uji.**
3. **Coverage class-conditional, tapi δ̂ hanya diterapkan ke kelas held-out sementara set masih
   membentang ke SEMUA kelas.** Terkonfound: coverage dibatasi pada kelas held-out sementara ukuran
   dibayar lintas setiap kelas, jadi "naikkan threshold tepat pada kelas yang coverage-nya diukur"
   menang gratis. Konstanta murni **+15,54 mengalahkan** oracle **−16,57**.
4. **Ruang label held-out + deflasi diizinkan.** Tampak bekerja, dan dipakai untuk satu run penuh.
   **Cacat, dan buktinya ada di kontrolnya sendiri:** `oracle +0,045` versus
   `shuffled oracle −15,39`. δ̂ **sempurna** membeli praktis nol — jadi plafonnya ~0 sementara lantai
   deraunya −15, dan seluruh rentang dinamisnya negatif. Cacat yang sama dengan desain 2, dan
   terlewat.

Akar penyebabnya sama dengan Phase 0: **mencocokkan COVERAGE lalu membaca UKURAN SET adalah arah
yang ill-conditioned.**

### 7.3 Desain 5 — pembalikannya (Amandemen 8)

**Cocokkan sumber dayanya, baca manfaatnya.** Ukuran set rata-rata **mulus dan monoton naik**
terhadap pergeseran skalar threshold, jadi bisektor untuk mencapai ukuran target itu well-conditioned.
Coverage **terbatas di [0,1]**, jadi terbaca stabil.

Prosedur lengkap per repetisi (20 repetisi, 152 kelas usable):

1. Partisi kelas jadi 𝒴_train / 𝒴_held-out.
2. Fit ridge pada 𝒴_train, prediksi δ̂ untuk kedua himpunan.
3. **Pilih λ pada ruang label 𝒴_train saja**, dengan menggeser `q̂_global + λ·δ̂` ke ukuran target
   dan mengambil λ yang memaksimalkan worst-class coverage.
4. Terapkan λ itu **tanpa diubah** pada ruang label 𝒴_held-out. Restriksi ke ruang label held-out
   mencakup **sampel DAN kolom skor**, sehingga asimetri desain 3 hilang.
5. Ukuran target = ukuran yang dicapai vektor tak-terkoreksi. Bandingkan worst-class coverage.

**Objektifnya worst-class, BUKAN macro.** Ini penting dan terukur:

| Objektif | Δ oracle pada ukuran tercocokkan |
|---|---|
| worst-class coverage | **+0,312** |
| macro-coverage | +0,012 |

Macro **tidak punya ruang gerak bahkan untuk oracle**, karena threshold seragam sudah nyaris optimal
untuk sebuah rata-rata tak-berbobot. Itu Jensen, bukan properti δ_y. Macro tetap statistik yang
benar untuk **laporan ekor** yang sifatnya deskriptif, bukan komparatif.

**Mengapa penyusutan wajib.** Objektif worst-class diatur oleh error **TERBESAR** di δ̂, bukan
variansnya — dan R² mengendalikan error kuadrat **rata-rata**, sehingga syaratnya makin ketat seiring
jumlah kelas bertambah (maksimum dari K error Gauss ≈ 2,7σ pada K=60). Break-even δ̂ mentah ada di
ρ=1,0, yaitu prediktor sempurna. Terukur pada dunia tanam:

| Kualitas prediktor | λ terbaik | Δ worst-class | λ=1 mentah |
|---|---|---|---|
| R² = 0,30 | 0,10 | +0,025 | −0,403 |
| R² = 0,56 | 0,10 | +0,054 | −0,352 |
| R² = 0,90 | 0,20 | +0,126 | −0,116 |
| oracle | 0,70 | +0,390 | +0,340 |

λ optimal jauh lebih agresif daripada yang disarankan atenuasi regresi, karena objektifnya
sensitif-ekor, bukan rata-rata.

### 7.4 Hasil §6.4

α = 0,10, 152 kelas usable, worst-class coverage pada ukuran set rata-rata tercocokkan:

| Kondisi | Δ worst-class coverage | Arti |
|---|---|---|
| **δ̂ terprediksi (tersusut)** | **+0,0748** [+0,0380; +0,1116] | hasil kami |
| null teracak | −0,1518 | dikalahkan dengan jelas |
| plafon oracle | +0,3926 | **19% ruang terpakai** |
| δ̂ mentah, λ=1 | −0,1733 | penerapan naif merugikan |
| λ dipilih pada 𝒴_train | 0,265 | nilai interior, tertransfer bersih |
| **desain 4 pada data yang sama** | **−60,575** | metriknya yang bermasalah |

**Worst-class coverage naik 7,5 poin persentase pada ukuran set rata-rata yang identik**, pada kelas
yang δ_y-nya tidak pernah diukur.

Angka 19% harus dibaca dua arah dengan jujur. Positifnya: masih ada **4–5× ruang naik** dari
deskriptor yang lebih baik. Negatifnya: baru seperlima yang terambil.

**Catatan koreksi.** Dari data tanam pada R²≈0,3 aku memprediksi `observed ≈ −0,002` dan menyebutnya
"nol dengan plafon terkuantifikasi". Data nyata memberi +0,0748 dengan CI mengecualikan nol. Dua
sebabnya: sebaran δ_y Pl@ntNet jauh lebih besar (sd 0,35), jadi kualitas prediktor yang sama membeli
lebih banyak coverage absolut; dan λ tertransfer bersih. **Kontrol sintetik membatasi apa yang bisa
dideteksi metrik; ia tidak meramalkan besar efek.**

### 7.5 Efek pada ekor

δ̂ tersusut (λ=0,265) diterapkan ke seluruh ruang label, threshold deployment-valid (kuantil
conformal):

| Stratum (sampel kal.) | macro-coverage | ukuran set |
|---|---|---|
| q0: 1 | 0,270 → 0,270 | 2,31 → **2,03** |
| q1: 1–3 | 0,414 → 0,414 | 2,25 → **1,95** |
| q2: 3–12 | 0,658 → 0,658 | 2,00 → **1,81** |
| q3: 12–616 | 0,877 → 0,863 | 1,51 → **1,40** |

Ukuran set turun di setiap stratum; macro-coverage praktis tak berubah di tiga stratum terlangka dan
turun 0,013 di yang terpadat. 105 kelas tanpa sampel kalibrasi tidak bisa dievaluasi per-kelas dan
dilaporkan terpisah, bukan dihitung nol.

**Catatan pembacaan.** §9 melaporkan `worst_class_coverage = 0,0` di kedua arm. Itu karena 105 kelas
tanpa data kalibrasi dan kelas bersampel-1 di ruang label penuh — statistiknya degenerate di sana.
§6.4 mengukur pada 152 kelas yang bisa dievaluasi, dan itu sebabnya ia bisa menunjukkan +0,075.

---

## 8. Temuan metodologis

Delapan amandemen protokol tercatat di `reports/protocol_amendments.md`. Semuanya lahir dari satu
pola: **instrumen pengukurannya rusak, dan kerusakannya menghasilkan angka yang tampak seperti
temuan.** Bagian ini adalah kerja yang sebenarnya.

### 8.1 Level kuantil conformal bergantung pada n — dan itu muncul empat kali

Level `⌈(n+1)(1−α)⌉/n` berbeda antara grup kecil per-kelas dan grup besar yang di-pool. Pada n=25,
α=0,1: kelas di level 0,96, pool di 0,9004.

Empat artefak terpisah dari akar yang sama:

1. **Setiap komponen Phase 0 negatif** pada logits CIFAR-100 nyata (offset per-kelas −5,6). Bukan
   temuan — ketidakcocokan level. Setelah diperbaiki: −5,98 → +1,98, dan itu cocok dengan oracle
   data-berlimpah +1,84.
2. **corr(δ_y, prevalensi) palsu −0,385** yang akan mematikan proyek lewat §6.5. Setelah matched-n:
   +0,016.
3. **mean δ_y = +0,116 pada kontrol yang δ_y sejatinya nol** — 90% kelas positif. Membuat §6.4
   melaporkan set bertambah 12,7 label.
4. **Gate A mengembalikan NaN di setiap split**, karena kuantil conformal mengembalikan `+inf` untuk
   kelas kecil dan filter `~isnan` meloloskan `inf`, meracuni korelasinya.

Aturan yang diturunkan dan dicatat: **setiap kali kuantil conformal per-grup dibandingkan dengan
kuantil yang di-pool, periksa levelnya sebelum mempercayai angkanya.**

### 8.2 Metrik harus divalidasi pada struktur tanam sebelum verdict-nya dibaca

Kriteria Phase 0 lama — ukuran set pada worst-class coverage — menyebut `temperature` sebagai
pemenang di **4 dari 4** dunia sintetik, termasuk dunia yang satu-satunya strukturnya adalah
kesulitan per-kelas. **Nol daya pisah.**

Tiga kaki bukti yang saling bebas menyalahkan metriknya, bukan datanya:

1. **Kapasitas bersarang memburuk monoton** pada logits nyata: `energy_b2 −47,2` → `b10 −67,4` →
   `b50 −74,7`. b50 bisa merepresentasikan segala yang b2 bisa; kriteria di mana kapasitas ekstra
   selalu kalah tidak mengukur lokasi struktur.
2. **4/4 dunia tanam dijawab salah** (di atas).
3. **Aritmetikanya.** Mencapai worst-class coverage lewat **inflasi seragam** menghukum **varians**
   threshold, apa pun sumbernya. Kelas yang masih kurang coverage setelah adaptasi justru yang
   threshold-nya paling rendah, jadi perbaikan aditif global melambungkan seluruh vektor
   (`class` butuh inflasi 0,0492 vs `global` 0,0117). Dan dengan `s = 1−p` pada model lemah hampir
   semua label salah menumpuk di [0,98; 1,0], jadi **inflasi +0,0117 memindahkan ukuran set dari
   13,6 ke 55,7**.

Mekanisme `class` **sempurna selaras dengan kriteria itu sendiri** (kriterianya per-kelas, indeksnya
per-kelas) dan tetap kalah dari tidak melakukan apa-apa. Kriteria yang mekanisme paling-selarasnya
pun kalah sedang mengukur hal lain.

Penggantinya divalidasi **4/4** lebih dulu (`tests/test_phase0_explain.py`), dan validasi itu adalah
prasyarat membaca verdict-nya.

### 8.3 Ukuran set nyaris vertikal terhadap threshold — cocokkan sumber daya, baca manfaatnya

Konsekuensi langsung dari §8.2, dan berlaku juga untuk §6.4. Mencocokkan coverage lalu membaca
ukuran memperbesar derau tanpa batas. Pembalikannya mengubah plafon oracle dari **+0,045** menjadi
**+0,31**.

### 8.4 R² adalah mata uang yang salah untuk statistik-minimum

Dibahas di §7.3. Konsekuensi praktisnya: **penyusutan wajib**, dan λ harus di-fit out-of-sample atau
ia jadi alat memanufaktur hasil.

### 8.5 Sebuah baseline bisa diam-diam bersarang di dalam model yang diuji

Screen stabilitas memilih `cos_knn_5` sebagai baseline jarak — padahal fitur itu **juga** salah satu
dari dua fitur set primer. Gate C jadi menguji "apakah `logit_margin` menambah sesuatu di atas
`cos_knn_5`", bukan "apakah geometri mengalahkan baseline prior art" yang dimaksud §6.5C. **Lebih
ketat daripada yang diregistrasi**, dan FAIL di bawah kriteria yang lebih ketat bukan temuan yang
sama dengan FAIL di bawah kriteria terdaftar.

Diperbaiki: kedua verdict dilaporkan terpisah — versi bersarang dan versi pre-registered dengan
baseline independen `cos_knn_1`. Terukur pada dunia tanam dengan sinyal hanya di `cos_knn_5`: versi
prereg lulus 4/4 sementara versi bersarang lulus 1/4.

### 8.6 Gate C juga pernah tidak dijalankan sama sekali

Amandemen 5 menjadikan set ter-screen-stabilitas sebagai PRIMER. Set itu tidak memuat
`log_prevalence` maupun `cos_knn_1`, dan notebook menyerahkan Phi yang sudah dipotong. Ablasinya jadi
kosong, `gate_C_pass` jadi `None`, dan **gerbang yang paling mungkin gagal tidak pernah dijalankan**
untuk set fitur primer. Ablasi adalah *baseline*, bukan fitur model — sekarang selalu diambil dari
matriks Phi lengkap.

### 8.7 Kegagalan senyap butuh gerbang, bukan review

Dibahas penuh di §4. Poin protokolnya: gerbang itu ditulis **sebelum** forward pass pertama, dan
ketika ia gagal, ia **tidak dilabel ulang jadi lulus**. Kriteria pengganti sempat dicoba lalu
dibatalkan karena tidak bisa diturunkan secara berprinsip — dan pembatalan itu dicatat, bukan
disembunyikan.

### 8.8 Kesalahan lain yang tertangkap dan tercatat

| Bug | Dampak jika lolos |
|---|---|
| `n_eff` menjumlahkan **nilai indeks**, bukan menghitung elemen | `n_eff` berkorelasi dengan class id → **memalsukan predictability** |
| val loader memakai subset ber-augmentasi (CIFAR-100) | akurasi val terbaca 0,688 vs 0,826 sebenarnya; pemilihan epoch terbaik pada metrik yang salah |
| `true_rank` sebagai deskriptor | reliabilitas 0,468 — derau; diganti `frac_top1` / `mean_log1p_rank` |
| kovarians O(d³) pada 2048×2048 | perhitungan stabilitas berjam-jam; trik Gram 185× lebih cepat |
| `mean_ci` tidak sadar-NaN | satu split NaN meracuni seluruh CI — keluarga bug `gate A: nan` yang sama |
| `prevalence_null` KeyError pada dataset berimbang | CIFAR-100 akan crash; sekarang mengembalikan `undefined_reason` |
| tes ditulis setelah blok `if __name__` | tesnya tidak pernah berjalan |

Dua alat pencegah dibangun: `scripts/check_notebooks.py` (AST-parse setiap sel kode **dan** menandai
nama yang dipakai sebelum pernah di-bind — menangkap 2 bug yang sudah lolos ke user), dan disiplin
dry-run seluruh jalur notebook pada data sintetik berbentuk-Pl@ntNet sebelum diserahkan.

---

## 9. Batasan — nyatakan sebelum ditanya

1. **Satu dataset.** Semuanya di Pl@ntNet-300K. CIFAR-100 hanya untuk debug pipeline (akurasi 0,826,
   backbone latih-sendiri) dan tidak pernah jadi verdict. iNaturalist-2018 dan ImageNet belum
   dijalankan.

2. **Satu α untuk Phase 1.** Hanya α=0,10. Kuantil classwise butuh
   `⌈(n+1)(1−α)⌉/n ≤ 1`; pada Pl@ntNet, α=0,05 butuh ≥19 sampel/kelas (hanya 187 kelas) dan α=0,01
   butuh ≥99 (hanya 57 kelas). Ini **batas jumlah sampel, bukan hasil** — dan alasan §6.2
   mengharuskan kepala versus ekor dilaporkan terpisah, bukan di-pool.

3. **§3.3 TIDAK terpenuhi.** Stabilitas deskriptor puncaknya 0,815 (<0,90), dan kuartil ekor datar
   di ~0,68 karena kelas-kelas itu hanya punya 2–7 citra — **tidak bisa diperbaiki kuota**; menaikkan
   kuota justru **memperlebar** selisih kepala–ekor. Konsekuensinya asimetris dan sudah disetujui:
   **gate-B LULUS bermakna; gate-B GAGAL AMBIGU.**

4. **Skor bukan skor rilis LTC.** Gate reproduksi checkpoint gagal permanen; berjalan di bawah
   pengecualian tertulis (§4.5). Perbandingan bit-level dengan tabel terbit LTC **tidak diklaim**.

5. **Subset kelas terseleksi-prevalensi.** Gate A/B/C dihitung pada kelas yang punya cukup sampel
   untuk matched-n, jadi subsetnya bias ke kepala **secara desain**. Ini disebut di setiap laporan.

6. **Set 15-fitur tidak terpakai** pada jumlah kelas ini. φ(y) perlu **dirancang ulang, bukan
   diperbanyak**.

7. **Gate B/C bertingkat kurang daya** di 4 stratum (19 kelas latih/stratum, lebar CI sampai 1,38).
   Pandangan 2-stratum dilaporkan berdampingan, dengan justifikasi daya dan pengakuan bahwa
   memilihnya setelah melihat 4 gagal tetap bentuk seleksi.

---

## 10. Phase 1 dijalankan ulang di ImageNet — keempat gerbang LULUS

Pl@ntNet gagal gate B/C, dan diagnosisnya **daya uji**, bukan ketiadaan sinyal. Konsekuensinya
ditetapkan **sebelum** dijalankan: kalau dataset dengan kelas layak jauh lebih banyak lolos, itu
mengonfirmasi diagnosis daya uji.

**Dump:** CCC ImageNet, `(1.153.051 × 1.000)` — sepuluh kali lebih besar dari yang tercatat di audit.
Subsample terstratifikasi 21,7% (fraksi per kelas, bukan cap tetap, supaya struktur prevalensi utuh).
Split DESC 40% / CAL 30% / EVAL 30%, terverifikasi terpisah.

| Gerbang | Yang diukur | Hasil | Verdict |
|---|---|---|---|
| **A** | reliabilitas δ_y (split-half + Spearman-Brown) | **0,829** [0,827, 0,831], 1.000 kelas | LULUS |
| **B** | R² held-out, bootstrap tingkat-**kelas** | **+0,4975** [+0,461, +0,534]; ter-normalisasi **0,600** | LULUS |
| **C** | permutasi tingkat-kelas vs prevalensi & jarak, Holm | p ≤ 0,001 (**102σ** dan **84σ**) | LULUS |
| **§6.4** | apakah δ̂ membeli ekuitas pada ukuran tercocokkan | **+0,1173**, 76% plafon oracle, λ=0,088 | LULUS |

**Plafon:** `r_δ` 0,829 × `r_φ` 0,929 = **0,770**. §3.3 akhirnya terpenuhi: 8 dari 15 fitur berstabilitas ≥ 0,90, lawan 2 dari 15 di Pl@ntNet.

**Konsekuensi pra-registrasi terpenuhi:** kegagalan Pl@ntNet **terkonfirmasi sebagai masalah daya
uji**, bukan ketiadaan sinyal.

---

## 11. Dua keluarga deskriptor — dan mengapa yang kedua yang menentukan

Hasil di atas memakai φ **ruang-output**: dihitung dari matriks skor. Masalahnya **struktural**, dan
terukur: δ_y **juga** diturunkan dari matriks skor model yang sama. `conf_mean` pada DESC dan `q_y`
pada CAL adalah dua estimasi distribusi skor kelas yang **sama**. Baseline jarak sendirian
menjelaskan ~0,155 dari R² 0,497 — sisanya ringkasan skor langsung.

Jadi gate B di sana lebih dekat ke **estimasi distribusional** daripada **ekstrapolasi geometrik**.

**Keluarga kedua mematahkan sirkularitas itu:** baris bobot kepala klasifier `w_y`.

| | ruang-output | **kepala `w_y`** |
|---|---|---|
| sumber | matriks skor | matriks bobot |
| derau sampling | ada | **nol — eksak secara konstruksi** |
| butuh sampel berlabel? | ya | **tidak** |
| butuh citra / GPU? | tidak | tidak |
| model asalnya | **sama** dengan skor | **berbeda** (ResNet-50 torchvision vs SimCLRv2+probe) |

| Uji | ruang-output | kepala |
|---|---|---|
| gate B R² | +0,4975 | **+0,3880** [+0,345, +0,439] |
| gate C jarak | +0,3426 (84σ) | **+0,3753** (83σ) |
| gate C prevalensi | +0,4902 (102σ) | **+0,3710** (73σ) |
| §6.4 | +0,1173 (76% oracle) | **+0,0643** (42% oracle) |

**Struktur internalnya berkebalikan, dan itu poinnya.** Di ruang-output, satu fitur jarak
(`prof_knn_1`) menjelaskan 0,155 dari 0,497. Di keluarga kepala, `w_cos_knn_1` sendirian hanya
**0,013** dari 0,388 — jadi prediktabilitasnya datang dari geometri keputusan multi-fitur yang asli,
bukan dari satu jarak atau ringkasan skor.

**Implikasi terkuat:** geometri model yang **berbeda** memprediksi threshold model ini. Berarti
strukturnya properti **ruang label** (kebingungan visual antar kelas), bukan properti satu jaringan.

---

## 12. Baseline pada skor yang SAMA

Dipanggil dengan **kode penulisnya sendiri** (`utils/conformal_utils.py`, repo Clustered CP), bukan
reimplementasi. Grid Tier-1 (fungsi skor) × Tier-2 (metode kelas), α = 0,10, ImageNet:

| skor | metode | mean gap | **max gap** | avg set size |
|---|---|---|---|---|
| THR | standard | 0,0580 | **0,420** | **1,911** |
| THR | classwise | **0,0367** | 0,173 | 4,101 |
| THR | **clustered** | 0,0380 | 0,233 | **2,568** |
| APS | clustered | 0,0348 | 0,147 | 29,10 |
| RAPS | clustered | 0,0363 | 0,167 | 5,391 |

**Tiga bacaan:**

1. **Clustered CP mereproduksi klaim sentralnya** — gap nyaris setara classwise di **63% ukuran
   set**-nya. Ia bukan strawman, dan itulah baris yang harus dilampaui.
2. **Fungsi skor mendominasi sumbu ukuran set** — set APS **10–13× lebih besar** dari THR untuk gap
   praktis sama. Perbandingan wajib menahan fungsi skor tetap.
3. **Kolom yang sebanding dengan objektif kita `max gap`, bukan `mean gap`.**

**Target PCC dipatok SEBELUM `g_θ` ditulis** (`baseline_reproduction.md`): Tabel 1 harus
`max_gap ≤ 0,233` pada `size ≤ 2,568`; Tabel 2 harus mengalahkan `THR|standard` pada ukuran
tercocokkan. Dan peringatan strawman dinyatakan di muka: **menang di Tabel 2 saja bukan lulus**,
karena di sana setiap pesaing memang tak terdefinisi.

**§7 belum terpenuhi:** angka ini sahih untuk perbandingan **internal**, belum sebagai klaim
reproduksi (subsample 21,7%, protokol split berbeda, α/n_cal dipilih untuk gerbang).

---

## 13. Phase 2 — metode dijalankan

`pcc/method/pcc.py`: `g_θ` (ridge φ → δ̂ pada kelas TRAIN saja), aturan ambang data §6.7, blending,
rekalibrasi marginal. `pcc/experiments/phase2_pcc.py` sebagai driver ber-`--seed`; notebook 06 runner
tipis. **44 konfigurasi**, 3 dataset × 2 keluarga φ × 2 α × n_cal × 5 seed, ~69 menit, tanpa GPU.

**Yang diklaim metodenya, dan yang TIDAK.** Pada `n_y = 0` tidak ada sampel kelas untuk diambil
kuantilnya, jadi **tidak ada metode** yang bisa punya jaminan class-conditional finite-sample di
sana. PCC mengklaim tepat dua hal: **coverage marginal by construction** (atas distribusi
kelas-terlihat), dan **ekuitas kelas empiris membaik pada ukuran tercocokkan**. Tidak lebih.

### Hasil

| konfigurasi | λ | **Tabel 2 (`n_y=0`)** | Tabel 1 (terlihat) | verdict |
|---|---|---|---|---|
| **ImageNet + kepala** | 0,080 | **+0,0249** [+0,0054, +0,0444] | **+0,0495** [+0,0059, +0,0930] | **LULUS** |
| ImageNet + ruang-output | 0,000 | +0,0000 | −0,1026 | GAGAL |
| Pl@ntNet (kedua keluarga) | 0,000 | +0,0000 | +0,0050 | GAGAL |
| iNat-2018 (kedua keluarga) | 0,000 | +0,0000 | +0,0000 | GAGAL |

Ukuran set tercocokkan di 5/5 seed untuk baris yang lolos.

**Harganya, dilaporkan bukan disembunyikan:** macro coverage **turun** di kedua tabel (−0,0041 dan
−0,0036, CI tidak memuat nol). Pada ukuran tercocokkan, mengangkat kelas terburuk berarti
**memindahkan** anggaran dari kelas lain. Itu redistribusi — perilaku yang diharapkan dari metode
ekuitas, tetapi wajib ditulis sebagai trade-off.

### Mengapa yang lain λ = 0

λ = 0 berarti pemilihan λ **menolak koreksinya seluruhnya**. Metodenya menahan diri, dan itu
perilaku yang benar — tetapi sebabnya berbeda di dua tempat:

- **ImageNet + ruang-output:** δ̂-nya memang tidak berguna untuk objektif worst-class, meski R²-nya
  tertinggi. Inilah temuan pembalikan R² di §0.
- **Pl@ntNet dan iNat:** ini **kesalahan desainku**, bukan sifat data. λ dipilih lewat coverage
  worst-class pada slice CAL, dan di sana Pl@ntNet punya segelintir baris per kelas, iNat sekitar
  dua. Minimum atas ratusan kelas yang coverage-nya hanya bisa 0 atau 1 **tidak bisa digerakkan**,
  jadi tidak ada λ yang bisa memperbaikinya.

  Terbukti begitu: baris `head` dan `output` di kedua dataset itu menghasilkan angka **identik
  sampai 16 digit**. Dengan λ=0, ambangnya hanya bergantung δ_obs dan n_star — keduanya tidak
  bergantung φ. **Jadi keluarga kepala tidak pernah benar-benar diuji di ekor panjang; ia ditolak
  sebelum sempat dipakai.**

  `prereg_metrics_per_dataset.md` sudah menetapkan aturannya untuk **pelaporan**: di bawah 30 baris
  per kelas, minimum per-kelas adalah derau. Aturan itu berlaku sama kuat untuk **pemilihan**, dan
  tidak menerapkannya di sana adalah kelalaian. Sudah diperbaiki: λ kini dipilih pada kuantil
  ekor-bawah (p25) bila slice-nya terlalu tipis, dan statistik yang dipakai dicatat di tiap laporan.

---

## 14. Status, batas, dan langkah berikutnya

### Yang sudah jadi

| Komponen | Status |
|---|---|
| Phase 0 | **LULUS** (metrik divalidasi 4/4 lebih dulu) |
| Phase 1 — gate A/B/C + §6.4 | **LULUS di ImageNet**, dua keluarga φ |
| `pcc/method/` — g_θ, §6.7, prediktor | **selesai**, 13 tes |
| `pcc/experiments/` — driver ber-`--seed` | **selesai**, bit-per-bit reproducible |
| Notebook 06 — runner grid | **selesai**, 44 konfigurasi berjalan |
| Baseline standard / classwise / Clustered / APS / RAPS | **dipanggil**, 9 kombinasi |
| Pra-registrasi metrik per dataset | **selesai**, ditegakkan kode |
| Tes regresi | **101 lolos** |

### Yang belum

| Item | Catatan |
|---|---|
| φ kepala di ekor panjang | ditolak λ=0; perbaikan sudah masuk, **belum dijalankan ulang** |
| Baseline masuk tabel paper | §7 minta reproduksi vs angka terbit; perlu `MAX_ROWS = None` |
| Baseline Tier-2 sisanya | Fuzzy Classwise, PAS/Interp-Q, RC3P, TACP, CFCP — §7 sebut dua pertama **paling berbahaya** |
| α = 0,05 | GAGAL untuk φ kepala; hasil yang lolos spesifik α = 0,10 |
| iNat ukuran set | tidak cocok pada 1 dari 5 seed; angka seed itu tidak boleh dibaca |
| Perbedaan nb05 §6.4 vs nb06 | λ dipilih di slice EVAL vs CAL; harus **diputuskan**, bukan dipilih yang angkanya bagus |

### Batas yang harus dinyatakan, bukan ditunggu ditanya

1. **ImageNet berimbang.** Klaim aplikasi ekor-panjang belum tertegakkan di tempat motivasinya
   hidup.
2. **Ketidakcocokan model** (geometri ResNet-50 memprediksi threshold SimCLR+probe) adalah klaim
   substantif tentang ruang label, dan harus diargumentasikan sebagai klaim.
3. **42%, bukan 76%.** Deskriptor eksogen membeli **lebih sedikit** daripada deskriptor sirkular.
   Kedua angka harus muncul berdampingan.
4. **Tanpa penyusutan, metode ini merusak.** δ̂ mentah pada λ=1 memberi −0,508 (kepala) dan −0,234
   (ruang-output). Aturan pemilihan λ adalah **bagian metode**, bukan detail implementasi.
5. **Jaminan marginalnya atas distribusi kelas-terlihat**, bukan populasi penuh — kelas dengan
   `n_y = 0` tidak menyumbang baris kalibrasi apa pun.

### Penilaian jujur

**Tulang punggungnya sekarang lengkap dan terukur, bukan lagi diperdebatkan.** Ketiga mata rantai —
δ_y terprediksi, bukan dari prediktor trivial, dan koreksinya membeli ekuitas pada kelas tanpa data —
semuanya tertegakkan pada dataset dengan daya uji memadai, memakai deskriptor yang **eksogen**:
parameter dari model lain, nol sampel berlabel, tanpa citra, tanpa GPU.

**Risiko terbesar yang tersisa satu: apakah ini berpindah ke ekor panjang.** Run terakhir belum
menjawabnya — ia gagal karena kesalahan desain yang kini sudah diperbaiki, bukan karena datanya
menolak. Run berikutnya yang menjawab, dan jawabannya bisa saja tetap tidak.

**Yang tidak berisiko lagi:** tiga temuan metodologis di §0 berdiri sendiri sebagai kontribusi,
apa pun nasib metodenya.
