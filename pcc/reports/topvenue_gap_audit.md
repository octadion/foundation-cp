# Audit kesiapan top venue — apa yang sudah, apa yang kurang

> **STATUS 2026-08-17.** Blok C (metrik) nol → **delapan**, blok D (statistik) satu →
> **lima**, blok E (ablasi) nol → **lima**, A3 kosong → **90 kondisi ImageNet-C**, A4
> THR-saja → **empat skor** (dua direproduksi lawan kode penulisnya), A2 dua → **tiga
> backbone** (top-1 cocok sampai empat desimal), B nol → **enam pesaing terpasang sebagai
> vektor ambang per-kelas**, F2 dan F4 → **ditulis** di `guarantees_and_exchangeability.md`.
> Yang tersisa: reproduksi §7 lawan angka terbit, SAPS/Class-Similarity/TACP/CFCP, G1, G4.
> Rincian per baris di bawah; tanda ✅/❌ aslinya dibiarkan agar perubahannya terbaca.

**Ditulis 2026-08-16, sebelum menulis kode tambahan apa pun.** Tujuannya supaya cakupan
eksperimen diputuskan sekali di depan, bukan ditambal tiap kali reviewer imajiner muncul.

Daftar ini **tidak** dibatasi pada apa yang sudah disebut sebelumnya. Beberapa item
terpenting justru tidak ada di daftar mana pun sejauh ini, dan ditandai **[BARU]**.

---

## A. Cakupan eksperimen

| # | Item | Status | Biaya menutup |
|---|---|---|---|
| A1 | **Dataset** — ImageNet, Pl@ntNet, iNat-2018 | ✅ 3 distinct | — |
| A2 | **Backbone** — SimCLRv2+probe, ResNet-50 | ⚠️ **2**; paper lama punya 4 | 2 run GPU (ViT-B/16, ConvNeXt-T) |
| A3 | **ImageNet-C** — jembatan ke UM-TTA | ❌ belum | 1–2 run GPU + unduhan Zenodo |
| A4 | **Fungsi skor** — THR, APS, RAPS | ⚠️ SAPS & PAS hilang | CPU |
| A5 | **α** — 0,01 / 0,05 / 0,10 | ✅ | — |
| A6 | **Seed** — 3–5 | ⚠️ paper lama 10 | CPU, gratis (Phase B) |
| A7 | **[BARU] Sapuan fraksi held-out** — 10/30/50% | ❌ **hanya 30%** | CPU |
| A8 | **[BARU] Sapuan kedalaman kalibrasi** — n_cal 10…200/kelas | ❌ ditemukan tak sengaja | CPU |

**A7 dan A8 adalah sumbu utama klaim kita, dan keduanya belum disapu.** A7 menjawab
"berapa banyak kelas boleh tanpa data sebelum metodenya runtuh". A8 mengubah temuan
kedalaman-kalibrasi dari **kecelakaan lintas-dataset** menjadi **kurva yang dirancang** —
pada CCC ImageNet ada 1168 baris/kelas, jadi kita bisa mensubsample 10 → 200 dan
menggambar batasnya secara langsung, di satu dataset, tanpa confound apa pun.

Itu satu-satunya perubahan yang bisa mengubah paper ini dari "menang di satu sel" menjadi
"**inilah kurva yang menentukan kapan ia bekerja**". Menurutku ini item paling bernilai
dalam seluruh daftar.

---

## B. Baseline — bagian terlemah

§7 melarang baseline masuk tabel sebelum direproduksi lawan angka terbit penulisnya.
**Nol dari yang kita punya memenuhi itu.**

| Metode | Kode | Status |
|---|---|---|
| standard / classwise / Clustered CP | repo LTC | ✅ terpasang sebagai **vektor ambang per-kelas**, dievaluasi lewat mesin yang sama dengan PCC; ❌ §7 reproduksi lawan angka terbit masih belum |
| APS, RAPS | repo CCC | ✅ dipanggil |
| **PAS + Interp-Q** | ~~repo LTC~~ | ❌ **tidak ada di repo itu** — diverifikasi 2026-08-17, klaim aslinya salah |
| **Fuzzy Classwise CP** | repo LTC | ✅ **terpasang**, 3 proyeksi × 5 bandwidth, oracle-tuned untuk pesaingnya. Satu-satunya metode terbit yang terdefinisi di `n_y=0` — `sd = bandwidth/(n_k+1)`, komentar penulisnya sendiri menyebut kelas berjumlah nol |
| **SAPS** | fungsi skor | ✅ **ditulis dari papernya** (tidak ada rujukan di rilis LTC), dikunci oleh properti; ❌ §7 reproduksi belum |
| **RC3P** | **repo LTC** (`rc3p`, `compute_rc3p_params`) | ✅ **terpasang** — ternyata ada di repo yang sama, tidak perlu dicari di github |
| Class similarity (Fargion) | github | ❌ — §7 sebut **paling berbahaya** |
| Macro-coverage (Bhattacharyya) | rilis | ❌ |
| TACP, CFCP | tidak ada kode | ❌ reimplementasi |

**Fuzzy Classwise dan Class Similarity keduanya memakai kemiripan kelas yang ditentukan
tangan** — itu pesaing konseptual terdekat PCC. Kalau keunggulan kita atas mereka masuk
noise, klaim kebaruannya runtuh. §7 sudah menyuruh menjalankan keduanya **awal**, dan kita
belum. Ini harus didahulukan di atas segalanya.

---

## C. Metrik — yang belum ada, dan reviewer CP akan mencarinya

| # | Metrik | Status |
|---|---|---|
| C1 | Marginal coverage | ✅ baru ditambahkan |
| C2 | Average set size | ✅ |
| C3 | Coverage gap vs target | ✅ baru ditambahkan |
| C4 | Worst-class / bin-worst | ✅ |
| C5 | **[BARU] SSCV** — size-stratified coverage violation | ✅ |
| C6 | **[BARU] Worst-slab coverage** | ✅ |
| C7 | **[BARU] %kelas di bawah target** | ✅ |
| C8 | **[BARU] Plafon oracle di tabel utama** | ✅ dan **diperbaiki 2026-08-17**: versi tak-disusutkan bisa dilewati PCC, jadi bukan plafon. Sekarang penyusutan terbaik dari δ sempurna. Plus **[BARU] `frac_empty_sets`**, yang dituntut oleh fase pergeseran |

**C6 dan C7 penting secara politis**, bukan cuma teknis: paper lamamu sudah melaporkannya,
jadi reviewer yang sama akan menganggap hilangnya sebagai kemunduran.

**C8 yang paling menentukan interpretasi.** Tanpa plafon oracle, +0,0249 tidak punya
skala. Di nb05, oracle memberi +0,1542 — jadi 42% ruang terpakai. Angka itu **harus** ada
di tabel utama, bukan di catatan gerbang.

---

## D. Rigor statistik

| # | Item | Status |
|---|---|---|
| D1 | CI antar-seed | ✅ |
| D2 | **[BARU] Uji berpasangan PCC vs tiap baseline** | ❌ mesinnya ada (`statistical_tests.py`, `pcc/eval/stats.py`), belum dipakai di Phase 2 |
| D3 | **[BARU] Koreksi Holm lintas metode** | ❌ — dipakai di nb05, hilang di Phase 2 |
| D4 | **[BARU] Effect size** | ❌ — reviewer uSxM secara spesifik memintanya di paper lama |
| D5 | Pra-registrasi + kriteria beku | ✅ **kekuatan yang jarang dimiliki paper lain** |

---

## E. Ablasi — top venue mewajibkan, kita nol

| # | Ablasi | Menjawab |
|---|---|---|
| E1 | **λ = 0 / λ terpilih / λ = 1** | apakah penyusutan memang bagian metode |
| E2 | **`n_star` mati vs hidup** | apakah aturan §6.7 berkontribusi |
| E3 | **Rekalibrasi marginal mati/hidup** | dari mana validitasnya datang |
| E4 | **Kelompok fitur φ** — jarak saja / prevalensi saja / penuh | mana yang membawa sinyal |
| E5 | **Keluarga φ** — kepala vs ruang-output vs ViT | ✅ sudah, satu-satunya yang ada |
| E6 | **Jumlah kelas latih g_θ** | berapa kelas minimum agar bisa diekstrapolasi |

Semua CPU. Tanpa ini, reviewer akan bertanya "bagian mana yang sebenarnya bekerja?" dan
kita tidak punya jawaban.

---

## F. Formalisasi

| # | Item | Status |
|---|---|---|
| F1 | Pernyataan jaminan yang tepat | ✅ ada di docstring `pcc.py`, belum jadi proposisi |
| F2 | **Bukti** marginal coverage terjaga | ✅ **Proposisi 1** di `guarantees_and_exchangeability.md`, 2026-08-17 |
| F3 | **[BARU] Nyatakan yang TIDAK dijamin** | ✅ sudah eksplisit — **ini kekuatan**, reviewer i5Df menghukum paper lama karena overclaim |
| F4 | **[BARU] Exchangeability di bawah shift** | ✅ **ditulis** 2026-08-17, dengan alasan konkret mengapa weighted CP dan adaptive CP tidak dipakai, dan `frac_empty_sets` sebagai bukti kuantitatif jaminannya batal |

**F4 tidak bisa dilewati.** Begitu ImageNet-C masuk, jaminan konformal tidak berlaku untuk
metode mana pun, dan itu harus dijawab dengan literatur, bukan didiamkan.

---

## G. Supplementary

| # | Item | Status |
|---|---|---|
| G1 | Tabel penuh semua dataset × skor × α | ⚠️ ada datanya, format belum |
| G2 | Parameter terpelajar per seed | ✅ tersimpan (λ, n_star, offset) |
| G3 | **[BARU] Tabel runtime/kompleksitas** | ❌ murah, dan **argumen kuat**: PCC tanpa training, ridge atas ≤1000 titik |
| G4 | **[BARU] Diagnostik kelas mana yang terbantu** | ❌ — mengubah angka jadi cerita |
| G5 | Reproduksibilitas: kode, seed, config | ✅ kuat |

**G3 murah dan bernilai tinggi.** "Nol training, milidetik, tanpa GPU, tanpa citra" adalah
klaim praktis yang membedakan dari Clustered CP (butuh clustering) dan RC3P.

---

## Urutan yang kusarankan, berdasarkan rasio nilai/biaya

**Gelombang 1 — CPU, dan ini yang menentukan nasib paper**

1. **Fuzzy Classwise + PAS/Interp-Q** (B) — pesaing terdekat; kalau kita kalah di sini,
   sisanya tidak relevan
2. **Sapuan kedalaman kalibrasi** (A8) — mengubah temuan aksidental jadi kurva rancangan
3. **Sapuan fraksi held-out** (A7) — sumbu utama klaim
4. **Plafon oracle + SSCV + worst-slab + %<target** (C5–C8)
5. **Ablasi E1–E4, E6**
6. **Uji berpasangan + Holm + effect size** (D2–D4)
7. **SAPS** (A4), **tabel runtime** (G3)

**Gelombang 2 — GPU**

8. **ImageNet-C** (A3) — jembatan ke UM-TTA, plus F4 harus ditulis
9. **+2 backbone** (A2) — ViT-B/16, ConvNeXt-T

**Gelombang 3 — opsional**

10. §7 reproduksi dump penuh
11. RC3P, Class Similarity, TACP, CFCP

---

## Penilaian jujur

Yang **sudah kuat** dan jarang dimiliki paper lain: pra-registrasi dengan kriteria beku,
aturan keterukuran metrik yang ditegakkan kode, batas yang dinyatakan sebelum ditanya, dan
jejak setiap cacat yang ditemukan beserta koreksinya.

Yang **belum**: baseline (paling parah), ablasi (nol), dan dua sumbu utama klaim kita
sendiri yang belum disapu.

Dan yang tidak bisa ditutup oleh eksperimen: **PCC positif di 1 dari 4 setting.** Gelombang
1 item 2 dan 3 adalah taruhan terbaik untuk mengubah itu — bukan dengan mencari kemenangan
baru, tetapi dengan **menggambar batasnya sebagai kurva** sehingga satu kemenangan itu
menjadi titik pada garis yang bisa diprediksi, bukan anomali.
