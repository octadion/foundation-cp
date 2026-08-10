# PRE-DEKLARASI — run gerbang ImageNet, banyak skenario sekaligus (Amandemen 10)

**Ditulis:** 2026-08-07, **SEBELUM** data diunduh dan sebelum angka apa pun dilihat.
**Disetujui manusia:** ya — "pakai dataset apapun untuk uji, yang penting segera, uji banyak
skenario sekaligus".

Ini **RUN GERBANG**, bukan run paper. Ia bisa gagal dan menghentikan pengembangan metode.
Dataset ini menjadi dataset paper **hanya jika** lulus, dan angka paper harus datang dari porsi
evaluasi yang disisihkan dan tidak tersentuh keputusan gerbang.

---

## 1. Mengapa pindah dataset — dan bahwa ini penyimpangan

Urutan pre-registered (`release_audit.md`) menyatakan dataset berikutnya dijalankan **hanya jika**
gate Pl@ntNet + Phase 1 (A/B/C) + §6.4 semuanya lulus. **Gate B/C tidak lulus.** Jadi menjalankan
dataset lain sekarang adalah **penyimpangan dari urutan pre-registered**, dan dicatat begitu.

Alasan yang dipertahankan, dan yang bisa dibantah reviewer:

Kegagalan gate B/C di Pl@ntNet terdiagnosis sebagai **daya uji**, bukan absennya efek — 152 kelas
punya δ_y pada `n_cal=25`, sehingga stratifikasi 4-arah menyisakan 38 kelas per sel dan lebar CI R²
mencapai 1,38. Batas itu **tidak bisa diangkat di Pl@ntNet**: ia ditentukan oleh jumlah sampel
kalibrasi per kelas, dan tidak ada pilihan kuota, deskriptor, atau estimator yang menambah kelas.

Terukur (`predictability.class_permutation_p`, dikalibrasi 2026-08-07):

| skenario | p |
|---|---|
| 38 kelas, tanpa sinyal | 0,066 |
| 38 kelas, sinyal **kuat** | 0,0066 |
| 1.000 kelas, tanpa sinyal | 0,384 |
| 1.000 kelas, sinyal **lemah** | 0,0066 |

Pada 38 kelas hanya sinyal kuat yang terdeteksi. Itu pernyataan daya uji, dan ia diturunkan tanpa
melihat hasil Pl@ntNet.

## 2. Dataset dan mengapa yang ini

Dump skor **CCC ImageNet-1k**, `115.301 × 1.000` → ~115 sampel/kelas.

| dump | sampel × kelas | per kelas |
|---|---|---|
| CCC imagenet | 115.301 × 1.000 | **115** |
| LTC plantnet cal | 21.783 × 1.081 | median **3** |

Konsekuensi: pada `n_cal=25`, **seluruh 1.000 kelas** punya δ_y — 6,6× Pl@ntNet — dan ImageNet
**berimbang**, sehingga confound kualitas-deskriptor↔prevalensi yang memaksa Amandemen 5 **tidak ada**.

Ia juga benchmark milik **Clustered CP** (Ding et al., NeurIPS 2023), yang implementasinya dirilis —
jadi reproduksi baseline pada skor yang sama bisa dijalankan di run yang sama.

**Ukuran yang harus dikonfirmasi lebih dulu**, karena 115.301 ≠ 50.000 val standar: isi dump dibaca
dari `shape` dan `bincount` sebelum analisis apa pun. Kalau isinya bukan yang diasumsikan, run
berhenti dan itu dilaporkan.

## 3. Tiga split terpisah — leak guard

| split | porsi | dipakai untuk |
|---|---|---|
| **DESC** | 40% | φ(y) saja |
| **CAL** | 30% | δ_y, q̂_global |
| **EVAL** | 30% | semua metrik, §6.4 |

Stratified per kelas. φ dihitung **hanya** dari DESC; menghitungnya dari baris yang sama yang
menghasilkan δ_y membuat gate B sirkular. Ditegakkan `assert` di notebook.

`log_prevalence` diambil dari hitungan **DESC**, bukan CAL — supaya ablasi prevalensi tidak sekadar
mengkodekan kuota split.

## 4. φ(y) — keluarga ruang-output, dan apa yang ia BUKAN

`pcc.descriptors.output_space`, 15 fitur dari matriks skor saja: kepercayaan diri kelas (mean, sd),
entropi, margin, `frac_top1`, rank, geometri konfusi (`leak_max`, `leak_top5`, `leak_entropy`),
kNN cosine antar profil softmax rata-rata kelas (k = 1, 5, 10, 50), `n_eff`, `log_prevalence`.

**Ini BUKAN φ(y) embedding.** Ia geometri ruang **output**, bukan ruang embedding. Hasil di sini
**tidak otomatis berpindah** ke deskriptor embedding. Yang ia jawab: *"apakah geometri tingkat-kelas
memprediksi δ_y begitu jumlah kelasnya memadai"* — pertanyaan yang Pl@ntNet tidak bisa jawab pada
daya berapa pun. φ embedding menyusul lewat ekstraksi SimCLR.

Screen stabilitas ≥0,90 diterapkan dengan cara yang sama; `n_eff` dan `log_prevalence` ditandai
QUOTA_DETERMINED dan tidak boleh dikreditkan stabilitas yang harus diperoleh fitur terestimasi.

## 5. UJI PRIMER — satu, ditetapkan sekarang

> **Primer:** set fitur `stable` (hasil screen ≥0,90), δ_y pada **n_cal = 25**, **α = 0,10**,
> seluruh 1.000 kelas **tanpa stratifikasi** (prevalensi konstan secara desain, jadi confound-nya
> tidak ada dan stratifikasi tidak diperlukan).
>
> **Gate B:** CI bawah R² ter-normalisasi > 0, dengan CI **bootstrap tingkat-KELAS**
> (`predictability_class_bootstrap`, n_boot = 400), bukan sebaran antar-split.
>
> **Gate C:** null **permutasi tingkat-KELAS** (`class_permutation_p`, **n_perm = 1.000**) terhadap
> `log_prevalence_only` dan `distance_only_prereg`, dengan **Holm–Bonferroni** lintas keluarga uji.

Ambang p minimum yang bisa dicapai adalah `1/(n_perm+1) = 0,001`, cukup untuk ambang Holm.

**Mengapa bukan kriteria yang lama.** "CI selisih mengecualikan 0" **bukan uji yang valid**: terukur
pada 38 kelas tanpa sinyal, selisih berpasangan teramati +0,109 sementara null permutasi berpusat di
−0,102. Nol bukan nilai null-nya. Dan CI antar-split bukan inferensi atas populasi kelas — unit yang
bisa ditukar adalah kelas. Lihat adendum di `prereg_stratum_ceiling.md`.

## 6. SEKUNDER — dilaporkan, tidak boleh jadi verdict

- `n_cal = 50` (sensitivitas; ImageNet sanggup)
- **α = 0,01 dan 0,05** — dengan 115 sampel/kelas, α=0,01 butuh ≥99 dan **terpenuhi**. §8.8
  mensyaratkan sweep multi-α dan **Pl@ntNet secara struktural tidak pernah bisa memenuhinya**
- set fitur `full` (15 fitur)
- CI antar-split, berdampingan dengan CI tingkat-kelas
- §6.4 (desain Amandemen 8: ekuitas worst-class pada ukuran set tercocokkan, λ dari 𝒴_train)
- reproduksi **Clustered CP** pada skor yang sama, diverifikasi terhadap angka terbitnya
- perbandingan berdampingan dengan Pl@ntNet

## 7. Apa yang TIDAK akan dilakukan

- **Tidak menyetel ulang apa pun setelah melihat hasil.** Satu run.
- **Tidak mengubah `n_perm`, `n_boot`, α primer, `n_cal` primer, atau kriteria lulus** setelah
  angkanya terlihat.
- **Tidak menambah keluarga deskriptor** ke run ini. Deskriptor bobot-`fc` (`w_y`) dan φ embedding
  adalah eksperimen terpisah.
- **Tidak menyentuh `pcc/method/`.** §10 tetap memblokirnya sampai gerbang diputuskan.
- **Tidak menganggap hasil ruang-output sebagai hasil ruang-embedding.**

## 8. Konsekuensi yang sudah ditetapkan

| Hasil primer | Artinya | Tindakan |
|---|---|---|
| Gate B **dan** C lulus | Kegagalan Pl@ntNet **terkonfirmasi sebagai daya uji**, bukan absennya efek | Phase 1 dicatat lulus; lanjut φ embedding lalu Phase 2 |
| Gate B lulus, C gagal | δ_y terprediksi, tetapi **tidak melampaui prediktor trivial** — ruangnya sudah terisi | §6.5C terpicu: laporkan dan pertimbangkan menghentikan klaim kebaruan |
| Gate B gagal | δ_y **tidak** terprediksi bahkan pada 1.000 kelas | Batasan sejati; tulis hasil negatif |

Ketiganya dapat diterima. Yang tidak dapat diterima adalah melihat hasilnya lalu memilih kriteria.

---

## 9. DUA KOREKSI, dibuat sebelum data disentuh (dry-run sintetik, 2026-08-07)

Dry-run pada data berbentuk-ImageNet menemukan dua cacat yang **akan membuat gate C lulus secara
hampa**. Keduanya diperbaiki sebelum data nyata diunduh, jadi keduanya perbaikan bug, bukan pilihan
yang didorong hasil.

**Koreksi 1 — baseline jarak sama sekali tidak ada.** Default `distance_col` di `predictability()`
berisi nama deskriptor *embedding* (`cos_knn_*`), sementara keluarga ruang-output memakai
`prof_knn_*`. Namanya tidak cocok, jadi ablasi jarak **hilang tanpa suara** dan gate C hanya diuji
terhadap prevalensi. `distance_col` sekarang mencakup kedua keluarga.

**Koreksi 2 — dan ablasi prevalensi itu HAMPA di ImageNet.** Data berimbang → `log_prevalence`
nyaris konstan → `log_prevalence_only` memprediksi rata-rata dan mendapat ≈ −1/n_train, sehingga
model penuh mengalahkannya **secara dijamin**. Digabung dengan Koreksi 1, gate C akan "lulus" tanpa
pernah diuji.

`predictability()` kini mendeteksinya (`prevalence_ablation_degenerate`), tetap **menghitung dan
melaporkan** ablasinya — supaya terlihat bahwa uji itu dicoba — tetapi menandainya `vacuous` dan
**mengeluarkannya dari verdict**. Menghapus kuncinya akan menyembunyikan bahwa ujinya ada, dan juga
merusak konsumen (satu tes memang gagal karena itu, lalu diperbaiki).

**Konsekuensi untuk uji primer, dinyatakan sekarang:** di ImageNet gate C menguji **HANYA** lawan
baseline jarak. Itu lengan prior-art yang lebih penting (gaya Fargion), tetapi **uji yang lebih
sempit** daripada di Pl@ntNet, dan harus dilaporkan sebagai lebih sempit — bukan dihitung sebagai
kelulusan gate C yang setara.

**Baseline jaraknya dicadangkan agar independen.** `prof_knn_1` dikeluarkan dari model penuh dan
disimpan sebagai baseline. Kalau ia ikut di dalam model, gate C hanya menguji apakah fitur sisanya
menambah sesuatu di atasnya — submodel bersarang, bukan perbandingan terhadap prior art. Ini
kesalahan yang sama dengan Amandemen 7 di Pl@ntNet, dan kali ini dicegah di depan.

---

**Status:** ditulis, dua koreksi tercatat. Menunggu satu run `notebooks/05_imagenet_gate.ipynb`.
