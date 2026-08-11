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

---

## HASIL SURVEI DUMP (2026-08-07) — PREMIS LTC TIDAK TERPENUHI

Dijalankan dan **berhenti di sel 4**, sesuai desain. Ini keluaran yang §8 sebut sebagai kemungkinan,
dan ia sebuah **temuan**, bukan kegagalan teknis.

| dump | n | K | median sampel/kelas | kelas dengan ≥84 sampel |
|---|---|---|---|---|
| plantnet **test** | 31.112 | 1.081 | 3 | **90** |
| plantnet cal | 21.783 | 1.081 | 2 | 66 |
| inaturalist **test** | 46.227 | 8.142 | 2 | 63 |
| inaturalist cal | 32.359 | 8.142 | 2 | **0** |

Ambang 84 berasal dari `n_cal = 25 / FRAC_CAL = 0,30`: `n_cal` primer harus tercapai **di dalam**
porsi CAL dari pembagian tiga-arah, bukan di keseluruhan dump.

### iNaturalist-2018 lebih buruk dari Pl@ntNet, bukan lebih baik

8.142 kelas tetapi median **2** sampel/kelas, dan dump `cal`-nya punya **NOL** kelas yang memenuhi
syarat. Rekomendasi awal untuk pindah ke iNat didasarkan pada "8.142 kelas → daya uji berhenti jadi
pembatas"; itu **salah**, dan koreksinya kini terukur, bukan terinferensi. Daya uji gate C ditentukan
oleh **kelas yang punya cukup sampel KALIBRASI**, bukan oleh jumlah kelas.

### Temuan yang layak masuk paper

> Dataset yang dipakai prior art untuk conformal prediction ekor-panjang **secara struktural tidak
> dapat menopang uji berdaya untuk predictability tingkat-kelas.** Properti yang menjadikannya
> ekor-panjang — sedikit sampel per kelas — adalah properti yang sama yang mengeringkan ujinya.
> Terukur: kelas yang memenuhi syarat 90 (Pl@ntNet), 63 (iNat-2018), versus 1.000 pada ImageNet
> berimbang.

Ini pernyataan tentang **praktik evaluasi bidangnya**, bukan hanya tentang metode kami, dan ia
menjelaskan mengapa batas 38-kelas/stratum di Pl@ntNet tidak bisa diangkat dengan memilih dataset
ekor-panjang lain.

### Konsekuensi, sesuai §8

Lanjut ke dump **ImageNet CCC** seperti yang sudah dipre-deklarasi. Nilai-nilai primer
(`ALPHA_PRIMARY`, `N_CAL_PRIMARY`, `N_BOOT_CLASS`, `N_PERM_CLASS`, ambang stabilitas) **tidak
disentuh** — hanya sumber datanya yang berpindah, dan perpindahan itu sudah tertulis di §8 sebelum
survei dijalankan.

### Cacat notebook yang diperbaiki di sepanjang jalan

1. **Sel 4 hanya memperingatkan, tidak berhenti.** Kegagalan premis lalu muncul di sel 7 sebagai
   "deskriptor tidak stabil" (puncak 0,706) padahal sebabnya "sampel per kelas tidak cukup untuk
   mengestimasi deskriptor apa pun" — dengan DESC 40% dari median 3, separuh-split menyisakan ~0,6
   sampel/kelas. Sel 4 sekarang `assert` dan berhenti; sel 7 melaporkan anggaran sampelnya.
2. **Pola unduhan ditebak.** Dipakai `gdown.download_folder(id=...)` padahal notebook 00 sudah
   membuktikan `gdown <id> -O x.zip` lewat CLI. ID-nya menunjuk ZIP, bukan folder.
3. **Lokasi salah.** Dibuat direktori baru padahal notebook 00 menaruh skor di
   `released_scores/<dataset>`, sehingga dump Pl@ntNet yang sudah ada akan terunduh ulang.
4. **Pencocokan skor/label hanya menangani konvensi LTC** (`_softmax`/`_labels`). Berkas CCC tidak
   memakainya, jadi jalur CCC akan gagal tanpa menemukan apa pun. Sekarang ada fallback generik
   (2-D + 1-D sepanjang sama di direktori sama) dan inventaris lengkap saat gagal.
5. **Varian focal-loss bernama identik** di subdirektori berbeda; survei kini menandai variannya dan
   mengabaikan yang bukan `cross_entropy`.

---

## ISI DUMP CCC IMAGENET — audit salah sepuluh kali (2026-08-07)

Terbaca dari header `.npz` tanpa dekompresi:

| anggota | bentuk | dtype |
|---|---|---|
| `softmax.npy` | **(1.153.051, 1.000)** | float32 |
| `labels.npy` | (1.153.051,) | int64 |

`release_audit.md` mencatat `(115301, 1000)` — **sepersepuluh** dari yang sebenarnya. Angka itu
dipakai di §2 dokumen ini untuk mengklaim "~115 sampel/kelas". Yang benar **~1.153 sampel/kelas**.

Arah kesalahannya menguntungkan (premis butuh ≥84/kelas dan tersedia 13× lipat), tetapi angka di §2
salah dan dikoreksi di sini. 1.153.051 mendekati 90% ImageNet-1k train (1.281.167), jadi dump ini
kemungkinan skor pada porsi besar train, bukan val 50.000.

### Dua konsekuensi yang harus ditangani sebelum dijalankan

**Memori.** 1.153.051 × 1.000 float32 = **4,61 GB**, dan salinan turunan (`thr_lac`, entropi,
`np.partition` di `_top2_margin`) menambah beberapa GB lagi di atas RAM Colab ~12,7 GB. Sesi ini
sudah pernah crash karena alokasi besar (48 GB di gate checkpoint), jadi ini diperbaiki lebih dulu:
dump di-`mmap`, lalu **hanya subsample yang dimaterialkan** — 4,61 GB → ~1 GB pada `MAX_ROWS =
250.000`.

**Subsampling BUKAN perubahan kriteria**, dan diambil sebagai **FRAKSI per kelas, bukan cap tetap.**
Alasannya bukan estetika: cap tetap membuat setiap hitungan kelas **persis sama** → `log_prevalence`
konstan → ablasi prevalensi menjadi **hampa** dan gate C kehilangan satu lengannya. Fraksi
mempertahankan struktur prevalensi. Terverifikasi pada dry-run: sd(`log_prevalence`) = 0,004, jadi
ablasi prevalensi **tetap bisa diuji**. Anggarannya menyisakan ~250 sampel/kelas terhadap premis
≥84.

**`log_prevalence` diambil dari hitungan SEJATI seluruh dump**, bukan dari hitungan split DESC. §3
dokumen ini sudah menyatakan prinsipnya ("supaya ablasi prevalensi tidak sekadar mengkodekan kuota
split") tetapi implementasinya memakai `cnt_desc` — yang dengan subsampling **adalah** kuota. Kini
memakai `cnt_full`.

### Lima bug notebook yang ditemukan sebelum run berhasil

Semuanya kelas yang sama — **mengasumsikan alih-alih memverifikasi**, di kode yang tidak dapat
dijalankan penulisnya:

1. **Pola `gdown` ditebak** (`download_folder(id=...)`) padahal notebook 00 sudah membuktikan
   `gdown <id> -O x.zip` lewat CLI. ID menunjuk ZIP, bukan folder.
2. **Lokasi salah** — direktori baru dibuat padahal notebook 00 memakai `released_scores/<ds>`,
   sehingga dump yang sudah ada tak terlihat dan akan terunduh ulang.
3. **Dua saklar untuk satu tujuan** (`DO_CCC` + `SOURCE`), dengan **default yang sudah diketahui
   gagal** — setiap run default berakhir di assert. Saklarnya dihapus; semua sumber disurvei sekaligus.
4. **`.npz` dilewati** karena pencocokan ekstensi literal `.zip`/`.tar`. `.npz` *adalah* zip;
   unduhan 4,62 GB berhasil penuh lalu diabaikan tanpa suara.
5. **Pencocokan generik ditulis sebagai FALLBACK** (`if not pairs`). Dump LTC selalu menghasilkan
   pasangan, jadi fallback tak pernah jalan dan berkas CCC diabaikan — tabel survei kehilangan baris
   `ccc` sepenuhnya. Either/or di tempat yang seharusnya both.

Bug 4 dan 5, plus sufiks `.npy` pada indeks `NpzFile`, ditemukan oleh **dry-run sintetik**, bukan
oleh run pengguna. Itu disiplin yang seharusnya diterapkan sejak awal dan kini diterapkan konsisten.

---

## HASIL (2026-08-10) — KEEMPAT GERBANG LULUS, dengan cakupan yang harus melekat

`reports/05_imagenet_gate_ccc_imagenet.json`. Dump CCC ImageNet, 1.153.051 baris disubsample
terstratifikasi ke 250.013 (21,7% per kelas), 1.000 kelas, akurasi top-1 0,7769.

| | Pl@ntNet (φ embedding) | ImageNet (φ ruang-output) |
|---|---|---|
| kelas layak | 90–152 | **1.000** |
| fitur stabil ≥0,90 | **2 / 15** | **8 / 15** |
| §3.3 terpenuhi | **tidak** | **ya** |
| gate A `r_δ` | 0,754 | **0,829** [0,827; 0,831] |
| `r_φ` | 0,805 | **0,929** |
| plafon gabungan | 0,607 | **0,770** |
| gate B R² | +0,303 (antar-split) | **+0,4975** [+0,461; +0,534] (bootstrap KELAS) |
| gate B ter-normalisasi | 0,402 | **0,600** [0,555; 0,643] |
| gate C | 0/4 stratum (p tak berlandas) | **LULUS, kedua ablasi** |
| §6.4 | +0,0748, **19%** plafon oracle | **+0,1173**, **76%** plafon oracle |

Gate C, null permutasi tingkat-kelas, n_perm = 1.000:

| ablasi | observed | null (sd) | jarak dari null | p | Holm |
|---|---|---|---|---|---|
| `distance_only` (`prof_knn_1`) | +0,3426 | −0,0104 (0,0042) | **84 sd** | 0,0010 | tolak |
| `log_prevalence_only` | +0,4902 | −0,0104 (0,0049) | **102 sd** | 0,0010 | tolak |

p menyentuh lantai `1/(n_perm+1)`, jadi nilai sebenarnya lebih kecil dan tak teresolusi.

### Yang TERTEGAKKAN

**Kegagalan gate B/C di Pl@ntNet adalah persoalan DAYA UJI, bukan absennya efek.** Ini konsekuensi
yang sudah dipatok di §8 sebelum run, dan ia terpenuhi: dengan 1.000 kelas alih-alih 38 per stratum,
efek yang sama terdeteksi pada ~84–102 simpangan baku dari null.

Dan §3.3 — prasyarat yang tidak pernah terpenuhi di Pl@ntNet — **terpenuhi di sini**: 8 dari 15 fitur
melewati stabilitas 0,90, dengan median 50 sampel/kelas per separuh-split versus ~0,6 di Pl@ntNet.
Konsekuensinya asimetri Amandemen 5 ("gate-B FAIL itu ambigu") **tidak lagi diperlukan** untuk dataset
ini.

### Yang TIDAK tertegakkan, dan ini pembatas utamanya

**Predictability di sini didominasi RINGKASAN SKOR langsung, bukan geometri kelas.** Baseline jarak
`prof_knn_1` — satu-satunya analog geometri kelas sejati dalam keluarga ini — **sendirian hanya
menjelaskan ~0,155** dari R² 0,497. Sisanya dari `conf_mean`, `leak_max`, `leak_top5`, `margin_mean`:
ringkasan distribusi skor.

Dan lebih dalam: **φ ruang-output dan δ_y keduanya diturunkan dari matriks skor model yang SAMA**,
hanya dari sampel terpisah. `conf_mean` pada DESC dan `q̂_y` pada CAL adalah dua estimasi parameter
distribusi yang sama. Memprediksi satu dari yang lain nyaris dijamin berhasil bila kelasnya punya
distribusi skor yang stabil sama sekali. Jadi gate B di sini **lebih dekat ke estimasi distribusional
daripada ekstrapolasi geometrik**.

§4 dokumen ini sudah menyatakan "hasil ini tidak otomatis berpindah ke deskriptor embedding", tetapi
kalimat itu terlalu lunak. Yang tepat: **klaim geometri embedding tidak tertegakkan oleh run ini**, dan
langkah berikutnya (φ embedding lewat encoder SimCLRv2 publik) adalah yang menutup jaraknya.

### Tiga cacat pelaporan, diperbaiki

1. **Teks caveat SALAH.** Ia mengklaim "ablasi prevalensi HAMPA, gate C menguji HANYA lawan jarak",
   sementara run melaporkan `prevalence_ablation_degenerate: false` dan **kedua** ablasi diuji
   (log-range prevalensi 0,594). Caveat itu string yang ditulis di muka, bukan diturunkan dari run —
   persis kelas klaim basi yang menyesatkan reviewer. Caveat kini **diturunkan dari hasil**.
2. **Baris α = 0,01 tidak bermakna** dan dilaporkan berdampingan seolah setara. `feasible_classes: 0`:
   kuantil conformal butuh ≥99 sampel/kelas, `n_cal = 25` hanya 25, jadi "kuantil ke-99 dari 25 titik"
   adalah maksimumnya. §8.8 terpenuhi di **α = 0,10 dan 0,05** (butuh ≥19), **tidak** di 0,01 — tetap
   kemajuan besar atas Pl@ntNet yang hanya sanggup 0,10. Baris tak-bermakna kini ditandai.
3. **Jalur dump dicatat relatif** (`../../../ccc_npy/...`), tidak reproducible. Kini absolut.

### Item terbuka

Reproduksi **Clustered CP** gagal impor: `cannot import name 'clustered_conformal' from
'utils.clustering_utils'`. Nama fungsinya berbeda dari yang kuasumsikan. Bukan pemblokir gerbang, tapi
verifikasi-vs-angka-terbit **belum berjalan**, jadi kesetiaan setup belum terbukti.

### KEPUTUSAN PHASE 1

Gate A, B, C dan §6.4 semuanya lulus pada uji primer yang di-pre-deklarasi, dengan §3.3 terpenuhi dan
koreksi multiplisitas diterapkan. **Phase 1 dicatat LULUS untuk keluarga deskriptor ruang-output pada
ImageNet.** §10 tidak lagi memblokir `pcc/method/`.

Cakupan yang harus melekat pada setiap klaim turunannya: **hasil ini menegakkan bahwa δ_y dapat
diprediksi dari deskriptor tingkat-kelas ketika jumlah kelas memadai — bukan bahwa geometri embedding
adalah sumber prediktabilitas itu.** Yang kedua menunggu run φ embedding.

---

## UJI NON-SIRKULAR (2026-08-10) — φ bobot kepala LULUS gate B dan C

Ini menjawab keberatan yang dicatat di bagian sebelumnya sebagai pembatas utama, dan menjawabnya
dengan uji, bukan dengan argumen.

φ dari `fc.weight` ResNet-50 torchvision (`IMAGENET1K_V2`), 9 fitur geometri keputusan +
`log_prevalence`. Skor CCC berasal dari **SimCLRv2 + linear probe**; kepala ini **ResNet-50
tersupervisi**. Dua model berbeda, jadi φ **eksogen** terhadap δ_y.

| | φ ruang-output | φ bobot kepala |
|---|---|---|
| eksogen terhadap δ_y | **tidak** | **ya** |
| derau sampling | ada | **nol** (parameter, bukan estimasi) |
| butuh sampel kelas | ya | **tidak** |
| gate B R² | +0,4975 [+0,461; +0,534] | **+0,3880** [+0,345; +0,439] |
| ter-normalisasi | 0,600 (`r_δ·r_φ` = 0,770) | **0,468** (plafon = `r_δ` = 0,829) |
| gate C vs jarak | +0,3426, p ≤ 0,001, **84 σ** | **+0,3753, p ≤ 0,001, 83 σ** |
| gate C vs prevalensi | +0,4902, p ≤ 0,001, **102 σ** | **+0,3710, p ≤ 0,001, 73 σ** |
| verdict | LULUS | **LULUS** |

`r_φ` untuk keluarga kepala **tidak** diambil dari screen stabilitas: fitur `w_y` mendapat 1,0 secara
konstruksi karena tidak bergantung pembagian data sama sekali. Itu tautologi, bukan mutu, jadi
plafonnya `r_δ` saja. Menggabungkannya akan membengkakkan plafon secara palsu dan membuat R²
ter-normalisasi terlihat lebih buruk dari seharusnya.

### Struktur internalnya BERKEBALIKAN, dan itu poinnya

Selisih berpasangan gate C adalah `R²(full) − R²(ablasi)`, jadi R² ablasi dapat dibaca balik:

```
ruang-output:  0,4975 − 0,3426  ->  prof_knn_1 sendirian  ~ 0,155
bobot kepala:  0,3880 − 0,3753  ->  w_cos_knn_1 sendirian ~ 0,013
```

Di keluarga ruang-output, sebagian besar prediktabilitas datang dari **ringkasan skor langsung**
(`conf_mean`, `leak_*`, `margin_mean`) — itulah sebabnya ia tidak dapat menegakkan klaim geometri. Di
keluarga bobot kepala, jarak-tetangga-terdekat tunggal menjelaskan **nyaris nol**, dan yang menjelaskan
adalah **geometri multi-fitur**: norma bobot, jarak ke rerata, sebaran ketetanggaan, margin ke rival
terdekat, kNN multi-skala.

Jadi kedua keluarga lolos, tetapi **karena alasan yang berbeda**, dan hanya yang kedua konsisten
dengan klaim geometrik.

### Temuan yang lebih kuat dari yang diantisipasi

φ berasal dari model yang **tidak pernah melihat** skor yang mendefinisikan δ_y, dan tetap
memprediksinya pada 73–83 simpangan baku dari null permutasi. Implikasinya melampaui metode ini:

> Struktur tingkat-kelas yang menentukan threshold conformal **bukan milik jaringan tertentu** — ia
> properti ruang label, yaitu keterkacauan visual antar kelas. Kelas yang mudah tertukar mudah
> tertukar bagi kedua model, dan itulah yang membuat geometri satu model memprediksi δ_y model lain.

Itu klaim yang jauh lebih kuat, dan lebih berguna, daripada "geometri model ini memprediksi threshold
model ini sendiri". Ia juga memberi jalan deployment: δ̂ dapat dihitung dari kepala klasifier **apa pun**
yang tersedia, tanpa citra dan tanpa label kelas sasaran.

### Yang MASIH belum tertegakkan

1. **§6.4 belum dijalankan untuk keluarga kepala.** Gate B/C menunjukkan δ_y terprediksi; belum
   ditunjukkan bahwa δ̂ dari `w_y` **membeli ekuitas coverage**. Itu uji outcome-nya, dan ia yang
   berikutnya.
2. **ImageNet berimbang.** Klaim aplikasi ekor-panjang tetap butuh Pl@ntNet/iNat — dan di sana daya
   ujinya tidak memadai (90 dan 63 kelas layak). Ketegangan itu belum terselesaikan.
3. **Ketidakcocokan model perlu dijawab di paper, bukan disembunyikan.** Reviewer akan bertanya
   mengapa geometri ResNet-50 memprediksi threshold SimCLR+probe. Jawaban yang didukung data ada di
   atas, tetapi ia klaim substantif dan harus dinyatakan sebagai klaim, bukan diselipkan.
4. **Baseline belum dijalankan.** API-nya sudah didaftar (lihat `release_audit.md`), belum dipanggil.

---

## AMANDEMEN 11 — pra-deklarasi §6.4 keluarga kepala + kesetiaan baseline (2026-08-10)

Ditulis **sebelum** run apa pun. Dua sel baru di notebook 05 (12c dan 12d), tujuh jalur logika
sudah diuji-kering sintetik.

### 11a. §6.4 untuk φ kepala — apa yang dihitung sebagai lulus

Desainnya **identik** dengan §6.4 ruang-output (Amandemen 8), hanya δ̂ yang berganti sumber:
ekuitas worst-class pada ukuran set tercocokkan, λ dipilih pada kelas TRAIN saja, 20 belahan kelas.
Tidak ada satu pun parameter yang disetel ulang untuk keluarga kepala — kalau disetel, hasilnya
bukan uji lagi.

| verdict | syarat |
|---|---|
| **LULUS** | `observed.ci_low > 0` **dan** `observed.ci_low > shuffled_null.ci_high` |
| **TIDAK POSITIF** | selain itu |
| **TIDAK TERBACA** | `oracle_ceiling ≤ 0.02` — tanpa ruang oracle metriknya tidak bisa positif, jadi angkanya tidak boleh dibaca sebagai bukti apa pun (pelajaran desain 4 yang lama) |

Empat kontrol dilaporkan bersamaan, bukan opsional: `observed`, `shuffled_null`, `oracle_ceiling`,
`raw_delta_lambda1`. Tiga terakhir yang membuat yang pertama bisa ditafsirkan.

**Prediksi, dicatat supaya bisa salah.** Kuperkirakan §6.4 kepala **positif tetapi lebih kecil**
dari ruang-output (+0,1173), kira-kira **+0,04 sampai +0,09**, dengan fraksi ruang oracle terpakai
**lebih rendah** dari 76%. Alasannya: R² kepala 0,3880 < 0,4975, dan δ̂ yang lebih berderau
menerjemah jadi koreksi yang lebih konservatif setelah penyusutan. Kalau ia justru **melampaui**
ruang-output, prediksiku salah dan itu yang dicatat — bukan alasannya yang direvisi.

**Konsekuensi yang sudah ditetapkan:**

- **Lulus** → tulang belakang paper lengkap dengan deskriptor eksogen: δ_y terprediksi **dan**
  koreksinya membeli ekuitas, keduanya dari geometri yang tidak menyentuh skor. `pcc/method/`
  dibangun di atas keluarga kepala.
- **Tidak positif** → klaimnya turun pangkat menjadi **prediktabilitas tanpa manfaat terbukti**,
  dan itu yang ditulis. Ruang-output tidak boleh dipakai sebagai gantinya untuk klaim geometri,
  karena sirkularitasnya sudah terukur (0,155 dari 0,497 dari satu fitur jarak).
- **Tidak terbaca** → dilaporkan sebagai desain yang gagal memberi ruang, bukan sebagai kegagalan
  metode. Desain diperbaiki, lalu dijalankan ulang — sekali, dengan perbaikan dinyatakan.

### 11b. Baseline — dipanggil, dan kesetiaannya dinyatakan di muka

`standard_conformal`, `classwise_conformal`, `clustered_conformal` dipanggil pada **array skor yang
sama** dengan yang dipakai semua gerbang di atas. Nilai kembaliannya **belum diketahui**, jadi
setiap fungsi dipanggil lalu bentuk kembaliannya dicetak sebelum metrik dihitung. Menebak signature
sudah dua kali salah di sesi ini (`gdown.download_folder`, modul `clustered_conformal`); polanya
tidak diulang. Tiap panggilan punya penjaga sendiri, jadi satu kegagalan tidak menghapus yang lain.

**Kesetiaan yang harus melekat pada angkanya, dan sekarang dicetak oleh selnya sendiri:** skor kita
adalah subsample terstratifikasi (`MAX_ROWS = 250_000` dari 1.153.051 baris), lalu 30% darinya jadi
CAL. Jadi reproduksi ini **tidak akan sama persis** dengan angka terbit CCC. Ia sahih untuk
perbandingan **internal** (metode kita vs baseline, array identik); ia **belum** sahih sebagai klaim
"kami mereproduksi angka mereka". Untuk tabel paper, `MAX_ROWS = None`.

Konsekuensinya untuk sitiran: kalau reproduksi pada dump penuh cocok dengan angka terbit, setup-nya
bisa disitir langsung dan baseline kita kredibel. Kalau tidak cocok, **selisihnya dilaporkan** dan
angka kita berdiri sebagai reimplementasi, bukan sebagai angka mereka.

---

## HASIL §6.4 KELUARGA KEPALA (2026-08-11) — LULUS, dan prediksi Amandemen 11 TEPAT

| besaran | ruang-output | **kepala `w_y`** |
|---|---|---|
| gate B R² | +0,4975 [+0,461, +0,534] | **+0,3880** [+0,345, +0,439] |
| §6.4 observed | +0,1173 [+0,0948, +0,1398] | **+0,0643** [+0,0536, +0,0751] |
| null teracak | −0,0490 | −0,0630 |
| plafon oracle | +0,1542 | +0,1542 (identik — plafonnya sifat data) |
| ruang oracle terpakai | 76% | **42%** |
| δ̂ mentah, λ=1 | −0,2343 | **−0,5079** |
| λ dari TRAIN | 0,088 | 0,095 |

**Prediksi Amandemen 11 diverifikasi.** Kuprediksi **+0,04 sampai +0,09** dengan fraksi ruang oracle
**di bawah 76%**; hasilnya +0,0643 dan 42%. Kedua sisi tepat, dan mekanisme yang kuajukan juga
terkonfirmasi lewat kolom `raw_delta_lambda1`: δ̂ kepala yang lebih berderau **merugikan dua kali
lebih dalam** sebelum disusutkan (−0,508 vs −0,234), persis alasan mengapa manfaat akhirnya lebih
kecil. Ini prediksi pertama di proyek ini yang benar setelah dua yang salah (ramalan §6.4 sintetik,
dan Amandemen 9); dicatat sebagai satu prediksi tepat, bukan sebagai pola.

### Apa yang sekarang tertegakkan

Untuk deskriptor yang **eksogen** — parameter dari model LAIN, tanpa satu pun sampel berlabel, tanpa
citra, tanpa GPU — ketiga mata rantai lengkap pada dataset dengan daya uji memadai (1000 kelas):

1. δ_y **terprediksi** dari `w_y`: R² +0,3880, CI kelas jauh di atas nol;
2. prediksinya **bukan** dari prediktor trivial: gate C lolos lawan jarak (+0,3753, 83σ) **dan**
   prevalensi (+0,3710, 73σ), Holm-terkoreksi, p ≤ 0,001 keduanya;
3. koreksinya **membeli ekuitas**: worst-class +0,0643 pada ukuran set tercocokkan, di atas nol dan
   di atas null teracak, memakai 42% ruang yang tersedia bagi oracle.

Itu tulang belakang klaim ekstrapolasi. `pcc/method/` dibangun di atas keluarga kepala.

### Yang tetap harus dinyatakan sebagai batas, bukan diselipkan

- **42%, bukan 76%.** Deskriptor eksogen membeli lebih sedikit daripada deskriptor sirkular. Itu
  harga kejujurannya, dan angkanya harus muncul di paper berdampingan, bukan hanya yang besar.
- **δ̂ mentah merugikan berat** (−0,508). Penyusutan bukan hiasan; tanpa λ dari TRAIN, metode ini
  merusak worst-class. Aturan pemilihan λ adalah bagian dari metode, bukan detail implementasi.
- **ImageNet berimbang.** Klaim ekor-panjang belum terselesaikan (lihat item terbuka di atas).

---

## DUA BUG BASELINE (2026-08-11) — keduanya kesalahan yang sama, diulang

Baseline gagal: `ModuleNotFoundError: No module named 'utils'`, padahal clone-nya sukses.

1. **Urutan salah.** Sel baseline (12d) berjalan **sebelum** sel yang meng-clone repo (sel 13). Sel
   12d memasang `/content/ccc` ke `sys.path` tetapi tidak pernah meng-clone apa pun. Konsumen
   ditempatkan di depan penyedianya.
2. **Jalur impor diasumsikan.** Bahkan di sel 13, clone sukses lalu `import utils.conformal_utils`
   gagal — jadi `utils/` **tidak** berada di tempat yang dicatat `release_audit.md`. Aku menebak
   lagi, di kode yang tidak bisa kujalankan sendiri.

Ini pola yang sama dengan `gdown.download_folder` dan modul `clustered_conformal`: **mengasumsikan,
bukan memeriksa.** Perbaikannya struktural, bukan tambal: `ensure_ccc()` meng-clone sekali lalu
**mencari** `conformal_utils.py` di dalam repo dan memasang induk direktorinya ke `sys.path`, lalu
melaporkan nama paket dari temuan itu. Kedua sel memakai satu fungsi ini, dan `ensure_ccc` kini
berada sebelum keduanya. Kalau berkasnya tidak ada, isi repo dicetak — supaya kesalahannya bisa
dibetulkan sekali, bukan ditebak lagi. Signature tiap fungsi juga dicetak sebelum dipanggil, jadi
kalau daftar di `release_audit.md` keliru, itu terlihat, bukan tersembunyi di balik `TypeError`.

Diuji-kering dengan layout yang **sengaja berbeda** dari yang tercatat (`code/srcutils/`, paket
bukan bernama `utils`): penemuannya berhasil, baseline terpanggil, sel penemuan API ikut memakai
paket yang sama. Lima jalur, termasuk berkas hilang dan repo absen.

### Satu koreksi pada `release_audit.md`

Daftar API di sana mengklaim dibaca oleh `notebooks/05` lewat `vars(module)`. **Tidak.** Sel itu
belum pernah berhasil mengimpor apa pun. Daftarnya berasal dari clone audit 2026-07-24 ke `refs/`
(kini tidak ada di mesin). Sumbernya sah, atribusinya salah — dan atribusi yang salah itu yang
membuatku memperlakukan jalur `utils.` sebagai fakta terverifikasi padahal bukan.
