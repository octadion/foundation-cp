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

## Ringkasan status klaim

| Klaim | Status |
|---|---|
| δ_y terprediksi dari geometri kelas | **tertegakkan** (nb 05, dua keluarga φ) |
| Koreksinya membeli ekuitas pada `n_y = 0` | **tertegakkan untuk φ kepala di ImageNet, α=0,10** |
| Tidak dibayar oleh kelas ber-data | **tertegakkan** (Tabel 1 juga positif) |
| Berlaku di dataset ekor-panjang | **belum** — keluarga yang berhasil belum diuji di sana |
| Mengalahkan Clustered CP pada ukuran tercocokkan | **belum diuji** |
| Berlaku lintas α | **tidak** pada α=0,05 sejauh ini |
