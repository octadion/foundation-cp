# Jaminan yang dimiliki PCC, dan yang batal di bawah pergeseran

**Ditulis 2026-08-17**, setelah ImageNet-C masuk dan sebelum tabelnya ditulis ke paper.
Menutup audit **F2** (bukti marginal coverage terjaga) dan **F4** (exchangeability di bawah
pergeseran distribusi). Audit menyebut F4 "tidak bisa dilewati", dan alasannya konkret:
begitu kalibrasi dan evaluasi datang dari distribusi berbeda, jaminan konformal batal untuk
**setiap** metode di tabel itu — bukan hanya milik kita — dan itu harus dijawab dengan
literatur, bukan didiamkan.

---

## 1. Notasi

Data kalibrasi $(X_i, Y_i)_{i=1}^{n}$, titik uji $(X_{n+1}, Y_{n+1})$, skor
nonkonformitas $s(x, y)$ dengan konvensi **lebih besar = lebih tidak konform**. Untuk
THR/LAC, $s(x,y) = 1 - \hat p(y \mid x)$.

- $\hat q$ — kuantil konformal marginal: $\lceil (n+1)(1-\alpha) \rceil / n$ kuantil empiris
  dari $\{s(X_i, Y_i)\}$.
- $\delta_y$ — koreksi tingkat kelas; ambang kelas $y$ adalah $\hat q + \delta_y$.
- $\hat\delta_y = \lambda \cdot g_\theta(\varphi(y))$ — koreksi **terprediksi** dari
  deskriptor geometris kelas, dengan $\lambda$ dipilih **hanya di ruang label TRAIN**.
- $c$ — satu skalar offset marginal, dicocokkan ulang setelah $\hat\delta$ terpasang.

Himpunan prediksi PCC:

$$\mathcal{C}(x) = \{\, y : s(x, y) \le \hat q + \hat\delta_y + c \,\}$$

---

## 2. Proposisi (F2): cakupan marginal terjaga, dan hanya itu

> **Proposisi 1.** Misalkan $(X_i, Y_i)_{i=1}^{n+1}$ *exchangeable*, dan misalkan
> $\hat\delta_\cdot$ **tidak bergantung** pada baris kalibrasi yang dipakai memilih $c$.
> Pilih $c$ sebagai kuantil konformal $\lceil (n'+1)(1-\alpha) \rceil / n'$ dari skor
> tergeser $\{\, s(X_i, Y_i) - \hat\delta_{Y_i} \,\}$ atas $n'$ baris tersebut. Maka
>
> $$\mathbb{P}\big( Y_{n+1} \in \mathcal{C}(X_{n+1}) \big) \;\ge\; 1 - \alpha .$$

**Bukti.** Definisikan skor termodifikasi $\tilde s(x, y) = s(x, y) - \hat\delta_y$. Karena
$\hat\delta_\cdot$ adalah fungsi dari $\varphi(\cdot)$ saja — dan $\varphi$ dihitung dari
bobot kepala model atau dari irisan DESC yang terpisah, **tidak** dari baris kalibrasi ini —
maka $\tilde s$ adalah fungsi skor yang **tetap** (fixed) terhadap sampel tersebut.

Exchangeability tertutup terhadap penerapan fungsi tetap per titik. Jadi
$\tilde s(X_1,Y_1), \dots, \tilde s(X_{n+1},Y_{n+1})$ juga exchangeable, dan argumen split
conformal baku berlaku apa adanya atas $\tilde s$: dengan $c$ sebagai kuantil konformal
tersebut,

$$\mathbb{P}\big( \tilde s(X_{n+1}, Y_{n+1}) \le c \big) \ge 1 - \alpha .$$

Dan $\tilde s(x,y) \le c \iff s(x,y) \le \hat q + \hat\delta_y + c'$ untuk pergeseran
konstan yang sesuai, yaitu tepat keanggotaan $\mathcal{C}(x)$. $\blacksquare$

**Yang membuat buktinya sah adalah satu derajat kebebasan.** $\hat\delta_y$ boleh serumit
apa pun; ia hanya mendefinisikan ulang fungsi skornya. Yang dikalibrasi tetap **satu**
skalar, atas irisan yang $\hat\delta$ tidak pernah lihat. Itulah sebabnya seleksi $\lambda$
dibatasi ke ruang label TRAIN oleh `restrict_to_classes` — dan sebabnya kebocoran yang
ditemukan lebih awal di proyek ini (λ dipilih atas semua kelas, menggeser 0,3 → 0,5)
bukan cacat kosmetik: ia melanggar prasyarat proposisi ini.

### 2.1 Yang TIDAK dijamin — dinyatakan sebelum ditanya

1. **Tidak ada jaminan class-conditional.** Untuk kelas dengan $n_y = 0$ tidak ada
   informasi berlabel apa pun tentang kelas itu, jadi tidak ada metode — termasuk PCC —
   yang bisa membawa jaminan cakupan hingga-sampel bersyarat kelas di sana. Yang diklaim
   PCC murni **empiris**: ekuitas worst-class yang terukur, pada ukuran set tercocokkan.
2. **$\lambda$ dan $n^\star$ dipilih dari data.** Keduanya dipilih di ruang label TRAIN.
   Itu menjaga Proposisi 1, tetapi berarti angka pada kelas TRAIN bersifat optimistis;
   karena itu Tabel 1 dan Tabel 2 tidak pernah digabung.
3. **Plafon oracle bukan metode.** Ia memakai label EVAL dua kali — untuk $\delta$ dan
   untuk $\lambda$ — jadi tak tercapai secara konstruksi. Ia ada untuk memberi skala, dan
   sejak 2026-08-17 ia **disusutkan**, karena versi tak-disusutkannya adalah $\lambda = 1$
   dan $\lambda = 1$ terukur $-0{,}58$: sebuah "plafon" yang bisa dilewati metodenya bukan
   plafon.

---

## 3. F4: exchangeability batal di ImageNet-C, dan itu berlaku untuk semua

Di fase ImageNet-C, kalibrasi memakai citra **bersih** dan evaluasi memakai citra
**terkorupsi**. Maka $(X_i, Y_i)_{i \le n}$ dan $(X_{n+1}, Y_{n+1})$ **tidak** exchangeable:
mereka berasal dari dua distribusi berbeda, $P_{\text{clean}}$ dan $P_{\text{corrupt}}$.

Prasyarat Proposisi 1 gagal. **Tidak ada jaminan cakupan yang berlaku di sana** — bukan
untuk PCC, bukan untuk classwise CP, bukan untuk clustered CP, bukan untuk fuzzy classwise
CP, bukan untuk split conformal biasa. Ini sifat setting-nya, bukan sifat metodenya.

### 3.1 Ini bukan hipotesis — terukur

Fase ImageNet-C melaporkan `frac_empty_sets`, dan angkanya menunjukkan kegagalannya
langsung: ambang yang dikalibrasi di citra bersih menghasilkan **himpunan kosong** untuk
sebagian besar baris terkorupsi, dan ukuran set rata-rata jatuh **di bawah 1,0**. Prediktor
bukan memberi himpunan yang sempit — ia tidak memberi apa pun. Cakupan dan ukuran set
keduanya hanya tampak "rendah"; tingkat himpunan kosong itulah yang menyebut namanya.

Metrik itu ditambahkan **karena** fase ini, bukan sebelumnya, dan ia harus dilaporkan di
paper: ia bukti kuantitatif bahwa jaminannya batal, bukan sekadar pernyataan bahwa ia batal.

### 3.2 Apa yang literatur tawarkan, dan mengapa kami tidak memakainya

Dua garis kerja memulihkan jaminan di bawah pergeseran, dan keduanya **butuh sesuatu yang
tidak kami punya**:

**Weighted conformal prediction** (Tibshirani, Barber, Candès, Ramdas, NeurIPS 2019).
Di bawah *covariate shift* $P_{\text{test}}(x) \ne P_{\text{cal}}(x)$ dengan
$P(y \mid x)$ tak berubah, cakupan dipulihkan dengan membobot skor kalibrasi memakai rasio
kepadatan $w(x) = \mathrm{d}P_{\text{test}} / \mathrm{d}P_{\text{cal}}(x)$.

Kenapa tidak dipakai di sini: korupsi ImageNet-C bukan covariate shift dengan
$P(y\mid x)$ tetap — korupsi mengubah citra sedemikian sehingga hubungan label-ke-citra
sendiri merosot (akurasi top-1 jatuh). Dan $w(x)$ harus ditaksir; menaksirnya di ruang
citra 224×224 adalah masalah yang lebih sulit daripada tugas aslinya. Memakainya berarti
memasukkan taksiran rasio kepadatan yang tak terverifikasi ke jalur kritis, lalu melaporkan
"jaminan" yang bergantung padanya.

**Adaptive conformal inference** (Gibbs & Candès, NeurIPS 2021, dan lanjutannya).
Memperbarui $\alpha_t$ secara daring dari kesalahan cakupan yang teramati, dan memberi
jaminan cakupan **jangka panjang** tanpa asumsi distribusi apa pun.

Kenapa tidak dipakai di sini: ia butuh **umpan balik label secara daring** — cakupan
teramati pada titik uji. Setting kami satu-tembak dan tanpa label uji: itu justru
premisnya. Metode daring akan menjawab pertanyaan yang berbeda.

### 3.3 Jadi apa yang dilaporkan

ImageNet-C dilaporkan sebagai **uji ketahanan (stress test)**, bukan sebagai bukti
jaminan. Yang ditanya bukan "apakah cakupannya 1−α di sana" — jawabannya tidak, untuk
semua metode — melainkan:

> Ketika exchangeability batal dan setiap metode kehilangan jaminannya, apakah koreksi
> tingkat kelas yang diekstrapolasi dari geometri **masih** memperbaiki ekuitas worst-class
> relatif terhadap ambang marginal, pada ukuran set tercocokkan?

Itu pertanyaan yang terjawab, dan perbandingannya sah karena **kedua lengan kehilangan
jaminannya dengan cara yang sama** — keduanya dikalibrasi pada irisan bersih yang sama dan
dievaluasi pada baris terkorupsi yang sama.

**Kalimat yang harus ada di paper:**

> Kami mengevaluasi pada 15 korupsi ImageNet-C, severity 3 dan 5, tiga seed, dengan
> kalibrasi pada citra bersih dan evaluasi pada citra terkorupsi. Karena kedua irisan
> berasal dari distribusi berbeda, exchangeability tidak berlaku dan jaminan cakupan
> konformal batal di sini **untuk setiap metode yang dibandingkan, termasuk split conformal
> biasa** — kami melaporkan bagian himpunan kosong sebagai bukti langsungnya. Memulihkan
> jaminan akan menuntut rasio kepadatan (weighted CP, Tibshirani dkk. 2019) atau umpan balik
> label daring (adaptive CP, Gibbs & Candès 2021), dan keduanya tidak tersedia dalam setting
> satu-tembak tanpa label uji ini. Kami karena itu membacanya sebagai uji ketahanan: pada
> ukuran set tercocokkan, apakah koreksi terprediksi masih memperbaiki ekuitas ketika
> setiap metode sama-sama kehilangan jaminannya.

---

## 4. Ringkasan

| Klaim | Status |
|---|---|
| Cakupan marginal $\ge 1-\alpha$ di bawah exchangeability | **Terbukti**, Proposisi 1 |
| Prasyaratnya ditegakkan oleh kode | `restrict_to_classes` pada seleksi λ; offset dicocokkan hanya pada baris kelas TRAIN |
| Jaminan class-conditional pada $n_y = 0$ | **Tidak ada**, dan tidak mungkin ada untuk metode apa pun |
| Cakupan di bawah pergeseran (ImageNet-C) | **Tidak dijamin**, untuk semua metode; dilaporkan sebagai uji ketahanan dengan bukti himpunan kosong |
| Pemulihan lewat weighted CP | Mungkin secara prinsip, butuh rasio kepadatan yang tak kami punya |
| Pemulihan lewat adaptive CP | Mungkin secara prinsip, butuh umpan balik label daring yang tak ada dalam setting ini |
