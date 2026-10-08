# Klasifikasi Citra Bunga dengan CNN

> **Tugas 2 — Machine Learning Day 2** | Penulis: **Gramandha Wega Intyanto**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-web%20app-lightgrey?logo=flask)
![Status](https://img.shields.io/badge/status-selesai-brightgreen)

Proyek *machine learning* untuk mengklasifikasikan foto bunga ke dalam lima kelas: **daisy**, **dandelion**, **rose**, **sunflower**, dan **tulip**. Proyek mencakup notebook pelatihan dan evaluasi dua arsitektur CNN, model hasil pelatihan, serta aplikasi web Flask untuk mencoba prediksi secara interaktif.

Publikasi Ilmiah
[Intyanto, Gramandha Wega. "Klasifikasi citra bunga dengan menggunakan deep learning: CNN (Convolution Neural Network)." Jurnal Arus Elektro Indonesia 7.3 (2021): 80-83.](https://scholar.google.com/citations?view_op=view_citation&hl=id&user=vrmh4RoAAAAJ&citation_for_view=vrmh4RoAAAAJ:-_dYPAW6P2MC)

---

## Daftar Isi

- [Fitur](#fitur)
- [Isi Folder](#isi-folder)
- [Arsitektur Model](#arsitektur-model)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Menyiapkan & Menjalankan Aplikasi Web](#menyiapkan--menjalankan-aplikasi-web)
- [Cara Menggunakan Aplikasi](#cara-menggunakan-aplikasi)
- [Menjalankan Notebook Pelatihan](#menjalankan-notebook-pelatihan)
- [Troubleshooting](#troubleshooting)
- [Catatan Teknis](#catatan-teknis)

---

## Fitur

- Melatih dan membandingkan **CNN sederhana** dengan **VGG16** (*transfer learning*) melalui notebook Jupyter.
- Antarmuka web dengan fitur *drag-and-drop* untuk mengunggah foto bunga.
- Mendukung format gambar **JPG, JPEG, PNG**, dan **WEBP** dengan ukuran maksimal **10 MB**.
- Menampilkan kelas bunga hasil prediksi secara langsung di halaman web.
- Validasi file dan pengamanan nama unggahan sebelum disimpan.

---

## Isi Folder

| File / Folder | Keterangan |
| --- | --- |
| `app.py` | Aplikasi Flask — memuat `model_cnn_bunga.h5`, memproses unggahan, melakukan prediksi, dan menyajikan halaman web. |
| `model_train.ipynb` | Notebook pelatihan — eksplorasi dataset, pelatihan dan evaluasi CNN sederhana serta VGG16, penyimpanan model. |
| `model_cnn_bunga.h5` | Model CNN sederhana (~40 MB) yang digunakan oleh aplikasi web. |
| `model_vgg_bunga.h5` | Model VGG16 hasil *transfer learning* (~80 MB) untuk perbandingan; **tidak** digunakan oleh `app.py`. |
| `data_bunga.zip` | Dataset *Flowers Recognition* — 4.317 gambar dalam lima folder kelas di bawah `flowers/`, bersumber dari [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). |
| `requirements.txt` | Dependensi aplikasi Flask: `flask`, `tensorflow-cpu`, `gunicorn`, `pillow`, `numpy`. |
| `templates/index.html` | Antarmuka web *Flora* (template Jinja2). |
| `static/uploads/` | Folder penyimpanan gambar contoh dan gambar yang diunggah saat aplikasi berjalan. |
| `.venv/` | Virtual environment lokal — bukan source code proyek. |

---

## Arsitektur Model

Notebook membandingkan dua arsitektur:

### 1. CNN Sederhana

Empat blok konvolusi + max-pooling dengan filter bertahap (32 → 64 → 128 → 128), diikuti:

```
Flatten → Dense(512, relu) → Dropout(0.5) → Dense(5, softmax)
```

### 2. VGG16 (*Transfer Learning*)

Backbone VGG16 (bobot ImageNet, lapisan dasar dibekukan) sebagai *feature extractor*, diikuti:

```
Flatten → Dense(256, relu) → Dropout(0.5) → Dense(5, softmax)
```

**Parameter umum pelatihan:**

| Parameter | Nilai |
| --- | --- |
| Ukuran input | 150 × 150 piksel (RGB) |
| Pembagian data | 80% latih / 20% validasi |
| Augmentasi | Flip, rotasi, zoom (pada data latih) |
| Epoch | 20 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |

---

## Hasil Evaluasi

> Hasil berikut bersumber dari notebook `model_train.ipynb` pada akhir epoch ke-20.

| Model | Akurasi Latih | Akurasi Validasi |
| --- | ---: | ---: |
| CNN Sederhana | ~85% | ~75% |
| VGG16 (*Transfer Learning*) | ~95% | ~88% |

> **Catatan:** Hasil aktual dapat sedikit berbeda tiap sesi pelatihan karena inisialisasi bobot dan augmentasi data yang acak. Lihat output notebook untuk metrik lengkap dan kurva pelatihan.

---

## Menyiapkan & Menjalankan Aplikasi Web

Gunakan **Python 3.10** (kompatibel dengan TensorFlow 2.x) dan buat environment baru dari direktori proyek.

### PowerShell (Windows)

```powershell
# 1. Buat dan aktifkan virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip dan install dependensi
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Jalankan aplikasi
python app.py
```

Buka **[http://127.0.0.1:5000](http://127.0.0.1:5000)** di browser. Tekan `Ctrl+C` di terminal untuk menghentikan server.

> **Penting:** Pastikan `model_cnn_bunga.h5` berada di direktori yang sama dengan `app.py`. Model dimuat saat proses Python dimulai. Jika model tidak dapat dimuat, halaman tetap dapat dibuka tetapi tombol prediksi akan dinonaktifkan.

---

## Cara Menggunakan Aplikasi

1. Pilih gambar bunga atau **seret gambar** ke area unggah.
2. Periksa pratinjau gambar yang muncul.
3. Klik tombol **Kenali bunga**.
4. Hasil prediksi ditampilkan di halaman.

Model menerima gambar RGB yang diubah ukurannya menjadi 150 × 150 piksel dan dinormalisasi ke rentang 0–1. Pemetaan indeks kelas keluaran:

| Indeks | Kelas |
| ---: | --- |
| 0 | daisy |
| 1 | dandelion |
| 2 | rose |
| 3 | sunflower |
| 4 | tulip |

---

## Menjalankan Notebook Pelatihan

Notebook dirancang untuk **Google Colab**. Sel pemasangan Google Drive dan lokasi dataset menggunakan path `/content/...`.

### Struktur Dataset

```text
data_bunga.zip
└── flowers/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
```

### Di Google Colab

1. Unggah atau tempatkan `data_bunga.zip` di Google Drive.
2. Ubah variabel `zip_file` di notebook agar menunjuk ke lokasi arsip yang benar.
3. Jalankan semua sel secara berurutan.

### Secara Lokal

Sesuaikan atau lewati sel khusus Google Drive, lalu ganti `zip_file` dan `root_dir` dengan lokasi dataset di komputer. Pasang dependensi notebook terlebih dahulu:

```powershell
pip install jupyter pandas matplotlib seaborn opencv-python scikit-learn
jupyter notebook model_train.ipynb
```

> **Catatan:** `requirements.txt` ditujukan untuk aplikasi web, **bukan** untuk notebook. Notebook memerlukan pustaka tambahan seperti `matplotlib`, `seaborn`, dan `scikit-learn`.

---

## Troubleshooting

### Error `Invalid dtype: tuple` atau `Unrecognized keyword arguments ... quantization_config`

Versi TensorFlow/Keras tidak sesuai dengan format model HDF5. Gunakan environment yang sudah diketahui kompatibel, atau ekspor ulang model dengan versi TensorFlow/Keras yang sama dengan yang digunakan aplikasi.

### Model gagal dimuat, halaman tetap terbuka

`app.py` menangani kegagalan muat model dengan `try/except`. Tombol prediksi dinonaktifkan dan pesan kesalahan ditampilkan. Periksa:
- Apakah file `model_cnn_bunga.h5` ada di folder proyek.
- Apakah versi TensorFlow di environment sesuai.

### Ukuran unggahan melebihi batas

Aplikasi membatasi ukuran file hingga **10 MB**. Untuk mengubah batas ini, edit konstanta `MAX_CONTENT_LENGTH` di `app.py`.

### VGG16 meminta bobot saat notebook dijalankan ulang

Saat notebook dijalankan pertama kali di environment baru, bobot pralatih ImageNet untuk VGG16 perlu diunduh. Pastikan koneksi internet tersedia. Bobot disimpan secara lokal oleh Keras setelah unduhan pertama.

---

## Catatan Teknis

- `app.py` menggunakan **CNN sederhana** (`model_cnn_bunga.h5`), bukan VGG16.
- Ekstensi file diperiksa, gambar divalidasi dengan `PIL`, dan nama file unggahan diamankan dengan `werkzeug.utils.secure_filename` sebelum disimpan.
- Setiap file yang diunggah diberi nama unik berbasis UUID untuk menghindari tabrakan nama.
- File unggahan disimpan di `static/uploads/`. Bersihkan gambar lama secara manual bila diperlukan.
- `app.py` menggunakan server bawaan Flask (`debug=True`) untuk pengembangan. Untuk *deployment* produksi, gunakan server WSGI seperti **Gunicorn** (sudah tercantum di `requirements.txt`) sesuai konfigurasi platform hosting.

---

## Lisensi

Proyek ini menggunakan lisensi **Apache License 2.0**. Lihat file [LICENSE](LICENSE) untuk ketentuan lengkap.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

