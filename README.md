# 🚗 SPARX – Sistem Smart Parking dengan Keamanan Multi-Variabel

**SPARX** adalah solusi smart parking cerdas berbasis _computer vision_ yang menggabungkan:

- Deteksi **wajah pengemudi**
- Deteksi **plat nomor**
- Deteksi **jenis kendaraan**
- Deteksi **warna dominan kendaraan**

Semua fitur ini diproses dalam satu <a href="https://en.wikipedia.org/wiki/Pipeline_(computing)" target="_blank" rel="noopener noreferrer">pipeline</a> terstruktur.

Dibangun dengan <a href="https://en.wikipedia.org/wiki/You_Only_Look_Once" target="_blank" rel="noopener noreferrer">YOLO</a> dan teknik <a href="https://en.wikipedia.org/wiki/Computer_vision" target="_blank" rel="noopener noreferrer">computer vision</a> modern, sistem ini dirancang untuk:

- Mendukung **manajemen parkir otomatis**
- Meningkatkan **pengawasan keamanan**
- Memungkinkan **integrasi mudah** dengan aplikasi backend seperti <a href="https://flask.palletsprojects.com/" target="_blank" rel="noopener noreferrer">Python/Flask</a> maupun dashboard web.

---

## 🎯 Fitur Utama

### 🔍 1. Deteksi Wajah

Menggunakan model YOLO kustom untuk mendeteksi wajah pengemudi saat masuk atau keluar area parkir.

### 🔢 2. Deteksi Plat Nomor

Membaca dan mengenali plat nomor menggunakan _license-plate model_ khusus.

### 🚘 3. Deteksi Jenis Kendaraan

Mengidentifikasi jenis kendaraan, seperti:

- Mobil
- Motor
- Kendaraan darurat
- Non-kendaraan

### 🎨 4. Deteksi Warna Dominan Kendaraan

Mengambil warna paling dominan dari citra kendaraan sebagai variabel tambahan keamanan.

---

## 📦 Model

Folder data/:
| Model | Fungsi |
| --------------- | ----------------------- |
| `wajah.pt` | Deteksi wajah |
| `plat_nomor.pt` | Deteksi plat nomor |
| `kendaraan.pt` | Deteksi jenis kendaraan |
| `yolov8n.pt` | Backbone dasar YOLO |

---

## 📜 Lisensi

Proyek SPARX dirilis menggunakan lisensi **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**.

## Anda bebas untuk:

- Menggunakan, menyalin, dan membagikan kode ini.
- Memodifikasi atau mengembangkan fitur baru dari kode ini.

## Dengan syarat:

1. Memberikan atribusi kepada pembuat asli (**ndeso17**).
2. Tidak digunakan untuk keperluan komersial, termasuk menjual, menyewakan, atau memasukkan kode ini ke produk/layanan berbayar.

Lisensi lengkap dapat dibaca di:  
[https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)
