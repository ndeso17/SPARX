# 📘 CONTRIBUTING.md – Panduan Kontribusi SPARX

Terima kasih telah berminat untuk berkontribusi pada **SPARX – Smart Parking System!**  
Panduan ini membantu Anda memahami cara berkontribusi dengan benar, terstruktur, dan tetap konsisten dengan standar proyek.

---

## 🧱 1. Bagaimana Cara Berkontribusi?

Anda dapat berkontribusi dalam bentuk:

- ✨ Menambahkan fitur baru
- 🐞 Memperbaiki bug
- 🧹 Merapikan kode
- 📚 Memperbaiki dokumentasi
- 🚀 Meningkatkan performa model (YOLO)
- 🧪 Menambahkan test

---

## 🛠 2. Persiapan Lingkungan Pengembangan

### Clone repository

```bash
git clone https://github.com/ndeso17/SPARX.git
cd SPARX
```

### Buat dan aktifkan virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🌿 3. Membuat Branch Baru

Sebelum mulai bekerja, buat branch baru agar perubahan tidak langsung masuk ke `main`.

**Gunakan format:**

| Prefix                         | Deskripsi             |
| ------------------------------ | --------------------- |
| `feature/<nama-fitur>`         | Fitur baru            |
| `fix/<jenis-perbaikan>`        | Perbaikan bug         |
| `docs/<perubahan-dokumentasi>` | Perubahan dokumentasi |
| `refactor/<perubahan-kode>`    | Refactoring kode      |

**Contoh:**

```bash
git checkout -b feature/deteksi-warna-kendaraan
```

---

## 🧩 4. Standar Penulisan Kode

- Gunakan **Python 3.10+**
- Patuhi gaya penulisan **PEP 8**
- Gunakan nama variabel dan fungsi yang jelas
- Jangan memasukkan credential atau file besar yang tidak diperlukan
- Jika menambah modul, jelaskan dalam komentar atau dokumentasi

---

## 📝 5. Commit Message

Gunakan format commit berikut agar mudah dibaca:

| Prefix      | Deskripsi                        |
| ----------- | -------------------------------- |
| `feat:`     | Menambahkan fitur baru           |
| `fix:`      | Perbaikan bug/error              |
| `docs:`     | Update dokumentasi               |
| `refactor:` | Merapikan/mengubah struktur kode |
| `perf:`     | Meningkatkan performa            |
| `test:`     | Menambahkan test                 |

**Contoh:**

```bash
git commit -m "feat: tambah deteksi warna dominan kendaraan"
```

---

## 🔄 6. Push dan Pull Request

### Push ke branch Anda

```bash
git push origin feature/deteksi-warna-kendaraan
```

### Buka Pull Request (PR) di GitHub:

1. Pilih **base:** `main`
2. Pilih **compare:** branch Anda
3. Jelaskan perubahan Anda

### Tunggu review:

- PR yang rapi lebih cepat diterima
- Jika diminta revisi, cukup commit tambahan di branch yang sama

---

## 🧪 7. Testing

Jika menambah fitur baru:

- ✅ Pastikan script berjalan tanpa error
- ✅ Cek kompatibilitas dengan modul lain
- ✅ Sertakan contoh input & output bila perlu

---

## 📦 8. Aturan untuk Dataset

Agar repo tetap ringan:

- ❌ **Tidak boleh** upload dataset berukuran besar ke GitHub
- ✅ Gunakan folder `data/` hanya untuk file kecil seperti:
  - `.pt` model YOLO ringan
  - Contoh input
  - File konfigurasi (`.yaml`)

**Dataset besar wajib ditempatkan di:**

- Google Drive
- HuggingFace
- Kaggle

dan cukup dibagikan melalui link.

---

## 🛡 9. Lisensi & Kepatuhan

Proyek SPARX menggunakan lisensi **CC BY-NC 4.0**:

- Pengguna wajib memberi atribusi
- Dilarang menggunakan proyek ini untuk tujuan komersial
- Kontribusi Anda secara otomatis mengikuti lisensi yang sama

---

## 🙌 10. Terima Kasih!

Terima kasih sudah ikut mengembangkan **SPARX**.  
Setiap kontribusi—besar maupun kecil—sangat berarti bagi perkembangan proyek ini.

Jika ada pertanyaan, silakan buka:

- [Issue](https://github.com/ndeso17/SPARX/issues)
- atau hubungi maintainer ([ndeso17](https://github.com/ndeso17))
