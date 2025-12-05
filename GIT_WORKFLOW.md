# 🚀 **Panduan Workflow Git untuk Tim SPARX**

Dokumen ini berisi aturan kerja Git yang digunakan oleh seluruh anggota tim SPARX. Tujuannya agar:

- Tidak terjadi konflik kode yang tidak perlu
- Riwayat commit rapi
- Setiap perubahan terkontrol dan mudah di-review
- Branch `main` tetap stabil

---

# 📌 **1. Struktur Branch di SPARX**

```
main
│
├── feature/<nama-fitur>
├── bugfix/<nama-bug>
├── docs/<nama-dokumen>
└── hotfix/<perbaikan-kritis>
```

### Penjelasan:

| Branch       | Fungsi                                         |
| ------------ | ---------------------------------------------- |
| **main**     | branch utama, harus selalu stabil, bebas error |
| **feature/** | tempat semua fitur baru dikembangkan           |
| **bugfix/**  | memperbaiki bug tanpa fitur baru               |
| **docs/**    | perbaikan dokumentasi                          |
| **hotfix/**  | perbaikan mendesak langsung dari `main`        |

---

# 📌 **2. Aturan Umum Kerja Tim**

### ✔ Selalu mulai dengan `git pull`

Sebelum melakukan perubahan:

```bash
git pull --rebase
```

### ✔ Dilarang push langsung ke `main`

Semua perubahan masuk via **Pull Request (PR)**.

### ✔ Satu fitur = satu branch

Contoh:

```bash
git checkout -b feature/deteksi-warna
```

### ✔ Satu PR hanya untuk satu perubahan

Tidak boleh mencampur:

❌ perbaikan bug + fitur baru
❌ fitur baru + refactor besar
❌ dokumentasi + logika kode

---

# 📌 **3. Alur Kerja Lengkap (Best Practice)**

## **Langkah 1 — Pastikan repo kamu up-to-date**

```bash
git checkout main
git pull --rebase
```

---

## **Langkah 2 — Buat branch baru**

Gunakan format:

```
feature/
bugfix/
docs/
hotfix/
```

Contoh membuat fitur deteksi kendaraan:

```bash
git checkout -b feature/deteksi-kendaraan
```

---

## **Langkah 3 — Lakukan perubahan dan commit**

Tambahkan file:

```bash
git add .
```

Commit dengan format:

```
feat: untuk fitur baru
fix: memperbaiki bug
docs: update dokumentasi
refactor: perubahan internal tanpa menambah fitur
test: menambah/perbaikan testing
style: formatting / linting
perf: meningkatkan performa
```

Contoh commit:

```bash
git commit -m "feat: tambah modul deteksi warna kendaraan"
```

---

## **Langkah 4 — Sinkronisasi sebelum push**

Sangat penting untuk menghindari konflik:

```bash
git pull --rebase origin main
```

---

## **Langkah 5 — Push ke GitHub**

```bash
git push -u origin feature/deteksi-kendaraan
```

---

## **Langkah 6 — Buat Pull Request**

- Pilih base: `main`
- Pilih compare: `feature/<namamu>`
- Tambahkan deskripsi perubahan
- Tambahkan screenshot bila perlu

Setelah PR dibuat:

- Anggota lain melakukan review
- Jika disetujui → merge ke `main`

---

# 📌 **4. Aturan Merge Pull Request**

1. Wajib melalui review setidaknya 1 anggota tim
2. Pastikan CI/Testing (jika ada) lulus
3. Tidak ada konflik
4. Tidak menghapus fitur lain
5. Tidak merusak repository

### **Tipe merge yang digunakan:**

Gunakan **Squash and Merge**
→ membuat riwayat commit di `main` tetap bersih.

---

# 📌 **5. Aturan Penamaan Commit**

| Format      | Contoh                                  |
| ----------- | --------------------------------------- |
| `feat:`     | feat: tambah modul deteksi wajah        |
| `fix:`      | fix: perbaiki bounding box error        |
| `docs:`     | docs: update README dan lisensi         |
| `refactor:` | refactor: perbaikan struktur folder     |
| `style:`    | style: formatting PEP8                  |
| `perf:`     | perf: optimasi model load waktu startup |
| `test:`     | test: tambah unit test YOLO model       |

---

# 📌 **6. Konflik Git — Cara Menyelesaikan**

Jika ada konflik:

1. Buka file yang konflik
2. Pilih kode yang tepat
3. Hapus tanda konflik:

   ```
   <<<<<<< HEAD
   =======
   >>>>>>>
   ```

4. Tambahkan file:

   ```bash
   git add .
   ```

5. Lanjutkan rebase:

   ```bash
   git rebase --continue
   ```

Kalau bingung, tim bisa jelaskan via chat sebelum merge.

---

# 📌 **7. Aturan Folder dan File**

Untuk menjaga repo tetap rapi:

- Folder `scripts/` → berisi utilitas
- Folder `assets/` → gambar dokumentasi
- Folder `output/` → hasil deteksi (tidak di-commit)
- Folder `data/` → model YOLO / dataset (opsional tidak disertakan)
- Folder `.github/` → issue template, PR template, dependabot
- Folder `docs/` → dokumentasi teknis

---

# 📌 **8. Hal yang Dilarang**

❌ Push langsung ke `main`
❌ Force push ke `main` (`git push --force`)
❌ Commit besar tanpa pembagian kecil
❌ Menghapus PR orang lain
❌ Mengubah riwayat commit yang sudah di-merge

---

# 📌 **9. Checklist Sebelum Push**

✔ Sudah `git pull --rebase`
✔ Commit jelas dan rapi
✔ Tidak ada file sementara (cache, output, dataset)
✔ Tidak ada credential/pw
✔ Sudah dites lokal (minimal basic run)

---

# 🚀 **10. Ringkasan Singkat Workflow**

```
git checkout main
git pull --rebase
git checkout -b feature/nama-fitur
... coding ...
git add .
git commit -m "feat: ..."
git pull --rebase origin main
git push -u origin feature/nama-fitur
Buat Pull Request → Review → Merge
```
