# Deteksi Wajah IP Camera

Script Python untuk deteksi wajah real-time dari IP Camera menggunakan YOLOv8. Sistem ini akan menyimpan wajah unik yang terdeteksi dan otomatis meng-upgrade foto jika menemukan kualitas yang lebih baik.

## Fitur

- ✅ Deteksi wajah real-time dari berbagai sumber video (RTSP, HTTP, RTMP, Webcam)
- ✅ Penyimpanan wajah unik dengan sistem face matching
- ✅ Auto-upgrade foto wajah ke kualitas lebih baik
- ✅ Validasi deteksi (ukuran minimum, aspect ratio)
- ✅ Database JSON untuk metadata wajah
- ✅ Mode headless untuk server

## Metode yang Digunakan

### 1. Object Detection

| Metode            | Deskripsi                                                    |
| ----------------- | ------------------------------------------------------------ |
| **YOLOv8**        | Model deep learning untuk deteksi wajah (`ultralytics.YOLO`) |
| `model.predict()` | Inferensi deteksi objek dengan threshold confidence dan IOU  |

### 2. Image Processing (OpenCV)

| Metode               | Deskripsi                                      |
| -------------------- | ---------------------------------------------- |
| `cv2.VideoCapture()` | Membuka stream video dari URL atau webcam      |
| `cv2.cvtColor()`     | Konversi warna (BGR → Grayscale, BGR → HSV)    |
| `cv2.Laplacian()`    | Deteksi edge untuk menghitung sharpness gambar |
| `cv2.Canny()`        | Deteksi edge untuk ekstraksi fitur             |
| `cv2.calcHist()`     | Menghitung histogram warna                     |
| `cv2.normalize()`    | Normalisasi histogram                          |
| `cv2.resize()`       | Resize gambar ke ukuran standar (128x128)      |
| `cv2.rectangle()`    | Menggambar bounding box                        |
| `cv2.putText()`      | Menampilkan teks pada frame                    |
| `cv2.imwrite()`      | Menyimpan gambar ke file                       |
| `cv2.imshow()`       | Menampilkan video frame                        |

### 3. Face Feature Extraction & Matching

| Metode                 | Deskripsi                                                         |
| ---------------------- | ----------------------------------------------------------------- |
| **Histogram Features** | Ekstraksi histogram grayscale (64 bins) dan HSV (H:32, S:32 bins) |
| **Edge Features**      | Histogram dari Canny edge detection                               |
| **Cosine Similarity**  | Perhitungan kesamaan antara dua vektor fitur                      |

### 4. Image Quality Assessment

| Metrik         | Formula                                  |
| -------------- | ---------------------------------------- |
| **Sharpness**  | Variance dari Laplacian (blur detection) |
| **Brightness** | Mean intensity grayscale                 |
| **Contrast**   | Standard deviation grayscale             |

### 5. Face Validation

| Validasi         | Kriteria                         |
| ---------------- | -------------------------------- |
| **Minimum Area** | Rasio area wajah ≥ 1% dari frame |
| **Aspect Ratio** | Antara 0.4 - 2.0 (width/height)  |

## Dependencies

```bash
pip install opencv-python numpy ultralytics
```

---

## Penggunaan `deteksiwajah.py`

### Basic (gunakan default)

```
python deteksiwajah.py path
```

### Custom confidence

```
python deteksiwajah.py bahlil.jpeg --conf 0.7
```

### Custom confidence + IOU

```
python deteksiwajah.py bahlil.jpeg --conf 0.6 --iou 0.5
```

### Semua parameter custom

```
python deteksiwajah.py bahlil.jpeg \
  --conf 0.65 \
  --iou 0.5 \
  --min-area 0.02 \
  --model models/wajah.pt \
  --fallback models/yolov8n.pt
```

### Lihat semua opsi

bash

```
python deteksiwajah.py --help
```

### Parameter yang Tersedia:

| Parameter           | Short | Default           | Deskripsi                                                       |
| ------------------- | ----- | ----------------- | --------------------------------------------------------------- |
| `image`             | -     | Required          | Path ke gambar                                                  |
| `--model`           | `-m`  | `data/wajah.pt`   | Path ke model YOLO                                              |
| `--fallback`        | -     | `data/yolov8n.pt` | Path ke model YOLO fallback                                     |
| `--conf`            | `-c`  | `0.5`             | Nilai threshold untuk confidence                                |
| `--iou`             | -     | `0.45`            | Nilai threshold untuk IOU                                       |
| `--min-area`        | -     | `0.01`            | Nilai threshold untuk area wajah                                |
| `--kmeans-k`        | -     | `3`               | Nilai jumlah cluster/kelompok warna yang akan dicari            |
| `--kmeans-attempts` | -     | `5`               | Nilai perulangan berapa kali algoritma k-means dijalankan ulang |

### JSON Structure:

```
{
  "image_path": "img/pakbahlil.jpeg",
  "timestamp": "2025-12-03T14:21:34.423476+00:00",
  "image_size": {
    "width": 212,
    "height": 237
  },
  "parameters": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "min_face_area_ratio": 0.01
  },
  "wajah": [
    {
      "class_name": "face",
      "confidence": 0.8932997584342957,
      "bbox": {
        "x1": 62,
        "y1": 29,
        "x2": 147,
        "y2": 149
      },
      "bbox_size": {
        "width": 85,
        "height": 120,
        "area": 10200
      },
      "validation": "OK"
    }
  ],
  "filtered_wajah": []
}
```

---

## Penggunaan `deteksiwajah_ipcam.py`

### Penggunaan Dasar

```bash
# RTSP stream (IP Camera)
python deteksiwajah_ipcam.py rtsp://admin:password@192.168.1.100:554/stream

# HTTP stream
python deteksiwajah_ipcam.py http://192.168.1.100:8080/video

# Webcam lokal (0 = default camera)
python deteksiwajah_ipcam.py 0
```

### Penggunaan dengan Parameter Kustom

```bash
# Confidence threshold tinggi (lebih strict)
python deteksiwajah_ipcam.py rtsp://192.168.1.100:554/stream --conf 0.7

# Similarity threshold tinggi (lebih strict matching)
python deteksiwajah_ipcam.py rtsp://192.168.1.100:554/stream --similarity 0.75

# Custom model
python deteksiwajah_ipcam.py 0 --model custom_model.pt

# Mode headless (tanpa display window)
python deteksiwajah_ipcam.py rtsp://192.168.1.100:554/stream --no-display
```

### Semua Parameter

| Parameter      | Default           | Deskripsi                    |
| -------------- | ----------------- | ---------------------------- |
| `stream_url`   | _required_        | URL stream atau nomor webcam |
| `--model, -m`  | `data/wajah.pt`   | Path ke model deteksi wajah  |
| `--fallback`   | `data/yolov8n.pt` | Path ke fallback model       |
| `--conf, -c`   | `0.5`             | Confidence threshold (0-1)   |
| `--iou`        | `0.45`            | IOU threshold untuk NMS      |
| `--min-area`   | `0.01`            | Minimum face area ratio      |
| `--similarity` | `0.70`            | Face similarity threshold    |
| `--frame-skip` | `1`               | Proses setiap N frame        |
| `--no-display` | `False`           | Mode headless                |

## Kontrol Keyboard

| Key | Fungsi              |
| --- | ------------------- |
| `q` | Keluar dari program |
| `s` | Ambil screenshot    |

## Format URL yang Didukung

- **RTSP**: `rtsp://[user:pass@]ip:port/path`
- **HTTP**: `http://ip:port/path`
- **RTMP**: `rtmp://ip:port/path`
- **Webcam**: `0`, `1`, `2`, ...

## Output

### 1. Database JSON (`detected_faces.json`)

```json
{
  "faces": [
    {
      "face_id": "face_20240101_120000_123456",
      "timestamp": "2024-01-01T12:00:00.123456",
      "confidence": 0.95,
      "quality": 78.5,
      "bbox": { "x1": 100, "y1": 50, "x2": 200, "y2": 180 },
      "face_image_path": "output/wajah_ipcam/faces/face_xxx.jpg",
      "frame_image_path": "output/wajah_ipcam/frames/face_xxx_frame.jpg"
    }
  ],
  "total_detected": 1
}
```

### 2. Statistik Akhir

```
==================================================
STATISTIK AKHIR:
  Total frame diproses: 1500
  Total deteksi wajah: 250
  Wajah baru tersimpan: 5
  Wajah di-upgrade: 12
  Total wajah di database: 5
==================================================
```

## Warna Bounding Box

| Warna     | Status                                 |
| --------- | -------------------------------------- |
| 🟢 Hijau  | Wajah baru tersimpan                   |
| 🟠 Oranye | Wajah sudah ada di database            |
| 🟡 Kuning | Wajah di-upgrade (kualitas lebih baik) |

## Algoritma Face Matching

1. **Feature Extraction**:

   - Resize wajah ke 128x128
   - Extract histogram grayscale (64 bins)
   - Extract histogram HSV (H: 32 bins, S: 32 bins)
   - Extract edge histogram dari Canny (32 bins)
   - Gabungkan semua fitur (160 dimensi)

2. **Similarity Calculation**:

   - Cosine similarity antara vektor fitur
   - Normalisasi ke range 0-1
   - Threshold default: 0.70

3. **Quality-Based Update**:
   - Jika wajah match dan kualitas baru > kualitas lama → Replace
   - Jika wajah match dan kualitas baru ≤ kualitas lama → Skip
   - Jika tidak match → Simpan sebagai wajah baru

## Fungsi Utama dalam Script

| Fungsi                       | Deskripsi                                  |
| ---------------------------- | ------------------------------------------ |
| `load_model_with_fallback()` | Memuat model YOLO dengan fallback          |
| `bbox_to_int()`              | Konversi bounding box ke integer           |
| `bbox_area()`                | Menghitung area bounding box               |
| `is_valid_face_detection()`  | Validasi deteksi wajah                     |
| `calculate_image_quality()`  | Menghitung skor kualitas gambar            |
| `extract_face_features()`    | Ekstraksi fitur wajah untuk matching       |
| `compare_face_features()`    | Membandingkan dua wajah                    |
| `load_detected_faces_db()`   | Memuat database wajah                      |
| `save_detected_faces_db()`   | Menyimpan database wajah                   |
| `find_matching_face()`       | Mencari wajah yang cocok di database       |
| `draw_box()`                 | Menggambar bounding box dengan label       |
| `save_new_face()`            | Menyimpan wajah baru ke database           |
| `replace_face()`             | Mengganti wajah dengan kualitas lebih baik |
| `run_ipcam_detection()`      | Fungsi utama deteksi real-time             |
| `main()`                     | Entry point dengan argument parser         |

---

## 📜 Lisensi

Proyek SPARX dirilis menggunakan lisensi **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**.

## Anda bebas untuk:

- Menggunakan, menyalin, dan membagikan kode ini.
- Memodifikasi atau mengembangkan fitur baru dari kode ini.

## Dengan syarat:

1. Memberikan atribusi kepada pembuat dan kontibutor.
2. Tidak digunakan untuk keperluan komersial, termasuk menjual, menyewakan, atau memasukkan kode ini ke produk/layanan berbayar.

Lisensi lengkap dapat dibaca di:  
[https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)
