# 📚 Panduan Lengkap Training Scripts SPARX

Panduan komprehensif untuk menggunakan script training YOLO, OCR, dan Evaluasi untuk sistem deteksi plat nomor kendaraan Indonesia.

---

## 📁 Struktur File

```
SPARX/
├── training/
│   ├── yolo.py          # Training YOLO detection
│   ├── ocr.py           # Training OCR character recognition
│   └── evaluate.py      # Evaluasi model
├── data/
│   ├── PlatNomor/       # Dataset YOLO (images + labels)
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── data.yaml
│   ├── DatasetCharacter/  # Dataset OCR (per karakter)
│   │   ├── train/
│   │   │   ├── A/, B/, ..., Z/, 0/, ..., 9/
│   │   └── val/
│   │       └── A/, B/, ..., Z/, 0/, ..., 9/
│   └── runs/            # Output training
│       ├── detect/      # Output YOLO
│       └── ocr/         # Output OCR
└── output/
    └── evaluation/      # Output evaluasi
```

---

## 🎯 1. Training YOLO (Detection)

### **Tujuan**: Melatih model untuk mendeteksi lokasi plat nomor dalam gambar

### **Persiapan Dataset**

Dataset harus dalam format YOLO:

```
PlatNomor/
├── train/
│   ├── images/          # Gambar training
│   └── labels/          # Label YOLO format (.txt)
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml            # Konfigurasi dataset
```

**Format label** (YOLO format):

```
0 0.5 0.5 0.3 0.1
# class_id x_center y_center width height (normalized 0-1)
```

### **Perintah Training**

#### **1.1 Validasi Dataset Saja**

```bash
python training/yolo.py \
    --data data/PlatNomor \
    --validate-only
```

- ✅ Mengecek struktur folder
- ✅ Validasi format annotations
- ✅ Visualisasi sample data

#### **1.2 Training Dasar**

```bash
python training/yolo.py \
    --data data/PlatNomor \
    --epochs 100 \
    --batch 16 \
    --img-size 640
```

#### **1.3 Training dengan Model Lebih Besar**

```bash
# YOLOv8s (small) - lebih akurat tapi lebih lambat
python training/yolo.py \
    --data data/PlatNomor \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 32 \
    --img-size 640
```

#### **1.4 Resume Training yang Terinterupsi**

```bash
python training/yolo.py \
    --data data/PlatNomor \
    --resume
```

#### **1.5 Training dengan Dataset Custom Split**

Jika dataset sudah ter-split (train/val/test):

```bash
python training/yolo.py \
    --data data/PlatNomor \
    --no-split \
    --epochs 100
```

#### **1.6 Training dengan GPU Spesifik**

```bash
# Gunakan GPU 0
python training/yolo.py --data data/PlatNomor --device 0

# Gunakan CPU
python training/yolo.py --data data/PlatNomor --device cpu
```

### **Parameter Penting**

| Parameter         | Default            | Deskripsi               |
| ----------------- | ------------------ | ----------------------- |
| `--data`          | `data/dataset`     | Path dataset YOLO       |
| `--output`        | `data/runs/detect` | Output directory        |
| `--model`         | `yolov8n.pt`       | Model base (n/s/m/l/x)  |
| `--epochs`        | `100`              | Jumlah epoch training   |
| `--batch`         | `16`               | Batch size              |
| `--img-size`      | `640`              | Ukuran input image      |
| `--patience`      | `50`               | Early stopping patience |
| `--device`        | `0`                | GPU device atau `cpu`   |
| `--resume`        | `False`            | Resume dari checkpoint  |
| `--validate-only` | `False`            | Hanya validasi dataset  |
| `--no-split`      | `False`            | Skip auto-split dataset |

### **Output YOLO Training**

```
data/runs/detect/plat_nomor_20241210_120000/
├── weights/
│   ├── best.pt          # Model terbaik (gunakan ini!)
│   ├── last.pt          # Model epoch terakhir
│   └── best.onnx        # Model ONNX (untuk deployment)
├── results.png          # Grafik training metrics
├── confusion_matrix.png # Confusion matrix
├── PR_curve.png         # Precision-Recall curve
└── args.yaml            # Training arguments
```

---

## 🔤 2. Training OCR (Character Recognition)

### **Tujuan**: Melatih model untuk mengenali karakter individual pada plat nomor

### **Persiapan Dataset**

Dataset harus berupa folder per karakter:

```
DatasetCharacter/
├── train/
│   ├── A/
│   │   ├── A_0001.png
│   │   ├── A_0002.png
│   │   └── ...
│   ├── B/
│   ├── ...
│   ├── Z/
│   ├── 0/
│   └── 9/
└── val/
    ├── A/
    ├── B/
    └── ...
```

**Catatan**: Setiap gambar harus berisi **1 karakter saja** dengan background putih/terang.

### **Perintah Training**

#### **2.1 Buat Dataset Synthetic (Jika Belum Ada)**

```bash
python training/ocr.py \
    --create-dataset \
    --samples 500 \
    --epochs 50 \
    --batch 32
```

- Otomatis membuat 500 sample per karakter
- Total: 18,000 samples (36 karakter × 500)
- Sudah termasuk augmentasi

#### **2.2 Training dengan Dataset Existing**

```bash
python training/ocr.py \
    --data data/DatasetCharacter \
    --epochs 100 \
    --batch 32 \
    --img-size 64
```

#### **2.3 Training Quick Test (Dataset Kecil)**

```bash
python training/ocr.py \
    --create-dataset \
    --samples 100 \
    --epochs 10 \
    --batch 16
```

#### **2.4 Resume Training**

```bash
python training/ocr.py \
    --data data/DatasetCharacter \
    --resume
```

#### **2.5 Training dengan Custom Learning Rate**

```bash
python training/ocr.py \
    --data data/DatasetCharacter \
    --epochs 100 \
    --batch 32 \
    --lr 0.0001
```

### **Parameter Penting**

| Parameter          | Default                 | Deskripsi                        |
| ------------------ | ----------------------- | -------------------------------- |
| `--data`           | `data/DatasetCharacter` | Path dataset karakter            |
| `--output`         | `data/runs/ocr`         | Output directory                 |
| `--create-dataset` | `False`                 | Buat dataset synthetic           |
| `--samples`        | `500`                   | Samples per karakter (synthetic) |
| `--epochs`         | `50`                    | Jumlah epoch training            |
| `--batch`          | `32`                    | Batch size                       |
| `--lr`             | `0.001`                 | Learning rate                    |
| `--img-size`       | `64`                    | Ukuran input (square)            |
| `--resume`         | `False`                 | Resume dari checkpoint           |
| `--gpu`            | `0`                     | GPU device atau `-1` untuk CPU   |

### **Output OCR Training**

```
data/runs/ocr/char_recognition_20241210_120000/
├── best_model.keras           # Model terbaik (gunakan ini!)
├── final_model.keras          # Model epoch terakhir
├── char_labels.json           # Mapping karakter
├── char_all_model.onnx        # Model ONNX
├── char_all_model.tflite      # Model TFLite
├── training_history.png       # Grafik training
├── confusion_matrix.png       # Confusion matrix
├── training_log.csv           # Log training
└── dataset_samples.png        # Sample dataset
```

**File di `data/` untuk production:**

```
data/
├── char_all_model.keras       # Model OCR
└── char_all_labels.json       # Label mapping
```

---

## 📊 3. Evaluasi Model

### **Tujuan**: Mengevaluasi performa model YOLO dan OCR

### **Persiapan**

Pastikan punya:

1. ✅ Model YOLO terlatih (`.pt`)
2. ✅ Model OCR terlatih (`.keras`)
3. ✅ Test dataset dengan ground truth
4. ✅ (Opsional) File ground truth JSON

### **Perintah Evaluasi**

#### **3.1 Evaluasi YOLO Saja**

```bash
python training/evaluate.py \
    --yolo data/plat_nomor.pt \
    --test data/PlatNomor/test \
    --yolo-only
```

**Output**:

- Precision, Recall, F1-Score
- Average IoU
- Detection speed (FPS)
- Confusion matrix

#### **3.2 Evaluasi Pipeline Lengkap (YOLO + OCR)**

```bash
python training/evaluate.py \
    --yolo data/plat_nomor.pt \
    --test data/PlatNomor/test \
    --pipeline
```

**Output**:

- Detection rate
- Character accuracy
- Plate accuracy
- End-to-end latency
- Pipeline FPS

#### **3.3 Evaluasi dengan Ground Truth Custom**

```bash
python training/evaluate.py \
    --yolo data/plat_nomor.pt \
    --test data/PlatNomor/test \
    --gt ground_truth.json \
    --pipeline
```

**Format `ground_truth.json`**:

```json
[
  {
    "image": "image_001.jpg",
    "plate_text": "B1234XYZ"
  },
  {
    "image": "image_002.jpg",
    "plate_text": "D5678ABC"
  }
]
```

#### **3.4 Evaluasi dengan Confidence Threshold Custom**

```bash
python training/evaluate.py \
    --yolo data/plat_nomor.pt \
    --test data/PlatNomor/test \
    --conf 0.7 \
    --pipeline
```

### **Parameter Penting**

| Parameter     | Default                        | Deskripsi              |
| ------------- | ------------------------------ | ---------------------- |
| `--yolo`      | `data/runs/detect/.../best.pt` | Path model YOLO        |
| `--ocr`       | `data/char_all_model.keras`    | Path model OCR         |
| `--test`      | `data/PlatNomor/test`          | Path test dataset      |
| `--output`    | `output/evaluation`            | Output directory       |
| `--conf`      | `0.5`                          | Confidence threshold   |
| `--gt`        | `None`                         | Ground truth JSON file |
| `--pipeline`  | `False`                        | Evaluasi end-to-end    |
| `--yolo-only` | `False`                        | Evaluasi YOLO saja     |

### **Output Evaluasi**

```
output/evaluation/evaluation_20241210_120000/
├── evaluation_report.json     # Report lengkap
├── yolo_evaluation.json       # Hasil YOLO
├── pipeline_evaluation.json   # Hasil pipeline
├── detection_evaluation.png   # Visualisasi detection
├── ocr_analysis.png           # Visualisasi OCR
└── comparison_plot.png        # Perbandingan semua metrics
```

**Contoh Output Console**:

```
================================================================================
EVALUATION SUMMARY
================================================================================

📊 YOLO Detection:
  • Precision:  95.30%
  • Recall:     92.80%
  • F1 Score:   94.03%
  • Avg IoU:    87.50%
  • Speed:      45.20 FPS

🔤 OCR Recognition:
  • Char Acc:   96.70%
  • Plate Acc:  89.50%
  • Avg Lev:    0.35

🚀 Pipeline Performance:
  • Detection:  93.20%
  • Speed:      28.50 FPS
  • Latency:    35.09 ms
================================================================================
```

---

## 🔄 Workflow Lengkap

### **1. Persiapan Data**

```bash
# Pastikan struktur folder benar
ls data/PlatNomor/train/images
ls data/PlatNomor/train/labels
```

### **2. Training YOLO**

```bash
# Validasi dulu
python training/yolo.py --data data/PlatNomor --validate-only

# Training
python training/yolo.py \
    --data data/PlatNomor \
    --epochs 100 \
    --batch 16 \
    --device 0
```

### **3. Training OCR**

```bash
# Buat dataset synthetic (jika belum ada)
python training/ocr.py \
    --create-dataset \
    --samples 500 \
    --epochs 50

# Atau gunakan dataset existing
python training/ocr.py \
    --data data/DatasetCharacter \
    --epochs 100 \
    --batch 32
```

### **4. Evaluasi**

```bash
# Evaluasi YOLO
python training/evaluate.py \
    --yolo data/runs/detect/plat_nomor_TIMESTAMP/weights/best.pt \
    --test data/PlatNomor/test \
    --yolo-only

# Evaluasi Pipeline
python training/evaluate.py \
    --yolo data/plat_nomor.pt \
    --test data/PlatNomor/test \
    --pipeline
```

---

## 🎯 Tips & Best Practices

### **YOLO Training**

- ✅ Gunakan `yolov8n.pt` untuk prototype cepat
- ✅ Gunakan `yolov8s.pt` atau `yolov8m.pt` untuk production
- ✅ Minimal 1000+ images untuk hasil bagus
- ✅ Augmentasi data sudah otomatis (mosaic, flip, dll)
- ✅ Monitor `mAP@0.5` dan `mAP@0.5:0.95`

### **OCR Training**

- ✅ Dataset synthetic sudah cukup bagus untuk prototype
- ✅ Untuk production, tambahkan real cropped characters
- ✅ Perhatikan confusion matrix untuk karakter mirip (O vs 0, I vs 1)
- ✅ Top-3 accuracy biasanya 99%+

### **Evaluasi**

- ✅ Gunakan test set yang berbeda dari training
- ✅ Monitor detection rate dan plate accuracy
- ✅ Pipeline FPS >20 FPS bagus untuk real-time
- ✅ Character accuracy >95% target minimum

### **Hardware**

- 🖥️ **CPU Only**: Bisa, tapi lambat (tambahkan `--device cpu`)
- 🎮 **GPU 4GB**: Cukup untuk `yolov8n` + batch 8-16
- 🎮 **GPU 8GB+**: Bisa `yolov8s/m` + batch 32+
- 💾 **RAM**: Minimal 8GB, recommended 16GB+

---

## ❓ Troubleshooting

### **Error: CUDA out of memory**

```bash
# Kurangi batch size
python training/yolo.py --data data/PlatNomor --batch 8

# Atau gunakan CPU
python training/yolo.py --data data/PlatNomor --device cpu
```

### **Error: Dataset not found**

```bash
# Pastikan struktur folder benar
python training/yolo.py --data data/PlatNomor --validate-only
```

### **Error: shutil not defined**

Sudah diperbaiki di script yang baru.

### **Training terlalu lambat**

```bash
# Kurangi image size
python training/yolo.py --data data/PlatNomor --img-size 416

# Kurangi epochs untuk testing
python training/yolo.py --data data/PlatNomor --epochs 10
```

### **Accuracy rendah**

- Tambah data training
- Training lebih lama (150-200 epochs)
- Gunakan model lebih besar (yolov8s/m)
- Cek kualitas annotations

---
