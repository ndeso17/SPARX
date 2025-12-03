## Penggunaan deteksiwajah.py

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
json{
  "image_path": "bahlil.jpeg",
  "timestamp": "2025-12-03T13:48:19+00:00",
  "image_size": {"width": 212, "height": 237},
  "parameters": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "min_face_area_ratio": 0.01
  },
  "faces": [
    {
      "class_name": "face",
      "confidence": 0.893,
      "bbox": {"x1": 62, "y1": 29, "x2": 147, "y2": 149},
      "bbox_size": {"width": 85, "height": 120, "area": 10200},
      "validation": "OK"
    }
  ],
  "filtered_faces": []
}
```
