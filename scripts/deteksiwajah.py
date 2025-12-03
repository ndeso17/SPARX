import os
import sys
import json
import argparse
import binascii
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  

DEFAULT_FACE_MODEL = "data/wajah.pt"
DEFAULT_FALLBACK_MODEL = "data/yolov8n.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_KMEANS_K = 3
DEFAULT_KMEANS_ATTEMPTS = 5
DEFAULT_MIN_FACE_AREA_RATIO = 0.01 

OUTPUT_DIR = "output/wajah"


def load_model_with_fallback(path: str, fallback: str):
    if os.path.isfile(path):
        print(f"[INFO] Memuat model: {path}")
        return YOLO(path)
    elif os.path.isfile(fallback):
        print(f"[WARN] Model tidak ditemukan di {path}. Menggunakan fallback {fallback}")
        return YOLO(fallback)
    else:
        raise FileNotFoundError(f"Model {path} maupun fallback {fallback} tidak ditemukan.")


def bbox_to_int(bbox: np.ndarray) -> Tuple[int,int,int,int]:
    """Konversi float xyxy ke tuple int (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def bbox_area(box: Tuple[int,int,int,int]) -> int:
    """Hitung luas bbox."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bgr_to_hex(color_bgr: Tuple[int,int,int]) -> str:
    b, g, r = color_bgr
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def file_to_hex(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return binascii.hexlify(data).decode("ascii")


def draw_box(img: np.ndarray, box: Tuple[int,int,int,int], label: str, 
             color: Tuple[int,int,int]=(0,255,0), thickness=2):
    x1,y1,x2,y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    # label background
    (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 8, y1), color, -1)
    cv2.putText(img, label, (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255,255,255), 1, cv2.LINE_AA)

def is_valid_face_detection(bbox: Tuple[int,int,int,int], 
                            conf: float,
                            img_width: int, 
                            img_height: int,
                            min_area_ratio: float) -> Tuple[bool, str]:
    """
    Validasi deteksi wajah berdasarkan ukuran dan rasio aspek. 
    Mengembalikan (is_valid, reason).
    """
    box_area = bbox_area(bbox)
    img_area = img_width * img_height
    area_ratio = box_area / img_area if img_area > 0 else 0
 
    if area_ratio < min_area_ratio:
        return False, f"Wajah terlalu kecil (rasio area: {area_ratio:.2%})"
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0
    
    #? aspect ratio antara 0.5 - 1.8 untuk wajah normal
    if aspect_ratio < 0.4 or aspect_ratio > 2.0:
        return False, f"Wajah dengan rasio aspek tidak biasa: {aspect_ratio:.2f}"
    
    return True, "OK"


def run_face_detection(image_path: str,
                       face_model_path: str,
                       fallback_model_path: str,
                       conf_thresh: float,
                       iou_thresh: float,
                       min_area_ratio: float):
    """
    Jalankan deteksi wajah hanya pada gambar yang diberikan.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_basename = Path(image_path).stem

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    annotated = img_bgr.copy()

    face_model = load_model_with_fallback(face_model_path, fallback_model_path)

    results_summary: Dict[str, Any] = {
        "image_path": str(image_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_size": {"width": orig_w, "height": orig_h},
        "parameters": {
            "confidence_threshold": conf_thresh,
            "iou_threshold": iou_thresh,
            "min_face_area_ratio": min_area_ratio
        },
        "wajah": [],
        "filtered_wajah": []
    }

    print(f"[STEP] Menjalankan deteksi wajah...")
    print(f"[PARAM] Confidence: {conf_thresh}, IOU: {iou_thresh}, Min Area: {min_area_ratio}")
    
    res = face_model.predict(
        source=image_path, 
        conf=conf_thresh, 
        iou=iou_thresh, 
        verbose=False, 
        device='cpu'
    )
    
    face_count = 0
    filtered_count = 0
    
    for r in res:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_idxs = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        
        for bbox, conf, cls_idx in zip(xyxy, confs, cls_idxs):
            x1, y1, x2, y2 = bbox_to_int(bbox)
            cls_name = face_model.names[int(cls_idx)] if int(cls_idx) in face_model.names else str(int(cls_idx))
            
            is_valid, reason = is_valid_face_detection(
                (x1, y1, x2, y2), 
                float(conf),
                orig_w, 
                orig_h,
                min_area_ratio
            )
            
            face_obj = {
                "class_name": cls_name,
                "confidence": float(conf),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "bbox_size": {
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": bbox_area((x1, y1, x2, y2))
                },
                "validation": reason
            }
            
            if not is_valid:
                print(f"[FILTER] Wajah ditolak - {reason}")
                results_summary["filtered_wajah"].append(face_obj)
                filtered_count += 1
                continue
            
            face_count += 1
            label = f"Wajah {face_count}: {conf:.2f}"
            draw_box(annotated, (x1, y1, x2, y2), label, color=(0, 255, 0))
            
            results_summary["wajah"].append(face_obj)
            print(f"[DETECTED] Wajah #{face_count}: confidence={conf:.3f}, bbox=[{x1},{y1},{x2},{y2}]")
    
 
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    out_img_path = os.path.join(OUTPUT_DIR, f"{img_basename}_{timestamp_str}_wajah.jpg")
    cv2.imwrite(out_img_path, annotated)
    print(f"\n[INFO] Gambar beranotasi disimpan di: {out_img_path}")
    
    out_json_path = os.path.join(OUTPUT_DIR, f"{img_basename}_{timestamp_str}_wajah.json")
    with open(out_json_path, "w") as jf:
        json.dump(results_summary, jf, indent=2)
    print(f"[INFO] Hasil JSON disimpan di: {out_json_path}")
    

    print(f"\n{'='*50}")
    print(f"RINGKASAN:")
    print(f"  Total wajah terdeteksi: {face_count}")
    print(f"  Wajah yang disaring: {filtered_count}")
    print(f"{'='*50}\n")

    return out_img_path, out_json_path, results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Face Detection Script - Deteksi wajah saja',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Penggunaan dasar
  python deteksiwajah.py foto.jpg
  
  # Dengan ambang confidence kustom
  python deteksiwajah.py foto.jpg --conf 0.7
  
  # Dengan semua parameter kustom
  python deteksiwajah.py foto.jpg --conf 0.6 --iou 0.5 --min-area 0.02
  
  # Dengan model kustom
  python deteksiwajah.py foto.jpg --model models/custom_face.pt
        """
    )
    
    # Required arguments
    parser.add_argument('image', 
                       help='Path to input image')
    
    # Optional arguments
    parser.add_argument('--model', '-m',
                       default=DEFAULT_FACE_MODEL,
                       help=f'Path to face detection model (default: {DEFAULT_FACE_MODEL})')
    
    parser.add_argument('--fallback',
                       default=DEFAULT_FALLBACK_MODEL,
                       help=f'Path to fallback model (default: {DEFAULT_FALLBACK_MODEL})')
    
    parser.add_argument('--conf', '-c',
                       type=float,
                       default=DEFAULT_CONFIDENCE,
                       help=f'Confidence threshold (default: {DEFAULT_CONFIDENCE})')
    
    parser.add_argument('--iou',
                       type=float,
                       default=DEFAULT_IOU,
                       help=f'IOU threshold for NMS (default: {DEFAULT_IOU})')
    
    parser.add_argument('--min-area',
                       type=float,
                       default=DEFAULT_MIN_FACE_AREA_RATIO,
                       help=f'Minimum face area ratio (default: {DEFAULT_MIN_FACE_AREA_RATIO})')
    
    parser.add_argument('--kmeans-k',
                       type=int,
                       default=DEFAULT_KMEANS_K,
                       help=f'K-means clusters (default: {DEFAULT_KMEANS_K}) - not used currently')
    
    parser.add_argument('--kmeans-attempts',
                       type=int,
                       default=DEFAULT_KMEANS_ATTEMPTS,
                       help=f'K-means attempts (default: {DEFAULT_KMEANS_ATTEMPTS}) - not used currently')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        out_img, out_json, summary = run_face_detection(
            image_path=args.image,
            face_model_path=args.model,
            fallback_model_path=args.fallback,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            min_area_ratio=args.min_area
        )
        
        print(json.dumps({
            "status": "success",
            "annotated_image": out_img,
            "json_file": out_json,
            "summary": {
                "total_wajah": len(summary["wajah"]),
                "filtered_out": len(summary["filtered_wajah"])
            }
        }, indent=2))
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()