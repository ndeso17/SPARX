import os
import sys
import json
import argparse
from datetime import datetime
from typing import Tuple, Dict, Any
import time

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_FACE_MODEL = "data/wajah.pt"
DEFAULT_FALLBACK_MODEL = "data/yolov8n.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_MIN_FACE_AREA_RATIO = 0.01
#? Threshold untuk face matching
DEFAULT_SIMILARITY_THRESHOLD = 0.70  
#? Process semua frame untuk deteksi yang lebih stabil
DEFAULT_FRAME_SKIP = 1  

OUTPUT_DIR = "output/wajah_ipcam"
DETECTED_FACES_DB = "output/wajah_ipcam/detected_faces.json"


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
    x1, y1, x2, y2 = bbox
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def bbox_area(box: Tuple[int,int,int,int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_valid_face_detection(bbox: Tuple[int,int,int,int], 
                            conf: float,
                            img_width: int, 
                            img_height: int,
                            min_area_ratio: float) -> Tuple[bool, str]:
    box_area = bbox_area(bbox)
    img_area = img_width * img_height
    area_ratio = box_area / img_area if img_area > 0 else 0
 
    if area_ratio < min_area_ratio:
        return False, f"Wajah terlalu kecil (rasio area: {area_ratio:.2%})"
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0
    
    if aspect_ratio < 0.4 or aspect_ratio > 2.0:
        return False, f"Rasio aspek tidak biasa: {aspect_ratio:.2f}"
    
    return True, "OK"


def calculate_image_quality(face_crop: np.ndarray) -> float:
    """
    Hitung kualitas gambar wajah berdasarkan:
    - Sharpness (Laplacian variance)
    - Brightness
    - Contrast
    Returns quality score (0-100)
    """
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        #? Sharpness menggunakan Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        #? Brightness (mean intensity)
        brightness = gray.mean()
        
        #? Contrast (standard deviation)
        contrast = gray.std()
        
        #? mengatur ketajaman
        sharpness_score = min(sharpness / 10, 100)
        
        #? mengatur kecerahan
        brightness_optimal = 125
        brightness_score = 100 - abs(brightness - brightness_optimal) / 1.25
        brightness_score = max(0, brightness_score)
        
        #? mengatur kontras
        contrast_score = min(contrast, 100)
        
        #? Gabungkan skor dengan bobot
        quality = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return float(quality)
    except Exception as e:
        print(f"[WARN] Kalkulasi kualitas gambar wajah gagal: {e}")
        return 0.0


def extract_face_features(face_crop: np.ndarray) -> np.ndarray:
    """
    Extract multiple features dari wajah untuk matching yang lebih robust.
    Menggunakan kombinasi histogram, LBP, dan color features.
    """
    #? Resize ke ukuran standar
    face_resized = cv2.resize(face_crop, (128, 128))
    
    #? Convert ke grayscale
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    #? Histogram grayscale
    hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
    
    #? Histogram warna (HSV)
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    
    #? Edge features
    edges = cv2.Canny(gray, 50, 150)
    hist_edges = cv2.calcHist([edges], [0], None, [32], [0, 256])
    hist_edges = cv2.normalize(hist_edges, hist_edges).flatten()
    
    #? Gabungkan semua features
    features = np.concatenate([hist_gray, hist_h, hist_s, hist_edges])
    
    return features


def compare_face_features(face1: np.ndarray, face2: np.ndarray) -> float:
    """
    Compare dua wajah menggunakan multiple feature comparison.
    Returns similarity score (0-1).
    """
    try:
        #? Extract features
        features1 = extract_face_features(face1)
        features2 = extract_face_features(face2)
        
        #? Hitung kesamaan kosinus
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        #? Normalisasi ke rentang 0-1
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    except Exception as e:
        print(f"[WARN] Error membandingkan wajah: {e}")
        return 0.0


def load_detected_faces_db() -> Dict[str, Any]:
    """Load database wajah yang sudah terdeteksi."""
    if os.path.exists(DETECTED_FACES_DB):
        try:
            with open(DETECTED_FACES_DB, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Error loading database: {e}")
    return {"faces": [], "total_detected": 0}


def save_detected_faces_db(db: Dict[str, Any]):
    """Simpan database wajah yang sudah terdeteksi."""
    os.makedirs(os.path.dirname(DETECTED_FACES_DB), exist_ok=True)
    with open(DETECTED_FACES_DB, 'w') as f:
        json.dump(db, f, indent=2)


def find_matching_face(face_crop: np.ndarray, 
                       db: Dict[str, Any], 
                       threshold: float) -> Tuple[bool, str, float, float]:
    """
    Cari wajah yang matching di database.
    Returns (is_match, face_id, similarity, existing_quality)
    """
    max_similarity = 0.0
    best_match_id = ""
    best_match_quality = 0.0
    
    for face_record in db["faces"]:
        face_id = face_record["face_id"]
        
        #? memuat wajah yang tersimpan
        saved_face_path = face_record.get("face_image_path")
        if not saved_face_path or not os.path.exists(saved_face_path):
            continue
        
        saved_face = cv2.imread(saved_face_path)
        if saved_face is None:
            continue
        
        #? Bandingkan wajah
        similarity = compare_face_features(face_crop, saved_face)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_id = face_id
            best_match_quality = face_record.get("quality", 0.0)
    
    #? Periksa apakah ada kecocokan
    if max_similarity >= threshold:
        return True, best_match_id, max_similarity, best_match_quality
    
    return False, "", max_similarity, 0.0


def draw_box(img: np.ndarray, box: Tuple[int,int,int,int], label: str, 
             color: Tuple[int,int,int]=(0,255,0), thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    #? Label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - 22), (x1 + w + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 1, cv2.LINE_AA)


def save_new_face(face_crop: np.ndarray, 
                  bbox: Tuple[int,int,int,int],
                  confidence: float,
                  quality: float,
                  full_frame: np.ndarray,
                  db: Dict[str, Any]) -> str:
    """
    Simpan wajah baru ke database.
    Returns face_id.
    """
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    face_id = f"face_{timestamp_str}"
    
    #? Simpan potongan wajah
    face_dir = os.path.join(OUTPUT_DIR, "faces")
    os.makedirs(face_dir, exist_ok=True)
    face_path = os.path.join(face_dir, f"{face_id}.jpg")
    cv2.imwrite(face_path, face_crop)
    
    #? Simpan full frame dengan annotasi
    frame_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = os.path.join(frame_dir, f"{face_id}_frame.jpg")
    cv2.imwrite(frame_path, full_frame)
    
    #? Simpan metadata
    face_record = {
        "face_id": face_id,
        "timestamp": timestamp.isoformat(),
        "confidence": confidence,
        "quality": quality,
        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
        "face_image_path": face_path,
        "frame_image_path": frame_path
    }
    
    db["faces"].append(face_record)
    db["total_detected"] = len(db["faces"])
    save_detected_faces_db(db)
    
    print(f"[SAVED] Wajah baru: {face_id} (quality: {quality:.1f})")
    return face_id


def replace_face(face_crop: np.ndarray,
                bbox: Tuple[int,int,int,int],
                confidence: float,
                quality: float,
                full_frame: np.ndarray,
                db: Dict[str, Any],
                existing_face_id: str) -> bool:
    """
    Replace existing face dengan yang lebih berkualitas.
    Returns True jika berhasil replace.
    """
    for face_record in db["faces"]:
        if face_record["face_id"] == existing_face_id:
            #? Hapus file lama
            old_face_path = face_record.get("face_image_path")
            old_frame_path = face_record.get("frame_image_path")
            
            if old_face_path and os.path.exists(old_face_path):
                os.remove(old_face_path)
            if old_frame_path and os.path.exists(old_frame_path):
                os.remove(old_frame_path)
            
            #? Simpan file baru dengan nama yang sama
            face_dir = os.path.join(OUTPUT_DIR, "faces")
            frame_dir = os.path.join(OUTPUT_DIR, "frames")
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(frame_dir, exist_ok=True)
            
            face_path = os.path.join(face_dir, f"{existing_face_id}.jpg")
            frame_path = os.path.join(frame_dir, f"{existing_face_id}_frame.jpg")
            
            cv2.imwrite(face_path, face_crop)
            cv2.imwrite(frame_path, full_frame)
            
            #? Update metadata
            face_record["timestamp"] = datetime.now().isoformat()
            face_record["confidence"] = confidence
            face_record["quality"] = quality
            face_record["bbox"] = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
            face_record["face_image_path"] = face_path
            face_record["frame_image_path"] = frame_path
            
            save_detected_faces_db(db)
            
            old_quality = face_record.get("quality", 0)
            print(f"[REPLACED] {existing_face_id}: quality {old_quality:.1f} → {quality:.1f}")
            return True
    
    return False


def run_ipcam_detection(stream_url: str,
                       face_model_path: str,
                       fallback_model_path: str,
                       conf_thresh: float,
                       iou_thresh: float,
                       min_area_ratio: float,
                       similarity_threshold: float,
                       frame_skip: int,
                       display: bool = True):
    """
    Jalankan deteksi wajah real-time dari IP Camera stream.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    #? Muat model
    face_model = load_model_with_fallback(face_model_path, fallback_model_path)
    
    #? Muat database
    db = load_detected_faces_db()
    print(f"[INFO] Database loaded: {len(db['faces'])} wajah tersimpan")
    
    #? Buka stream
    print(f"[INFO] Membuka stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka stream: {stream_url}")
    
    #? info stream
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Stream terbuka: {width}x{height} @ {fps:.1f} FPS")
    print(f"[INFO] Memproses setiap frame ke-{frame_skip}")
    print(f"[INFO] Tekan 'q' untuk keluar, 's' untuk screenshot\n")
    
    frame_count = 0
    detection_count = 0
    new_face_count = 0
    replaced_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Tidak dapat membaca frame, mencoba reconnect...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            
            frame_count += 1
            
            #? Skip frames untuk performa
            if frame_count % frame_skip != 0:
                if display:
                    cv2.imshow('IP Camera - Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            orig_h, orig_w = frame.shape[:2]
            display_frame = frame.copy()
            
            #? Jalankan deteksi
            results = face_model.predict(
                source=frame, 
                conf=conf_thresh, 
                iou=iou_thresh, 
                verbose=False, 
                device='cpu'
            )
            
            faces_in_frame = []
            
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                cls_idxs = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                
                for bbox, conf, cls_idx in zip(xyxy, confs, cls_idxs):
                    x1, y1, x2, y2 = bbox_to_int(bbox)
                    
                    #? Validasi deteksi
                    is_valid, reason = is_valid_face_detection(
                        (x1, y1, x2, y2), 
                        float(conf),
                        orig_w, 
                        orig_h,
                        min_area_ratio
                    )
                    
                    if not is_valid:
                        continue
                    
                    detection_count += 1
                    
                    #? Crop face
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(orig_w, x2), min(orig_h, y2)
                    face_crop = frame[y1c:y2c, x1c:x2c]
                    
                    if face_crop.size == 0:
                        continue
                    
                    #? Hitung kualitas
                    quality = calculate_image_quality(face_crop)
                    
                    #? Periksa apakah wajah sudah ada
                    is_match, existing_face_id, similarity, existing_quality = find_matching_face(
                        face_crop, db, similarity_threshold
                    )
                    
                    if is_match:
                        #? Wajah sudah ada di database
                        #? Cek apakah kualitas baru lebih baik
                        if quality > existing_quality:
                            #? Replace dengan yang lebih jelas
                            replace_face(
                                face_crop,
                                (x1, y1, x2, y2),
                                float(conf),
                                quality,
                                display_frame,
                                db,
                                existing_face_id
                            )
                            replaced_count += 1
                            label = f"{existing_face_id[:12]} (UPGRADED)"
                            color = (0, 255, 255)  # Yellow - upgraded
                        else:
                            #? Kualitas lebih rendah, skip
                            label = f"{existing_face_id[:12]} (Q:{quality:.0f})"
                            color = (255, 128, 0)  # Orange - known face
                    else:
                        #? Wajah baru, simpan
                        face_id = save_new_face(
                            face_crop, 
                            (x1, y1, x2, y2),
                            float(conf),
                            quality,
                            display_frame,
                            db
                        )
                        new_face_count += 1
                        label = f"NEW: {face_id[:12]} (Q:{quality:.0f})"
                        color = (0, 255, 0)  # Green - new face
                    
                    draw_box(display_frame, (x1, y1, x2, y2), label, color)
                    faces_in_frame.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(conf),
                        "quality": quality
                    })
            
            #? Display info
            info_text = [
                f"Frame: {frame_count}",
                f"Deteksi: {detection_count}",
                f"Wajah baru: {new_face_count}",
                f"Upgraded: {replaced_count}",
                f"Total DB: {len(db['faces'])}",
                f"Wajah di frame: {len(faces_in_frame)}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(display_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            #? Tampilkan frame
            if display:
                cv2.imshow('IP Camera - Face Detection', display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n[INFO] Keluar...")
                    break
                elif key == ord('s'):
                    #? Screenshot
                    screenshot_path = os.path.join(OUTPUT_DIR, 
                        f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"[INFO] Screenshot disimpan: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh user")
    
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\n{'='*50}")
        print(f"STATISTIK AKHIR:")
        print(f"  Total frame diproses: {frame_count}")
        print(f"  Total deteksi wajah: {detection_count}")
        print(f"  Wajah baru tersimpan: {new_face_count}")
        print(f"  Wajah di-upgrade: {replaced_count}")
        print(f"  Total wajah di database: {len(db['faces'])}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='IP Camera Face Detection - Deteksi wajah real-time dari IP Camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:

  # RTSP stream
  python deteksiwajah_ipcam.py rtsp://admin:password@192.168.1.100:554/stream
  
  # HTTP stream
  python deteksiwajah_ipcam.py http://192.168.1.100:8080/video
  
  # Webcam lokal (0 = default camera)
  python deteksiwajah_ipcam.py 0
  
  # Dengan parameter kustom
  python deteksiwajah_ipcam.py rtsp://192.168.1.100:554/stream --conf 0.7 --similarity 0.75
  
  # Tanpa display (headless server)
  python deteksiwajah_ipcam.py rtsp://192.168.1.100:554/stream --no-display

Format URL yang didukung:
  - RTSP: rtsp://[user:pass@]ip:port/path
  - HTTP: http://ip:port/path
  - RTMP: rtmp://ip:port/path
  - Webcam: 0, 1, 2, ...

Sistem akan:
  - Menyimpan HANYA wajah baru yang belum pernah terdeteksi
  - Meng-upgrade foto wajah lama jika menemukan yang lebih jelas
  - Tidak menyimpan duplikat dengan pose/ekspresi berbeda
        """
    )
    
    parser.add_argument('stream_url', 
                       help='URL IP Camera stream atau nomor webcam (0, 1, 2, ...)')
    
    parser.add_argument('--model', '-m',
                       default=DEFAULT_FACE_MODEL,
                       help=f'Path ke model deteksi wajah (default: {DEFAULT_FACE_MODEL})')
    
    parser.add_argument('--fallback',
                       default=DEFAULT_FALLBACK_MODEL,
                       help=f'Path ke fallback model (default: {DEFAULT_FALLBACK_MODEL})')
    
    parser.add_argument('--conf', '-c',
                       type=float,
                       default=DEFAULT_CONFIDENCE,
                       help=f'Confidence threshold (default: {DEFAULT_CONFIDENCE})')
    
    parser.add_argument('--iou',
                       type=float,
                       default=DEFAULT_IOU,
                       help=f'IOU threshold (default: {DEFAULT_IOU})')
    
    parser.add_argument('--min-area',
                       type=float,
                       default=DEFAULT_MIN_FACE_AREA_RATIO,
                       help=f'Minimum face area ratio (default: {DEFAULT_MIN_FACE_AREA_RATIO})')
    
    parser.add_argument('--similarity',
                       type=float,
                       default=DEFAULT_SIMILARITY_THRESHOLD,
                       help=f'Face similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})')
    
    parser.add_argument('--frame-skip',
                       type=int,
                       default=DEFAULT_FRAME_SKIP,
                       help=f'Process setiap N frame (default: {DEFAULT_FRAME_SKIP})')
    
    parser.add_argument('--no-display',
                       action='store_true',
                       help='Nonaktifkan display window (untuk headless server)')
    
    args = parser.parse_args()
    
    #? Handle nomor webcam
    stream_url = args.stream_url
    if stream_url.isdigit():
        stream_url = int(stream_url)
    
    try:
        run_ipcam_detection(
            stream_url=stream_url,
            face_model_path=args.model,
            fallback_model_path=args.fallback,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            min_area_ratio=args.min_area,
            similarity_threshold=args.similarity,
            frame_skip=args.frame_skip,
            display=not args.no_display
        )
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()