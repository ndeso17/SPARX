import os
import sys
import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

# Optional libs
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False
    print("[INFO] Tesseract not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False
    print("[INFO] EasyOCR not available")

# ========== CONFIG ==========
DEFAULT_PLATE_MODEL = "data/plat_nomor.pt"
DEFAULT_FALLBACK_MODEL = "data/yolov8n.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_MIN_PLATE_AREA_RATIO = 0.001
DEFAULT_MAX_PLATE_AREA_RATIO = 0.3
OUTPUT_DIR = "output/tnkb"
KODE_PLAT_PATH = "data/kode_plat.json"
TNKB_DATABASE_PATH = "data/tnkb.json"

# Global data
VALID_PLATE_CODES = set()
PLATE_CODE_INFO = {}
TNKB_DATABASE = []

# ========== LOAD VALIDATION DATA ==========
def load_kode_plat():
    global VALID_PLATE_CODES, PLATE_CODE_INFO
    if os.path.exists(KODE_PLAT_PATH):
        try:
            with open(KODE_PLAT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for wilayah in data.get('wilayah', []):
                for kode_data in wilayah.get('kode_plat', []):
                    kode = kode_data['kode']
                    VALID_PLATE_CODES.add(kode)
                    PLATE_CODE_INFO[kode] = {
                        'wilayah': wilayah['nama'],
                        'keterangan': kode_data['keterangan']
                    }
            print(f"[INFO] Loaded {len(VALID_PLATE_CODES)} valid plate codes")
        except Exception as e:
            print(f"[WARN] Failed to load kode_plat.json: {e}")

def load_tnkb_database():
    global TNKB_DATABASE
    if os.path.exists(TNKB_DATABASE_PATH):
        try:
            with open(TNKB_DATABASE_PATH, 'r', encoding='utf-8') as f:
                TNKB_DATABASE = json.load(f)
            print(f"[INFO] Loaded {len(TNKB_DATABASE)} TNKB records")
            for rec in TNKB_DATABASE[:5]:  # Show first 5
                print(f"  - {rec.get('nopol', 'N/A')}")
        except Exception as e:
            print(f"[INFO] Could not load TNKB database: {e}")

load_kode_plat()
load_tnkb_database()

# ========== IMAGE PREPROCESSING ==========
def preprocess_plate_complete(plate_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Complete preprocessing optimized for Indonesian license plates.
    Returns: (binary_image, grayscale_enhanced, info_dict)
    """
    info = {"original_size": plate_img.shape[:2]}
    
    # Convert to grayscale
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    h, w = gray.shape
    
    # Resize if too small (important for OCR)
    if h < 80 or w < 200:
        scale = max(80/h, 200/w, 2.0)
        new_h, new_w = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape
        info["resized"] = True
        info["scale"] = scale
    
    info["processing_size"] = (h, w)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Multiple thresholding approaches
    # Method 1: Otsu
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )
    
    # Combine: use adaptive as base
    binary = adaptive.copy()
    
    # Check if we need to invert (text should be dark on light background for processing)
    # Count white pixels
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.6:  # Too much white, likely inverted
        binary = cv2.bitwise_not(binary)
        info["inverted"] = True
    
    # Clean up with morphology
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    return binary, enhanced, info

def segment_characters_improved(binary_img: np.ndarray, gray_img: np.ndarray) -> List[Tuple[np.ndarray, Tuple]]:
    """
    Improved character segmentation that properly detects all characters.
    Returns list of (char_image, bbox) tuples.
    """
    h, w = binary_img.shape
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[WARN] No contours found in segmentation")
        return []
    
    # Filter and collect character candidates
    char_boxes = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        
        # Filter by size
        if area < 50:  # Too small
            continue
        
        # Height ratio check
        if ch < h * 0.25:  # Too short (less than 25% of plate height)
            continue
        
        # Minimum dimensions
        if cw < 8 or ch < 15:
            continue
        
        # Aspect ratio check (characters are taller than wide)
        aspect_ratio = cw / ch if ch > 0 else 0
        if aspect_ratio > 2.0:  # Too wide
            continue
        
        char_boxes.append((x, y, cw, ch))
    
    # Sort by x position (left to right)
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    
    # Merge overlapping boxes
    merged_boxes = []
    if char_boxes:
        current = list(char_boxes[0])
        
        for i in range(1, len(char_boxes)):
            x, y, cw, ch = char_boxes[i]
            cx, cy, ccw, cch = current
            
            # Check if overlapping or very close
            if x <= cx + ccw + 5:  # Overlapping or close
                # Merge
                new_x = min(cx, x)
                new_y = min(cy, y)
                new_x2 = max(cx + ccw, x + cw)
                new_y2 = max(cy + cch, y + ch)
                current = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            else:
                merged_boxes.append(tuple(current))
                current = list(char_boxes[i])
        
        merged_boxes.append(tuple(current))
    
    # Extract character images with padding
    char_images = []
    
    for (x, y, cw, ch) in merged_boxes:
        # Add padding
        pad_x = max(3, int(cw * 0.15))
        pad_y = max(3, int(ch * 0.15))
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
        
        # Extract from grayscale (better for OCR)
        char_crop = gray_img[y1:y2, x1:x2]
        
        if char_crop.size > 0:
            char_images.append((char_crop, (x1, y1, x2, y2)))
    
    print(f"[SEG] Found {len(char_images)} character segments")
    
    return char_images

# ========== OCR FUNCTIONS ==========
def ocr_with_tesseract_complete(gray_img: np.ndarray, binary_img: np.ndarray) -> Dict[str, Any]:
    """Complete OCR using Tesseract with multiple attempts."""
    if not TESSERACT_AVAILABLE:
        return {"success": False, "error": "Tesseract not available"}
    
    results = []
    
    # Config variations for better detection
    configs = [
        '--psm 7 --oem 3',  # Single line, LSTM
        '--psm 8 --oem 3',  # Single word
        '--psm 6 --oem 3',  # Uniform block
        '--psm 7 --oem 1',  # Single line, legacy
    ]
    
    images_to_try = [
        ("binary", binary_img),
        ("grayscale", gray_img),
    ]
    
    for img_name, img in images_to_try:
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config, lang='eng')
                text = text.strip()
                
                if text and len(text) >= 3:
                    # Get confidence
                    try:
                        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                        confs = [int(c) for c in data['conf'] if c != '-1' and int(c) >= 0]
                        avg_conf = sum(confs) / len(confs) if confs else 0
                    except:
                        avg_conf = 0
                    
                    results.append({
                        "text": text,
                        "confidence": avg_conf,
                        "method": f"tesseract_{img_name}",
                        "config": config
                    })
            except Exception as e:
                continue
    
    if results:
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        best = results[0]
        return {"success": True, "text": best["text"], "confidence": best["confidence"], "all_results": results}
    
    return {"success": False, "error": "No text detected"}

def ocr_with_easyocr_complete(plate_img_color: np.ndarray, reader) -> Dict[str, Any]:
    """Complete OCR using EasyOCR."""
    if not reader:
        return {"success": False, "error": "EasyOCR reader not available"}
    
    try:
        results = reader.readtext(
            plate_img_color,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False,
            detail=1
        )
        
        if not results:
            return {"success": False, "error": "No text detected"}
        
        # Combine all detected text from left to right
        results_sorted = sorted(results, key=lambda x: x[0][0][0])  # Sort by x position
        
        full_text = ""
        confidences = []
        
        for bbox, text, conf in results_sorted:
            full_text += text
            confidences.append(conf)
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "success": True,
            "text": full_text,
            "confidence": avg_conf * 100,
            "segments": len(results),
            "all_results": [{"text": r[1], "conf": r[2] * 100} for r in results]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ========== PLATE VALIDATION & CORRECTION ==========
def clean_ocr_text(text: str) -> str:
    """Clean OCR output."""
    if not text:
        return ""
    
    # Remove non-alphanumeric
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    return text

def correct_with_context(text: str) -> str:
    """Apply context-aware corrections for Indonesian plates."""
    if not text or len(text) < 6:
        return text
    
    chars = list(text)
    
    # Corrections based on position
    for i in range(len(chars)):
        # Prefix (first 1-2 chars): should be letters
        if i <= 1:
            if chars[i] == '0':
                chars[i] = 'O'
            elif chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '5':
                chars[i] = 'S'
            elif chars[i] == '8':
                chars[i] = 'B'
        
        # Number section (chars 2-5): should be digits
        elif i >= 2 and i <= 5:
            if chars[i] == 'O':
                chars[i] = '0'
            elif chars[i] == 'I' or chars[i] == 'L':
                chars[i] = '1'
            elif chars[i] == 'S':
                chars[i] = '5'
            elif chars[i] == 'B':
                chars[i] = '8'
            elif chars[i] == 'Z':
                chars[i] = '2'
            elif chars[i] == 'G':
                chars[i] = '6'
        
        # Suffix (chars 6+): should be letters
        elif i >= 6:
            if chars[i] == '0':
                chars[i] = 'O'
            elif chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '5':
                chars[i] = 'S'
    
    corrected = ''.join(chars)
    
    return corrected

def find_database_match(ocr_text: str) -> Optional[Dict[str, Any]]:
    """Find best match from database."""
    if not ocr_text or not TNKB_DATABASE:
        return None
    
    ocr_clean = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())
    
    best_match = None
    best_score = 0
    
    for record in TNKB_DATABASE:
        db_plate = record.get('nopol', '').replace(' ', '').upper()
        if not db_plate:
            continue
        
        # Simple character match score
        matches = sum(1 for a, b in zip(ocr_clean, db_plate) if a == b)
        max_len = max(len(ocr_clean), len(db_plate))
        score = matches / max_len if max_len > 0 else 0
        
        if score > best_score and score > 0.6:
            best_score = score
            best_match = {
                "plate": db_plate,
                "score": score,
                "record": record
            }
    
    return best_match

def format_plate_standard(text: str) -> str:
    """Format plate with spaces: XX 1234 XXX."""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(text) < 6:
        return text
    
    # Detect pattern
    # Find where numbers start
    num_start = -1
    for i, c in enumerate(text):
        if c.isdigit():
            num_start = i
            break
    
    if num_start < 0:
        return text
    
    # Find where numbers end
    num_end = num_start
    for i in range(num_start, len(text)):
        if text[i].isalpha():
            num_end = i
            break
        num_end = i + 1
    
    prefix = text[:num_start]
    number = text[num_start:num_end]
    suffix = text[num_end:]
    
    # Format
    parts = [prefix, number, suffix]
    formatted = ' '.join(p for p in parts if p)
    
    return formatted

def validate_plate_complete(ocr_text: str) -> Dict[str, Any]:
    """Complete validation with database matching."""
    
    if not ocr_text:
        return {"valid": False, "reason": "empty text"}
    
    # Clean text
    cleaned = clean_ocr_text(ocr_text)
    
    # Apply context corrections
    corrected = correct_with_context(cleaned)
    
    # Try to match with database first
    db_match = find_database_match(corrected)
    
    if db_match:
        db_plate = db_match["plate"]
        formatted = format_plate_standard(db_plate)
        
        # Extract parts
        parts = formatted.split()
        if len(parts) >= 2:
            prefix = parts[0]
            number = parts[1]
            suffix = parts[2] if len(parts) > 2 else ""
            
            prefix_valid = prefix in VALID_PLATE_CODES
            
            result = {
                "valid": True,
                "original_ocr": ocr_text,
                "cleaned": cleaned,
                "corrected": corrected,
                "final": formatted,
                "prefix": prefix,
                "number": number,
                "suffix": suffix,
                "prefix_valid": prefix_valid,
                "source": "database_match",
                "match_score": db_match["score"],
                "record": db_match["record"]
            }
            
            if prefix_valid and prefix in PLATE_CODE_INFO:
                result.update(PLATE_CODE_INFO[prefix])
            
            return result
    
    # Manual validation
    formatted = format_plate_standard(corrected)
    parts = formatted.split()
    
    if len(parts) >= 2:
        prefix = parts[0]
        number = parts[1]
        suffix = parts[2] if len(parts) > 2 else ""
        
        # Validate format
        if not prefix.isalpha():
            return {"valid": False, "reason": "prefix not alpha", "text": formatted}
        
        if not number.isdigit():
            return {"valid": False, "reason": "number not digit", "text": formatted}
        
        if suffix and not suffix.isalpha():
            return {"valid": False, "reason": "suffix not alpha", "text": formatted}
        
        # Check number validity
        if len(number) == 4 and number[0] == '0':
            return {"valid": False, "reason": "4-digit number starts with 0", "text": formatted}
        
        prefix_valid = prefix in VALID_PLATE_CODES
        
        result = {
            "valid": True,
            "original_ocr": ocr_text,
            "cleaned": cleaned,
            "corrected": corrected,
            "final": formatted,
            "prefix": prefix,
            "number": number,
            "suffix": suffix,
            "prefix_valid": prefix_valid,
            "source": "manual_validation"
        }
        
        if prefix_valid and prefix in PLATE_CODE_INFO:
            result.update(PLATE_CODE_INFO[prefix])
        
        return result
    
    return {"valid": False, "reason": "invalid format", "text": formatted}

# ========== MAIN PIPELINE ==========
def perform_complete_ocr(plate_img: np.ndarray, easyocr_reader=None) -> Dict[str, Any]:
    """Perform complete OCR pipeline."""
    
    result = {
        "success": False,
        "engine": "none",
        "raw_text": "",
        "validation": {}
    }
    
    # Preprocess
    binary, gray, prep_info = preprocess_plate_complete(plate_img)
    result["preprocessing"] = prep_info
    
    # Segment characters for debugging
    char_segments = segment_characters_improved(binary, gray)
    result["segments_found"] = len(char_segments)
    
    # Try EasyOCR first (best for Indonesian plates)
    if easyocr_reader:
        print("[OCR] Trying EasyOCR...")
        easy_result = ocr_with_easyocr_complete(plate_img, easyocr_reader)
        
        if easy_result.get("success"):
            raw_text = easy_result["text"]
            print(f"[OCR] EasyOCR result: '{raw_text}'")
            
            # Validate
            validation = validate_plate_complete(raw_text)
            
            result.update({
                "success": True,
                "engine": "easyocr",
                "raw_text": raw_text,
                "confidence": easy_result.get("confidence", 0),
                "validation": validation,
                "easy_details": easy_result
            })
            
            if validation.get("valid"):
                return result
    
    # Try Tesseract as fallback
    if TESSERACT_AVAILABLE:
        print("[OCR] Trying Tesseract...")
        tess_result = ocr_with_tesseract_complete(gray, binary)
        
        if tess_result.get("success"):
            raw_text = tess_result["text"]
            print(f"[OCR] Tesseract result: '{raw_text}'")
            
            # Validate
            validation = validate_plate_complete(raw_text)
            
            result.update({
                "success": True,
                "engine": "tesseract",
                "raw_text": raw_text,
                "confidence": tess_result.get("confidence", 0),
                "validation": validation,
                "tess_details": tess_result
            })
            
            if validation.get("valid"):
                return result
    
    return result

# ========== DETECTION PIPELINE ==========
def load_model_with_fallback(path: str, fallback: str):
    if os.path.isfile(path):
        print(f"[INFO] Loading model: {path}")
        return YOLO(path)
    elif os.path.isfile(fallback):
        print(f"[WARN] Using fallback: {fallback}")
        return YOLO(fallback)
    else:
        raise FileNotFoundError(f"Model not found: {path}")

def bbox_to_int(bbox) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bbox
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def is_valid_detection(bbox, conf, w, h, min_ratio, max_ratio):
    x1, y1, x2, y2 = bbox
    area = max(0, x2-x1) * max(0, y2-y1)
    img_area = w * h
    ratio = area / img_area if img_area > 0 else 0
    
    if ratio < min_ratio or ratio > max_ratio:
        return False, f"area ratio {ratio:.4f}"
    
    bw = x2 - x1
    bh = y2 - y1
    
    if bw < 30 or bh < 10:
        return False, "too small"
    
    ar = bw / bh if bh > 0 else 0
    if ar < 1.4 or ar > 8.0:
        return False, f"aspect ratio {ar:.2f}"
    
    return True, "OK"

def draw_box(img, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1-22), (x1+w+8, y1), color, -1)
    cv2.putText(img, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def run_detection(image_path: str, model_path: str, fallback_path: str,
                  conf: float, iou: float, min_area: float, max_area: float,
                  save_crops: bool = True):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    h, w = img.shape[:2]
    annotated = img.copy()
    
    # Load model
    model = load_model_with_fallback(model_path, fallback_path)
    
    # Initialize EasyOCR
    easy_reader = None
    if EASYOCR_AVAILABLE:
        print("[INFO] Initializing EasyOCR...")
        easy_reader = easyocr.Reader(['en'], gpu=False)
    
    # Run detection
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False, device='cpu')
    
    plates_data = []
    plate_count = 0
    
    crops_dir = os.path.join(OUTPUT_DIR, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        
        for bbox, det_conf in zip(xyxy, confs):
            x1, y1, x2, y2 = bbox_to_int(bbox)
            
            valid, reason = is_valid_detection((x1,y1,x2,y2), float(det_conf), w, h, min_area, max_area)
            
            if not valid:
                continue
            
            plate_count += 1
            print(f"\n[PLATE #{plate_count}] Detected at [{x1},{y1},{x2},{y2}]")
            
            # Extract crop
            plate_crop = img[y1:y2, x1:x2]
            
            # Perform OCR
            ocr_result = perform_complete_ocr(plate_crop, easy_reader)
            
            # Get validated plate text
            validation = ocr_result.get("validation", {})
            final_text = validation.get("final", ocr_result.get("raw_text", "N/A"))
            
            # Determine color
            if validation.get("valid") and validation.get("prefix_valid"):
                color = (0, 255, 0)  # Green
                prefix_symbol = "✓"
            elif validation.get("valid"):
                color = (0, 255, 255)  # Yellow
                prefix_symbol = "?"
            else:
                color = (0, 165, 255)  # Orange
                prefix_symbol = "!"
            
            label = f"{prefix_symbol} {final_text}"
            draw_box(annotated, (x1, y1, x2, y2), label, color)
            
            # Save crops
            if save_crops:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
                crop_path = os.path.join(crops_dir, f"plate{plate_count}_{ts}.jpg")
                cv2.imwrite(crop_path, plate_crop)
                
                # Save preprocessed
                binary, gray, _ = preprocess_plate_complete(plate_crop)
                prep_path = os.path.join(crops_dir, f"plate{plate_count}_{ts}_prep.jpg")
                cv2.imwrite(prep_path, binary)
            
            # Store data
            plate_data = {
                "plate_number": plate_count,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "detection_confidence": float(det_conf),
                "ocr": ocr_result
            }
            
            plates_data.append(plate_data)
            
            # Console output
            print(f"  Detection conf: {det_conf:.3f}")
            print(f"  OCR engine: {ocr_result.get('engine', 'none')}")
            print(f"  OCR raw: '{ocr_result.get('raw_text', '')}'")
            print(f"  Final text: '{final_text}'")
            print(f"  Valid: {validation.get('valid', False)}")
            
            if validation.get('valid'):
                print(f"  Prefix: {validation.get('prefix')} ({'VALID' if validation.get('prefix_valid') else 'UNKNOWN'})")
                if validation.get('prefix_valid'):
                    print(f"  Wilayah: {validation.get('wilayah', 'N/A')}")
                print(f"  Number: {validation.get('number')}")
                print(f"  Suffix: {validation.get('suffix')}")
                if validation.get('source') == 'database_match':
                    print(f"  Database match: {validation.get('match_score', 0):.2f}")
    
    # Save outputs
    basename = Path(image_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    ann_path = os.path.join(OUTPUT_DIR, f"{basename}_{ts}_result.jpg")
    cv2.imwrite(ann_path, annotated)
    
    json_path = os.path.join(OUTPUT_DIR, f"{basename}_{ts}_result.json")
    
    summary = {
        "image_path": image_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_size": {"width": w, "height": h},
        "parameters": {
            "confidence": conf,
            "iou": iou,
            "min_area_ratio": min_area,
            "max_area_ratio": max_area
        },
        "plates": plates_data,
        "total_detected": len(plates_data)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"DETECTION COMPLETE")
    print(f"  Total plates: {len(plates_data)}")
    print(f"  Annotated image: {ann_path}")
    print(f"  JSON result: {json_path}")
    print(f"{'='*60}\n")
    
    return ann_path, json_path, summary

# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(
        description="Indonesian License Plate Detection with Complete OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deteksitnkb.py plate.jpg
  python deteksitnkb.py plate.jpg --conf 0.3
  python deteksitnkb.py plate.jpg --no-crops

Features:
  - Complete OCR (prefix + number + suffix)
  - Database matching and correction
  - Validation with kode_plat.json
  - Enhanced preprocessing for Indonesian plates
  - Multiple OCR engines (EasyOCR + Tesseract)
        """
    )
    
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--fallback", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_PLATE_AREA_RATIO)
    parser.add_argument("--max-area", type=float, default=DEFAULT_MAX_PLATE_AREA_RATIO)
    parser.add_argument("--no-crops", action="store_true")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print(f"[ERROR] File not found: {args.image}")
        sys.exit(1)
    
    try:
        run_detection(
            args.image,
            args.model,
            args.fallback,
            args.conf,
            args.iou,
            args.min_area,
            args.max_area,
            save_crops=not args.no_crops
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()