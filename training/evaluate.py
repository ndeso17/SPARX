#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Features:
- YOLO detection performance (mAP, precision, recall)
- OCR accuracy evaluation (character-level and plate-level)
- End-to-end pipeline evaluation
- Performance benchmarking (speed, FPS)
- Error analysis and visualization
- Comparison with ground truth database
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# YOLO
from ultralytics import YOLO

# OCR
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False
    print("[WARN] EasyOCR not available")

# ========== CONFIGURATION ==========
DEFAULT_YOLO_MODEL = "data/runs/detect/plat_nomor/weights/best.pt"
DEFAULT_OCR_MODEL = "data/char_all_model.keras"
DEFAULT_TEST_DIR = "data/PlatNomor/test"
DEFAULT_OUTPUT_DIR = "output/evaluation"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45

# ========== METRICS CALCULATION ==========
class DetectionMetrics:
    """Calculate detection metrics (mAP, precision, recall)"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.ious = []
        
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, pred_boxes: List[Tuple], gt_boxes: List[Tuple]):
        """Update metrics with prediction and ground truth"""
        matched_gt = set()
        
        for pred_box in pred_boxes:
            matched = False
            best_iou = 0
            
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    
                if iou >= self.iou_threshold:
                    matched = True
                    matched_gt.add(idx)
                    self.tp += 1
                    self.ious.append(iou)
                    break
            
            if not matched:
                self.fp += 1
        
        # False negatives
        self.fn += len(gt_boxes) - len(matched_gt)
    
    def get_metrics(self) -> Dict:
        """Calculate final metrics"""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = np.mean(self.ious) if self.ious else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_iou': avg_iou,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn
        }


class OCRMetrics:
    """Calculate OCR accuracy metrics"""
    
    def __init__(self):
        self.total_chars = 0
        self.correct_chars = 0
        self.total_plates = 0
        self.correct_plates = 0
        self.char_errors = defaultdict(int)
        self.levenshtein_distances = []
        
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        
        return distances[-1]
    
    def update(self, pred_text: str, gt_text: str):
        """Update metrics with prediction and ground truth"""
        pred_clean = pred_text.replace(' ', '').upper()
        gt_clean = gt_text.replace(' ', '').upper()
        
        # Character-level accuracy
        min_len = min(len(pred_clean), len(gt_clean))
        for i in range(min_len):
            self.total_chars += 1
            if pred_clean[i] == gt_clean[i]:
                self.correct_chars += 1
            else:
                self.char_errors[f"{gt_clean[i]}->{pred_clean[i]}"] += 1
        
        # Handle length differences
        self.total_chars += abs(len(pred_clean) - len(gt_clean))
        
        # Plate-level accuracy
        self.total_plates += 1
        if pred_clean == gt_clean:
            self.correct_plates += 1
        
        # Levenshtein distance
        distance = self.levenshtein_distance(pred_clean, gt_clean)
        self.levenshtein_distances.append(distance)
    
    def get_metrics(self) -> Dict:
        """Calculate final metrics"""
        char_accuracy = self.correct_chars / self.total_chars if self.total_chars > 0 else 0
        plate_accuracy = self.correct_plates / self.total_plates if self.total_plates > 0 else 0
        avg_levenshtein = np.mean(self.levenshtein_distances) if self.levenshtein_distances else 0
        
        # Top error pairs
        top_errors = sorted(self.char_errors.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'char_accuracy': char_accuracy,
            'plate_accuracy': plate_accuracy,
            'avg_levenshtein': avg_levenshtein,
            'total_chars': self.total_chars,
            'correct_chars': self.correct_chars,
            'total_plates': self.total_plates,
            'correct_plates': self.correct_plates,
            'top_errors': top_errors,
            'avg_inference_time': 0  # Will be set by evaluator
        }


# ========== EVALUATORS ==========
class YOLOEvaluator:
    """Evaluate YOLO detection model"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.metrics = DetectionMetrics()
        self.inference_times = []
        
    def evaluate_dataset(self, test_dir: str) -> Dict:
        """Evaluate on test dataset"""
        print("\n" + "="*80)
        print("EVALUATING YOLO DETECTION")
        print("="*80)
        
        test_path = Path(test_dir)
        images_dir = test_path / 'images'
        labels_dir = test_path / 'labels'
        
        if not images_dir.exists():
            print(f"[ERROR] Images directory not found: {images_dir}")
            return {}
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.jpeg')) + \
                     list(images_dir.glob('*.png'))
        
        print(f"Found {len(image_files)} test images")
        
        results = []
        
        for img_path in tqdm(image_files, desc="Processing"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Get ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_boxes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, x_center, y_center, width, height = map(float, parts)
                            
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            gt_boxes.append((x1, y1, x2, y2))
            
            # Run detection
            start_time = time.time()
            detections = self.model.predict(img, conf=self.conf_threshold, verbose=False)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Extract predicted boxes
            pred_boxes = []
            for result in detections:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    pred_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            # Update metrics
            self.metrics.update(pred_boxes, gt_boxes)
            
            results.append({
                'image': img_path.name,
                'gt_boxes': len(gt_boxes),
                'pred_boxes': len(pred_boxes),
                'inference_time': inference_time
            })
        
        # Calculate metrics
        metrics = self.metrics.get_metrics()
        metrics['avg_inference_time'] = np.mean(self.inference_times)
        metrics['fps'] = 1.0 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
        
        print("\n" + "-"*80)
        print("YOLO DETECTION METRICS")
        print("-"*80)
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1 Score:     {metrics['f1_score']:.4f}")
        print(f"Avg IoU:      {metrics['avg_iou']:.4f}")
        print(f"True Pos:     {metrics['tp']}")
        print(f"False Pos:    {metrics['fp']}")
        print(f"False Neg:    {metrics['fn']}")
        print(f"Avg Time:     {metrics['avg_inference_time']*1000:.2f} ms")
        print(f"FPS:          {metrics['fps']:.2f}")
        print("-"*80)
        
        return {
            'metrics': metrics,
            'results': results
        }


class OCREvaluator:
    """Evaluate OCR model"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.easy_reader = None
        
        if model_path and TF_AVAILABLE:
            print(f"[INFO] Loading OCR model: {model_path}")
            try:
                self.model = keras.models.load_model(model_path)
            except Exception as e:
                print(f"[WARN] Could not load Keras model: {e}")
        
        if EASYOCR_AVAILABLE:
            print("[INFO] Initializing EasyOCR...")
            self.easy_reader = easyocr.Reader(['en'], gpu=False)
        
        self.metrics = OCRMetrics()
        self.inference_times = []
    
    def evaluate_with_ground_truth(self, test_data: List[Dict]) -> Dict:
        """Evaluate OCR with ground truth labels"""
        print("\n" + "="*80)
        print("EVALUATING OCR ACCURACY")
        print("="*80)
        
        if not self.easy_reader:
            print("[ERROR] EasyOCR not available")
            return {}
        
        results = []
        
        for item in tqdm(test_data, desc="Processing"):
            img_path = item['image_path']
            gt_text = item['plate_text']
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Run OCR
            start_time = time.time()
            ocr_results = self.easy_reader.readtext(img, detail=0, paragraph=False)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            pred_text = ''.join(ocr_results).replace(' ', '').upper() if ocr_results else ''
            
            # Update metrics
            self.metrics.update(pred_text, gt_text)
            
            results.append({
                'image': Path(img_path).name,
                'gt_text': gt_text,
                'pred_text': pred_text,
                'correct': pred_text == gt_text.replace(' ', '').upper(),
                'inference_time': inference_time
            })
        
        # Calculate metrics
        metrics = self.metrics.get_metrics()
        metrics['avg_inference_time'] = np.mean(self.inference_times) if self.inference_times else 0
        
        print("\n" + "-"*80)
        print("OCR METRICS")
        print("-"*80)
        print(f"Character Accuracy:  {metrics['char_accuracy']*100:.2f}%")
        print(f"Plate Accuracy:      {metrics['plate_accuracy']*100:.2f}%")
        print(f"Avg Levenshtein:     {metrics['avg_levenshtein']:.2f}")
        print(f"Total Plates:        {metrics['total_plates']}")
        print(f"Correct Plates:      {metrics['correct_plates']}")
        print(f"Avg Time:            {metrics['avg_inference_time']*1000:.2f} ms")
        
        if metrics['top_errors']:
            print(f"\nTop Character Errors:")
            for error, count in metrics['top_errors']:
                print(f"  {error}: {count} times")
        
        print("-"*80)
        
        return {
            'metrics': metrics,
            'results': results
        }


class EndToEndEvaluator:
    """Evaluate complete pipeline (detection + OCR)"""
    
    def __init__(self, yolo_model_path: str, ocr_model_path: str = None, conf: float = 0.5):
        self.yolo_eval = YOLOEvaluator(yolo_model_path, conf)
        self.ocr_metrics = OCRMetrics()
        
        # Initialize EasyOCR
        self.easy_reader = None
        if EASYOCR_AVAILABLE:
            print("[INFO] Initializing EasyOCR for end-to-end evaluation...")
            self.easy_reader = easyocr.Reader(['en'], gpu=False)
        
        self.pipeline_times = []
    
    def evaluate_pipeline(self, test_dir: str, ground_truth: Dict) -> Dict:
        """Evaluate complete detection + OCR pipeline"""
        print("\n" + "="*80)
        print("EVALUATING END-TO-END PIPELINE")
        print("="*80)
        
        test_path = Path(test_dir)
        images_dir = test_path / 'images'
        
        if not images_dir.exists():
            print(f"[ERROR] Images directory not found: {images_dir}")
            return {}
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.jpeg')) + \
                     list(images_dir.glob('*.png'))
        
        results = []
        
        for img_path in tqdm(image_files, desc="Processing Pipeline"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Get ground truth
            img_name = img_path.stem
            gt_text = ground_truth.get(img_name, '')
            
            if not gt_text:
                continue
            
            # Step 1: Detect plate
            start_time = time.time()
            
            detections = self.yolo_eval.model.predict(img, conf=self.yolo_eval.conf_threshold, verbose=False)
            
            pred_text = ''
            detected = False
            
            if detections and len(detections[0].boxes) > 0:
                detected = True
                
                # Get first detection
                box = detections[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Extract plate crop
                plate_crop = img[y1:y2, x1:x2]
                
                # Step 2: OCR
                if self.easy_reader and plate_crop.size > 0:
                    ocr_results = self.easy_reader.readtext(plate_crop, detail=0, paragraph=False)
                    pred_text = ''.join(ocr_results).replace(' ', '').upper() if ocr_results else ''
            
            pipeline_time = time.time() - start_time
            self.pipeline_times.append(pipeline_time)
            
            # Update metrics
            if detected and pred_text:
                self.ocr_metrics.update(pred_text, gt_text)
            
            results.append({
                'image': img_path.name,
                'gt_text': gt_text,
                'pred_text': pred_text,
                'detected': detected,
                'correct': pred_text == gt_text.replace(' ', '').upper(),
                'pipeline_time': pipeline_time
            })
        
        # Calculate metrics
        ocr_metrics = self.ocr_metrics.get_metrics()
        avg_pipeline_time = np.mean(self.pipeline_times) if self.pipeline_times else 0
        
        detection_rate = sum(1 for r in results if r['detected']) / len(results) if results else 0
        
        print("\n" + "-"*80)
        print("END-TO-END PIPELINE METRICS")
        print("-"*80)
        print(f"Detection Rate:      {detection_rate*100:.2f}%")
        print(f"Character Accuracy:  {ocr_metrics['char_accuracy']*100:.2f}%")
        print(f"Plate Accuracy:      {ocr_metrics['plate_accuracy']*100:.2f}%")
        print(f"Avg Pipeline Time:   {avg_pipeline_time*1000:.2f} ms")
        print(f"Pipeline FPS:        {1.0/avg_pipeline_time if avg_pipeline_time > 0 else 0:.2f}")
        print("-"*80)
        
        return {
            'detection_rate': detection_rate,
            'ocr_metrics': ocr_metrics,
            'avg_pipeline_time': avg_pipeline_time,
            'results': results
        }


# ========== VISUALIZATION ==========
def visualize_detection_results(results: List[Dict], output_dir: Path):
    """Visualize detection evaluation results"""
    
    # Inference time distribution
    times = [r['inference_time'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(np.array(times) * 1000, bins=30, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Inference Time (ms)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Detection Inference Time Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Predictions per image
    pred_counts = [r['pred_boxes'] for r in results]
    gt_counts = [r['gt_boxes'] for r in results]
    
    axes[1].scatter(gt_counts, pred_counts, alpha=0.6, s=50, color='coral')
    max_val = max(max(gt_counts) if gt_counts else 0, max(pred_counts) if pred_counts else 0)
    if max_val > 0:
        axes[1].plot([0, max_val], [0, max_val], 'k--', label='Perfect Detection')
    axes[1].set_xlabel('Ground Truth Plates', fontsize=12)
    axes[1].set_ylabel('Predicted Plates', fontsize=12)
    axes[1].set_title('Detection Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'detection_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Detection evaluation saved: {save_path}")
    plt.close()


def visualize_ocr_results(results: List[Dict], output_dir: Path):
    """Visualize OCR evaluation results"""
    
    # Accuracy by plate length
    length_acc = defaultdict(list)
    for r in results:
        length = len(r['gt_text'].replace(' ', ''))
        length_acc[length].append(1 if r['correct'] else 0)
    
    lengths = sorted(length_acc.keys())
    accuracies = [np.mean(length_acc[l]) * 100 for l in lengths]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    if lengths and accuracies:
        axes[0].bar(lengths, accuracies, color='mediumseagreen', edgecolor='black')
        axes[0].set_xlabel('Plate Text Length', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('OCR Accuracy by Plate Length', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Inference time
    times = [r['inference_time'] for r in results]
    axes[1].hist(np.array(times) * 1000, bins=30, color='salmon', edgecolor='black')
    axes[1].set_xlabel('OCR Time (ms)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('OCR Inference Time Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'ocr_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ OCR analysis saved: {save_path}")
    plt.close()


def create_comparison_plot(yolo_metrics: Dict, ocr_metrics: Dict, pipeline_metrics: Dict, output_dir: Path):
    """Create comprehensive comparison plot"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # YOLO metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [
        yolo_metrics['precision'] * 100,
        yolo_metrics['recall'] * 100,
        yolo_metrics['f1_score'] * 100
    ]
    bars = ax1.bar(metrics_names, metrics_values, color=['steelblue', 'coral', 'mediumseagreen'], edgecolor='black')
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('YOLO Detection Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # OCR accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    acc_names = ['Character\nAccuracy', 'Plate\nAccuracy']
    acc_values = [
        ocr_metrics['char_accuracy'] * 100,
        ocr_metrics['plate_accuracy'] * 100
    ]
    bars = ax2.bar(acc_names, acc_values, color=['purple', 'orange'], edgecolor='black')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('OCR Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Pipeline performance
    ax3 = fig.add_subplot(gs[0, 2])
    pipeline_names = ['Detection\nRate', 'Plate\nAccuracy']
    pipeline_values = [
        pipeline_metrics['detection_rate'] * 100,
        pipeline_metrics['ocr_metrics']['plate_accuracy'] * 100
    ]
    bars = ax3.bar(pipeline_names, pipeline_values, color=['teal', 'crimson'], edgecolor='black')
    ax3.set_ylabel('Score (%)', fontsize=12)
    ax3.set_title('End-to-End Pipeline', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Speed comparison
    ax4 = fig.add_subplot(gs[1, :])
    components = ['YOLO\nDetection', 'OCR\nRecognition', 'Full\nPipeline']
    times_ms = [
        yolo_metrics['avg_inference_time'] * 1000,
        ocr_metrics.get('avg_inference_time', 0) * 1000,
        pipeline_metrics['avg_pipeline_time'] * 1000
    ]
    fps_values = [
        yolo_metrics['fps'],
        1000 / times_ms[1] if times_ms[1] > 0 else 0,
        1000 / times_ms[2] if times_ms[2] > 0 else 0
    ]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, times_ms, width, label='Time (ms)', color='skyblue', edgecolor='black')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, fps_values, width, label='FPS', color='lightcoral', edgecolor='black')
    
    ax4.set_xlabel('Component', fontsize=12)
    ax4.set_ylabel('Time (ms)', fontsize=12, color='skyblue')
    ax4_twin.set_ylabel('FPS', fontsize=12, color='lightcoral')
    ax4.set_title('Performance Benchmark', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(components)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.suptitle('Model Evaluation Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = output_dir / 'comparison_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved: {save_path}")
    plt.close()


# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate YOLO only
  python %(prog)s --yolo data/runs/detect/plat_nomor/weights/best.pt --test data/PlatNomor/test
  
  # Evaluate complete pipeline
  python %(prog)s --yolo data/plat_nomor.pt --test data/PlatNomor/test --pipeline
  
  # Evaluate with custom ground truth
  python %(prog)s --yolo data/plat_nomor.pt --test data/PlatNomor/test --gt ground_truth.json --pipeline
        """
    )
    
    parser.add_argument('--yolo', type=str, default=DEFAULT_YOLO_MODEL,
                       help='Path to YOLO model')
    parser.add_argument('--ocr', type=str, default=DEFAULT_OCR_MODEL,
                       help='Path to OCR model (Keras)')
    parser.add_argument('--test', type=str, default=DEFAULT_TEST_DIR,
                       help='Path to test dataset directory')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE,
                       help='Confidence threshold for detection')
    parser.add_argument('--gt', type=str,
                       help='Ground truth JSON file (for OCR evaluation)')
    parser.add_argument('--pipeline', action='store_true',
                       help='Evaluate complete pipeline (detection + OCR)')
    parser.add_argument('--yolo-only', action='store_true',
                       help='Evaluate YOLO detection only')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output) / f"evaluation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("MODEL EVALUATION PIPELINE")
        print("="*80)
        print(f"YOLO Model: {args.yolo}")
        print(f"Test Data: {args.test}")
        print(f"Output: {output_dir}")
        print("="*80)
        
        all_results = {}
        
        # ===== YOLO EVALUATION =====
        if not args.pipeline or args.yolo_only:
            yolo_evaluator = YOLOEvaluator(args.yolo, args.conf)
            yolo_results = yolo_evaluator.evaluate_dataset(args.test)
            
            if yolo_results:
                all_results['yolo'] = yolo_results
                
                # Visualize
                visualize_detection_results(yolo_results['results'], output_dir)
                
                # Save results
                with open(output_dir / 'yolo_evaluation.json', 'w') as f:
                    json.dump(yolo_results, f, indent=2)
        
        # ===== PIPELINE EVALUATION =====
        if args.pipeline:
            # Load ground truth if provided
            ground_truth = {}
            if args.gt and os.path.exists(args.gt):
                with open(args.gt, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                    # Convert to dict: image_name -> plate_text
                    for item in gt_data:
                        if isinstance(item, dict):
                            img_name = Path(item.get('image', '')).stem
                            plate_text = item.get('plate_text', '')
                            ground_truth[img_name] = plate_text
            else:
                print("[WARN] No ground truth file provided, using TNKB database")
                # Try to load from TNKB database
                tnkb_path = Path('data/tnkb.json')
                if tnkb_path.exists():
                    with open(tnkb_path, 'r', encoding='utf-8') as f:
                        tnkb_data = json.load(f)
                        for item in tnkb_data:
                            nopol = item.get('nopol', '').strip().upper()
                            if nopol:
                                # Use plate number as key
                                ground_truth[nopol] = nopol
            
            if not ground_truth:
                print("[ERROR] No ground truth available for pipeline evaluation")
                sys.exit(1)
            
            print(f"[INFO] Loaded {len(ground_truth)} ground truth entries")
            
            # Run pipeline evaluation
            pipeline_evaluator = EndToEndEvaluator(args.yolo, args.ocr, args.conf)
            pipeline_results = pipeline_evaluator.evaluate_pipeline(args.test, ground_truth)
            
            if pipeline_results:
                all_results['pipeline'] = pipeline_results
                
                # Save results
                with open(output_dir / 'pipeline_evaluation.json', 'w') as f:
                    json.dump(pipeline_results, f, indent=2)
        
        # ===== GENERATE SUMMARY =====
        if all_results:
            print("\n" + "="*80)
            print("GENERATING EVALUATION SUMMARY")
            print("="*80)
            
            # Create comprehensive report
            report = {
                'timestamp': timestamp,
                'configuration': {
                    'yolo_model': args.yolo,
                    'ocr_model': args.ocr,
                    'test_directory': args.test,
                    'confidence_threshold': args.conf
                },
                'results': all_results
            }
            
            # Save report
            report_path = output_dir / 'evaluation_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✓ Full report saved: {report_path}")
            
            # Create comparison visualization
            if 'yolo' in all_results and 'pipeline' in all_results:
                create_comparison_plot(
                    all_results['yolo']['metrics'],
                    all_results['pipeline']['ocr_metrics'],
                    all_results['pipeline'],
                    output_dir
                )
            
            # Print final summary
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            
            if 'yolo' in all_results:
                yolo_metrics = all_results['yolo']['metrics']
                print("\n📊 YOLO Detection:")
                print(f"  • Precision:  {yolo_metrics['precision']*100:.2f}%")
                print(f"  • Recall:     {yolo_metrics['recall']*100:.2f}%")
                print(f"  • F1 Score:   {yolo_metrics['f1_score']*100:.2f}%")
                print(f"  • Avg IoU:    {yolo_metrics['avg_iou']*100:.2f}%")
                print(f"  • Speed:      {yolo_metrics['fps']:.2f} FPS")
            
            if 'pipeline' in all_results:
                pipeline_metrics = all_results['pipeline']
                print("\n🔤 OCR Recognition:")
                print(f"  • Char Acc:   {pipeline_metrics['ocr_metrics']['char_accuracy']*100:.2f}%")
                print(f"  • Plate Acc:  {pipeline_metrics['ocr_metrics']['plate_accuracy']*100:.2f}%")
                print(f"  • Avg Lev:    {pipeline_metrics['ocr_metrics']['avg_levenshtein']:.2f}")
                
                print("\n🚀 Pipeline Performance:")
                print(f"  • Detection:  {pipeline_metrics['detection_rate']*100:.2f}%")
                print(f"  • Speed:      {1000/pipeline_metrics['avg_pipeline_time'] if pipeline_metrics['avg_pipeline_time'] > 0 else 0:.2f} FPS")
                print(f"  • Latency:    {pipeline_metrics['avg_pipeline_time']*1000:.2f} ms")
            
            print("\n📁 Output Files:")
            print(f"  • Report:     {report_path}")
            print(f"  • Visualizations: {output_dir}")
            print("="*80 + "\n")
            
            return report
        
        else:
            print("[ERROR] No evaluation results generated")
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()