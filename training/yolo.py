#!/usr/bin/env python3
"""
YOLO Training Script for Indonesian License Plate Detection
Features:
- Automated dataset preparation and validation
- Data augmentation pipeline
- Training with callbacks and monitoring
- Model evaluation and export
- Training visualization and metrics logging
- Support for resume training
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# ========== CONFIGURATION ==========
DEFAULT_DATA_DIR = "data/dataset"
DEFAULT_OUTPUT_DIR = "data/runs/detect"
DEFAULT_EPOCHS = 100
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_MODEL = "yolov8n.pt"  # nano model for license plates

# ========== DATASET PREPARATION ==========
class DatasetValidator:
    """Validate and prepare dataset for YOLO training"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
    def validate_structure(self) -> bool:
        """Check if dataset has correct structure"""
        print("\n" + "="*80)
        print("VALIDATING DATASET STRUCTURE")
        print("="*80)
        
        # Check directories
        if not self.data_dir.exists():
            print(f"[ERROR] Data directory not found: {self.data_dir}")
            return False
            
        if not self.images_dir.exists():
            print(f"[ERROR] Images directory not found: {self.images_dir}")
            return False
            
        if not self.labels_dir.exists():
            print(f"[ERROR] Labels directory not found: {self.labels_dir}")
            return False
        
        # Count files
        image_files = list(self.images_dir.glob("*.jpg")) + \
                     list(self.images_dir.glob("*.jpeg")) + \
                     list(self.images_dir.glob("*.png"))
        label_files = list(self.labels_dir.glob("*.txt"))
        
        print(f"[OK] Found {len(image_files)} images")
        print(f"[OK] Found {len(label_files)} label files")
        
        if len(image_files) == 0:
            print("[ERROR] No images found")
            return False
            
        if len(label_files) == 0:
            print("[ERROR] No label files found")
            return False
        
        # Check matching files
        unmatched = []
        for img_path in image_files[:10]:  # Check first 10
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                unmatched.append(img_path.name)
        
        if unmatched:
            print(f"[WARN] Some images missing labels (showing first 10):")
            for name in unmatched[:10]:
                print(f"  - {name}")
        else:
            print("[OK] All checked images have matching labels")
        
        return True
    
    def validate_annotations(self, num_samples: int = 10) -> bool:
        """Validate annotation format"""
        print("\nVALIDATING ANNOTATIONS")
        print("-"*80)
        
        label_files = list(self.labels_dir.glob("*.txt"))[:num_samples]
        
        if not label_files:
            print("[ERROR] No label files found for validation")
            return False
        
        valid_count = 0
        for label_path in label_files:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    print(f"[WARN] Empty label file: {label_path.name}")
                    continue
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"[WARN] Invalid format in {label_path.name}: {line.strip()}")
                        continue
                    
                    class_id, x, y, w, h = map(float, parts)
                    
                    # Validate ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"[WARN] Out of range values in {label_path.name}")
                        continue
                
                valid_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to read {label_path.name}: {e}")
        
        print(f"[OK] Validated {valid_count}/{len(label_files)} sample annotations")
        return valid_count > 0
    
    def visualize_samples(self, num_samples: int = 5, output_dir: str = None):
        """Visualize sample images with annotations"""
        print(f"\nVISUALIZING {num_samples} SAMPLE ANNOTATIONS")
        print("-"*80)
        
        image_files = list(self.images_dir.glob("*.jpg")) + \
                     list(self.images_dir.glob("*.jpeg")) + \
                     list(self.images_dir.glob("*.png"))
        
        if not image_files:
            print("[WARN] No images found for visualization")
            return
        
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for idx, img_path in enumerate(samples):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Read label
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # Convert to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, 'Plate', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_path.name}", fontsize=10)
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'dataset_samples.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved visualization: {save_path}")
        
        plt.show()
        plt.close()


def prepare_yolo_dataset(data_dir: str, output_dir: str, 
                         train_ratio: float = 0.8, val_ratio: float = 0.15):
    """Prepare dataset in YOLO format with train/val/test split"""
    print("\n" + "="*80)
    print("PREPARING YOLO DATASET")
    print("="*80)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    images_dir = data_path / "images"
    image_files = list(images_dir.glob("*.jpg")) + \
                 list(images_dir.glob("*.jpeg")) + \
                 list(images_dir.glob("*.png"))
    
    if not image_files:
        print("[ERROR] No images found in dataset")
        return None
    
    print(f"Total images: {len(image_files)}")
    
    # Split dataset
    train_files, temp_files = train_test_split(
        image_files, train_size=train_ratio, random_state=42
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_files, test_files = train_test_split(
        temp_files, train_size=val_size, random_state=42
    )
    
    print(f"Train: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Val:   {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
    print(f"Test:  {len(test_files)} ({len(test_files)/len(image_files)*100:.1f}%)")
    
    # Copy files
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"\nCopying {split_name} files...")
        
        for img_path in files:
            # Copy image
            dst_img = output_path / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy label
            label_path = data_path / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = output_path / split_name / 'labels' / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dst_label)
    
    # Create data.yaml
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # number of classes
        'names': ['license_plate']
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n[OK] Dataset prepared at: {output_path}")
    print(f"[OK] Config saved: {yaml_path}")
    
    return str(yaml_path)


# ========== TRAINING ==========
class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / 'training_log.json'
        self.logs = []
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics"""
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.logs.append(log_entry)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def plot_metrics(self):
        """Plot training metrics"""
        if not self.logs:
            print("[WARN] No training logs available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = [log['epoch'] for log in self.logs]
        
        # Loss plots
        if 'train_loss' in self.logs[0]:
            train_loss = [log.get('train_loss', 0) for log in self.logs]
            val_loss = [log.get('val_loss', 0) for log in self.logs]
            
            axes[0, 0].plot(epochs, train_loss, label='Train Loss', marker='o')
            axes[0, 0].plot(epochs, val_loss, label='Val Loss', marker='s')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # mAP plots
        if 'mAP50' in self.logs[0]:
            map50 = [log.get('mAP50', 0) for log in self.logs]
            map50_95 = [log.get('mAP50-95', 0) for log in self.logs]
            
            axes[0, 1].plot(epochs, map50, label='mAP@0.5', marker='o', color='green')
            axes[0, 1].plot(epochs, map50_95, label='mAP@0.5:0.95', marker='s', color='blue')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].set_title('Mean Average Precision')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision/Recall
        if 'precision' in self.logs[0]:
            precision = [log.get('precision', 0) for log in self.logs]
            recall = [log.get('recall', 0) for log in self.logs]
            
            axes[1, 0].plot(epochs, precision, label='Precision', marker='o', color='purple')
            axes[1, 0].plot(epochs, recall, label='Recall', marker='s', color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Precision and Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in self.logs[0]:
            lr = [log.get('lr', 0) for log in self.logs]
            axes[1, 1].plot(epochs, lr, marker='o', color='red')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'training_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Metrics plot saved: {save_path}")
        
        plt.show()
        plt.close()


def train_yolo_model(
    data_yaml: str,
    model_name: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    resume: bool = False,
    patience: int = 50,
    device: str = '0'
):
    """Train YOLO model for license plate detection"""
    
    print("\n" + "="*80)
    print("STARTING YOLO TRAINING")
    print("="*80)
    
    # Validate data_yaml exists
    if not Path(data_yaml).exists():
        print(f"[ERROR] Data config not found: {data_yaml}")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"plat_nomor_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    print(f"Data config: {data_yaml}")
    print(f"Base model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Initialize model
    if resume and (run_dir / 'weights' / 'last.pt').exists():
        print("\n[OK] Resuming from last checkpoint")
        model = YOLO(str(run_dir / 'weights' / 'last.pt'))
    else:
        print(f"\n[OK] Loading base model: {model_name}")
        # Check if model file exists
        if not Path(model_name).exists():
            print(f"[INFO] Downloading base model: {model_name}")
        model = YOLO(model_name)
    
    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'name': run_dir.name,
        'project': str(output_path),
        'patience': patience,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'device': device,
        'workers': 8,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': True,  # Single class (license plate)
        'rect': False,  # Rectangular training
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation for last N epochs
        'resume': resume,
        # Augmentation parameters
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,  # No rotation for license plates
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,  # No vertical flip
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n" + "="*80)
    print("TRAINING IN PROGRESS")
    print("="*80)
    
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\nFinal Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        # Export model
        best_model_path = run_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            print(f"\n[OK] Best model saved: {best_model_path}")
            
            # Export to ONNX (optional)
            try:
                model = YOLO(str(best_model_path))
                onnx_path = model.export(format='onnx')
                print(f"[OK] ONNX model exported: {onnx_path}")
            except Exception as e:
                print(f"[WARN] ONNX export failed: {e}")
        
        return str(best_model_path)
        
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(
        description="YOLO Training Script for License Plate Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python %(prog)s --data data/dataset --epochs 100
  
  # Custom configuration
  python %(prog)s --data data/dataset --model yolov8s.pt --epochs 150 --batch 32
  
  # Resume training
  python %(prog)s --data data/dataset --resume
  
  # Validate dataset only
  python %(prog)s --data data/dataset --validate-only
        """
    )
    
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_DIR,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for training results')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help='Base YOLO model (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE,
                       help='Image size for training')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0, 1, cpu)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset without training')
    parser.add_argument('--no-split', action='store_true',
                       help='Skip dataset splitting (use existing split)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    
    args = parser.parse_args()
    
    try:
        # Validate dataset
        validator = DatasetValidator(args.data)
        
        if not validator.validate_structure():
            print("\n[ERROR] Dataset validation failed")
            sys.exit(1)
        
        if not validator.validate_annotations():
            print("\n[ERROR] Annotation validation failed")
            sys.exit(1)
        
        # Visualize samples
        validator.visualize_samples(num_samples=5, output_dir=args.output)
        
        if args.validate_only:
            print("\n[OK] Dataset validation completed")
            sys.exit(0)
        
        # Prepare dataset
        if not args.no_split:
            prepared_dir = Path(args.output) / 'dataset_prepared'
            data_yaml = prepare_yolo_dataset(
                data_dir=args.data,
                output_dir=str(prepared_dir),
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio
            )
            
            if not data_yaml:
                print("\n[ERROR] Dataset preparation failed")
                sys.exit(1)
        else:
            # Use existing data.yaml
            data_yaml = Path(args.data) / 'data.yaml'
            if not data_yaml.exists():
                print(f"[ERROR] data.yaml not found at {data_yaml}")
                sys.exit(1)
            data_yaml = str(data_yaml)
        
        # Train model
        best_model = train_yolo_model(
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            output_dir=args.output,
            resume=args.resume,
            patience=args.patience,
            device=args.device
        )
        
        if best_model:
            print(f"\n{'='*80}")
            print("TRAINING SUMMARY")
            print(f"{'='*80}")
            print(f"[OK] Best model: {best_model}")
            print(f"[OK] Ready for inference")
            print(f"{'='*80}\n")
        else:
            print("\n[ERROR] Training did not complete successfully")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()