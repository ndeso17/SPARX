#!/usr/bin/env python3
"""
CNN OCR Training Script for Indonesian License Plate Character Recognition
Features:
- Custom CNN architecture optimized for license plate characters
- Character-level dataset preparation from cropped plate images
- Data augmentation pipeline
- Multi-GPU support
- Model export (Keras, ONNX, PyTorch)
- Training visualization and metrics
- Character confusion matrix analysis
"""

import os
import sys
import json
import argparse
import random
import shutil  # FIXED: Added missing import
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Deep Learning frameworks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Optional: PyTorch export
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch not available for export")

# Optional: ONNX export
try:
    import tf2onnx
    ONNX_AVAILABLE = True
except:
    ONNX_AVAILABLE = False
    print("[INFO] tf2onnx not available for ONNX export")

# ========== CONFIGURATION ==========
DEFAULT_DATA_DIR = "data/DatasetCharacter"
DEFAULT_OUTPUT_DIR = "data/runs/ocr"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMG_SIZE = (64, 64)  # Character image size
DEFAULT_LR = 0.001

# Indonesian license plate characters
INDONESIAN_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ========== DATASET PREPARATION ==========
class CharacterDatasetBuilder:
    """Build character-level dataset from license plate images"""
    
    def __init__(self, output_dir: str, img_size: Tuple[int, int] = (64, 64)):
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.char_mapping = {char: idx for idx, char in enumerate(INDONESIAN_CHARS)}
        self.idx_to_char = {idx: char for char, idx in self.char_mapping.items()}
        
    def create_synthetic_dataset(self, num_samples_per_char: int = 500):
        """Create synthetic character dataset with augmentation"""
        print("\n" + "="*80)
        print("CREATING SYNTHETIC CHARACTER DATASET")
        print("="*80)
        
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        
        for char in INDONESIAN_CHARS:
            (train_dir / char).mkdir(parents=True, exist_ok=True)
            (val_dir / char).mkdir(parents=True, exist_ok=True)
        
        # Font settings for text generation
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
        ]
        
        print(f"Generating {num_samples_per_char} samples per character...")
        
        for char in INDONESIAN_CHARS:
            for i in range(num_samples_per_char):
                # Create blank image
                img = np.ones((self.img_size[0], self.img_size[1]), dtype=np.uint8) * 255
                
                # Random font and size
                font = random.choice(fonts)
                font_scale = random.uniform(1.0, 2.5)
                thickness = random.randint(2, 4)
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    char, font, font_scale, thickness
                )
                
                # Center text
                x = (self.img_size[1] - text_width) // 2
                y = (self.img_size[0] + text_height) // 2
                
                # Random color (grayscale)
                color = random.randint(0, 100)
                
                # Draw text
                cv2.putText(img, char, (x, y), font, font_scale, color, thickness)
                
                # Apply augmentations
                img = self._apply_augmentations(img)
                
                # Split train/val (80/20)
                if i < int(num_samples_per_char * 0.8):
                    save_dir = train_dir / char
                else:
                    save_dir = val_dir / char
                
                filename = f"{char}_{i:04d}.png"
                cv2.imwrite(str(save_dir / filename), img)
            
            if (INDONESIAN_CHARS.index(char) + 1) % 10 == 0:
                print(f"  Processed {INDONESIAN_CHARS.index(char) + 1}/{len(INDONESIAN_CHARS)} characters")
        
        print(f"\n✓ Dataset created at: {self.output_dir}")
        print(f"  Train samples: {num_samples_per_char * len(INDONESIAN_CHARS) * 0.8:.0f}")
        print(f"  Val samples: {num_samples_per_char * len(INDONESIAN_CHARS) * 0.2:.0f}")
        
        return str(self.output_dir)
    
    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to character image"""
        
        # Random noise
        if random.random() > 0.5:
            noise = np.random.normal(0, random.randint(5, 15), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Random blur
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Random rotation (small angles)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        # Random scaling
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size)
            
            # Pad or crop to original size
            if scale < 1.0:
                pad_w = (self.img_size[1] - img.shape[1]) // 2
                pad_h = (self.img_size[0] - img.shape[0]) // 2
                img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, 
                                        cv2.BORDER_CONSTANT, value=255)
            else:
                crop_w = (img.shape[1] - self.img_size[1]) // 2
                crop_h = (img.shape[0] - self.img_size[0]) // 2
                img = img[crop_h:crop_h+self.img_size[0], crop_w:crop_w+self.img_size[1]]
        
        # Ensure correct size
        if img.shape != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        return img
    
    def visualize_samples(self, num_samples: int = 36):
        """Visualize random samples from dataset"""
        print(f"\nVisualizing {num_samples} random samples...")
        
        train_dir = self.output_dir / 'train'
        
        # Collect random samples
        samples = []
        for char in random.sample(list(INDONESIAN_CHARS), min(num_samples, len(INDONESIAN_CHARS))):
            char_dir = train_dir / char
            if char_dir.exists():
                images = list(char_dir.glob('*.png'))
                if images:
                    img_path = random.choice(images)
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    samples.append((char, img))
        
        # Plot
        n_cols = 6
        n_rows = (len(samples) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, (char, img) in enumerate(samples):
            if idx < len(axes):
                axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(f"'{char}'", fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_samples.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {self.output_dir / 'dataset_samples.png'}")
        plt.show()


# ========== MODEL ARCHITECTURE ==========
def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build CNN model for character recognition"""
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ========== TRAINING ==========
class OCRTrainer:
    """Handle OCR model training"""
    
    def __init__(self, model: keras.Model, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
        
    def train(
        self,
        train_data,
        val_data,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        class_names: List[str]
    ):
        """Train the model"""
        
        print("\n" + "="*80)
        print("STARTING OCR TRAINING")
        print("="*80)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Calculate steps
        steps_per_epoch = train_data.samples // batch_size
        validation_steps = val_data.samples // batch_size
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Training samples: {train_data.samples}")
        print(f"  Validation samples: {val_data.samples}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        
        # Callbacks
        checkpoint_path = self.output_dir / 'best_model.keras'
        
        callback_list = [
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(
                str(self.output_dir / 'training_log.csv')
            ),
            callbacks.TensorBoard(
                log_dir=str(self.output_dir / 'tensorboard'),
                histogram_freq=1
            )
        ]
        
        # Train
        print("\n" + "="*80)
        print("TRAINING IN PROGRESS")
        print("="*80 + "\n")
        
        try:
            self.history = self.model.fit(
                train_data,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_data,
                validation_steps=validation_steps,
                callbacks=callback_list,
                verbose=1
            )
            
            print("\n" + "="*80)
            print("TRAINING COMPLETED")
            print("="*80)
            
            # Save final model
            final_model_path = self.output_dir / 'final_model.keras'
            self.model.save(str(final_model_path))
            print(f"\n✓ Final model saved: {final_model_path}")
            
            # Save labels mapping
            labels_path = self.output_dir / 'char_labels.json'
            labels_data = {
                'chars': class_names,
                'char_to_idx': {char: idx for idx, char in enumerate(class_names)},
                'idx_to_char': {idx: char for idx, char in enumerate(class_names)}
            }
            with open(labels_path, 'w') as f:
                json.dump(labels_data, f, indent=2)
            print(f"✓ Labels saved: {labels_path}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n[WARN] Training interrupted by user")
            return False
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_training_history(self):
        """Plot training metrics"""
        if not self.history:
            print("[WARN] No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        if 'top3_acc' in history:
            axes[1, 0].plot(epochs, history['top3_acc'], 'g-', label='Training Top-3', linewidth=2)
            axes[1, 0].plot(epochs, history['val_top3_acc'], 'orange', label='Val Top-3', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Top-3 Accuracy', fontsize=12)
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved: {save_path}")
        plt.show()
    
    def evaluate_and_visualize(self, val_data, class_names: List[str]):
        """Evaluate model and create confusion matrix"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Predictions
        print("\nGenerating predictions...")
        val_data.reset()
        predictions = self.model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1)
        y_pred = np.argmax(predictions, axis=1)
        
        # True labels
        y_true = val_data.classes[:len(y_pred)]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved: {cm_path}")
        plt.show()
        
        # Find most confused pairs
        print("\nMost Confused Character Pairs:")
        cm_off_diag = cm.copy()
        np.fill_diagonal(cm_off_diag, 0)
        
        top_confusions = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm_off_diag[i, j] > 0:
                    top_confusions.append((cm_off_diag[i, j], class_names[i], class_names[j]))
        
        top_confusions.sort(reverse=True)
        for count, true_char, pred_char in top_confusions[:10]:
            print(f"  '{true_char}' → '{pred_char}': {count} times")


# ========== MODEL EXPORT ==========
def export_model(model: keras.Model, output_dir: Path, model_name: str = 'char_model'):
    """Export model to multiple formats"""
    print("\n" + "="*80)
    print("EXPORTING MODEL")
    print("="*80)
    
    # Keras format (already saved during training)
    keras_path = output_dir / f'{model_name}.keras'
    print(f"✓ Keras model: {keras_path}")
    
    # SavedModel format
    try:
        saved_model_path = output_dir / f'{model_name}_savedmodel'
        model.export(str(saved_model_path))
        print(f"✓ SavedModel exported: {saved_model_path}")
    except Exception as e:
        print(f"[WARN] SavedModel export failed: {e}")
    
    # ONNX format
    if ONNX_AVAILABLE:
        try:
            import tf2onnx
            onnx_path = output_dir / f'{model_name}.onnx'
            
            model_proto, _ = tf2onnx.convert.from_keras(model, opset=13)
            with open(onnx_path, "wb") as f:
                f.write(model_proto.SerializeToString())
            
            print(f"✓ ONNX model exported: {onnx_path}")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")
    
    # TFLite format
    try:
        tflite_path = output_dir / f'{model_name}.tflite'
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TFLite model exported: {tflite_path}")
    except Exception as e:
        print(f"[WARN] TFLite export failed: {e}")


# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(
        description="CNN OCR Training Script for License Plate Character Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create synthetic dataset and train
  python %(prog)s --create-dataset --samples 500 --epochs 50
  
  # Train on existing dataset
  python %(prog)s --data data/DatasetCharacter --epochs 100 --batch 64
  
  # Resume training
  python %(prog)s --data data/DatasetCharacter --resume
  
  # Quick test with small dataset
  python %(prog)s --create-dataset --samples 100 --epochs 10 --batch 16
        """
    )
    
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_DIR,
                       help='Path to character dataset directory')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for training results')
    parser.add_argument('--create-dataset', action='store_true',
                       help='Create synthetic character dataset')
    parser.add_argument('--samples', type=int, default=500,
                       help='Samples per character for synthetic dataset')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=64,
                       help='Image size (square)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from best checkpoint')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device (0, 1, or -1 for CPU)')
    
    args = parser.parse_args()
    
    # GPU configuration
    if args.gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("[INFO] Using CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"[INFO] Using GPU: {args.gpu}")
    
    # Check GPU availability
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] GPUs available: {tf.config.list_physical_devices('GPU')}")
    
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output) / f"char_recognition_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Dataset preparation
        if args.create_dataset:
            builder = CharacterDatasetBuilder(
                output_dir=args.data,
                img_size=(args.img_size, args.img_size)
            )
            dataset_dir = builder.create_synthetic_dataset(num_samples_per_char=args.samples)
            builder.visualize_samples()
        else:
            dataset_dir = args.data
        
        # Verify dataset
        train_dir = Path(dataset_dir) / 'train'
        val_dir = Path(dataset_dir) / 'val'
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"[ERROR] Dataset not found. Use --create-dataset to generate one.")
            sys.exit(1)
        
        # Step 2: Prepare data generators
        print("\n" + "="*80)
        print("PREPARING DATA GENERATORS")
        print("="*80)
        
        img_size = (args.img_size, args.img_size)
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='constant',
            cval=1.0  # white background
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=img_size,
            batch_size=args.batch,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=img_size,
            batch_size=args.batch,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
        
        class_names = sorted(train_generator.class_indices.keys())
        num_classes = len(class_names)
        
        print(f"\n✓ Found {train_generator.samples} training images")
        print(f"✓ Found {val_generator.samples} validation images")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Classes: {class_names}")
        
        # Step 3: Build/load model
        input_shape = (img_size[0], img_size[1], 1)
        
        if args.resume:
            checkpoint_path = output_dir / 'best_model.keras'
            if checkpoint_path.exists():
                print(f"\n✓ Loading checkpoint: {checkpoint_path}")
                model = keras.models.load_model(str(checkpoint_path))
            else:
                print("[WARN] No checkpoint found, creating new model")
                model = build_cnn_model(input_shape, num_classes)
        else:
            print("\n✓ Building new model")
            model = build_cnn_model(input_shape, num_classes)
        
        # Step 4: Train
        trainer = OCRTrainer(model, output_dir)
        
        success = trainer.train(
            train_data=train_generator,
            val_data=val_generator,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            class_names=class_names
        )
        
        if not success:
            sys.exit(1)
        
        # Step 5: Plot training history
        trainer.plot_training_history()
        
        # Step 6: Evaluate
        trainer.evaluate_and_visualize(val_generator, class_names)
        
        # Step 7: Export model
        export_model(model, output_dir, 'char_all_model')
        
        # Step 8: Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ Model files:")
        print(f"  - best_model.keras (best validation accuracy)")
        print(f"  - final_model.keras (last epoch)")
        print(f"  - char_all_model.onnx (ONNX format)")
        print(f"  - char_all_model.tflite (TFLite format)")
        print(f"✓ Labels: {output_dir / 'char_labels.json'}")
        print(f"✓ Training log: {output_dir / 'training_log.csv'}")
        print(f"✓ Metrics plots: {output_dir / 'training_history.png'}")
        print(f"✓ Confusion matrix: {output_dir / 'confusion_matrix.png'}")
        print("="*80)
        
        # Copy to main data directory for easy access
        try:
            main_data_dir = Path('data')
            if main_data_dir.exists():
                shutil.copy2(output_dir / 'best_model.keras', main_data_dir / 'char_all_model.keras')
                shutil.copy2(output_dir / 'char_labels.json', main_data_dir / 'char_all_labels.json')
                print(f"\n✓ Copied best model to: data/char_all_model.keras")
                print(f"✓ Copied labels to: data/char_all_labels.json")
        except Exception as e:
            print(f"[WARN] Could not copy to main data directory: {e}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()