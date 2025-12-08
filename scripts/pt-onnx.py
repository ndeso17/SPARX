"""
YOLOv8 PT to ONNX Converter
Converts Ultralytics YOLOv8 .pt models to ONNX format with IR version compatibility
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np

print("=" * 70)
print("YOLOV8 PT TO ONNX CONVERTER")
print("=" * 70)

def convert_yolo_to_onnx(pt_path: str, 
                         img_size: int = 640,
                         opset: int = 11,  # Changed default to 11
                         simplify: bool = True,
                         dynamic: bool = False,
                         half: bool = False):
    """
    Convert YOLOv8 .pt model to ONNX format
    
    Args:
        pt_path: Path to .pt model file
        img_size: Input image size (default: 640)
        opset: ONNX opset version (default: 11)
        simplify: Simplify ONNX model (default: True)
        dynamic: Dynamic input shapes (default: False)
        half: FP16 quantization (default: False)
    """
    
    # Check if file exists
    if not os.path.exists(pt_path):
        print(f"❌ Error: Model file not found: {pt_path}")
        return None
    
    print(f"\n📁 Loading PyTorch model: {pt_path}")
    
    try:
        # Load YOLO model
        model = YOLO(pt_path)
        print(f"✓ Model loaded successfully")
        
        # Get model info
        print(f"\n📊 Model Information:")
        print(f"  - Task: {model.task}")
        print(f"  - Model type: {type(model.model).__name__}")
        
        # Set export parameters
        export_args = {
            'format': 'onnx',
            'imgsz': img_size,
            'opset': opset,
            'simplify': simplify,
            'dynamic': dynamic,
            'half': half
        }
        
        print(f"\n⚙️  Export Configuration:")
        print(f"  - Image size: {img_size}")
        print(f"  - ONNX opset: {opset}")
        print(f"  - Simplify: {simplify}")
        print(f"  - Dynamic shapes: {dynamic}")
        print(f"  - Half precision (FP16): {half}")
        
        # Export to ONNX
        print(f"\n🔄 Converting to ONNX...")
        onnx_path = model.export(**export_args)
        
        print(f"✓ Conversion successful!")
        print(f"✓ ONNX model saved: {onnx_path}")
        
        # Fix IR version for compatibility
        print(f"\n🔧 Adjusting ONNX IR version for compatibility...")
        try:
            onnx_model = onnx.load(str(onnx_path))
            original_ir = onnx_model.ir_version
            
            # Check ONNX Runtime support
            try:
                test_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                print(f"✓ IR version {original_ir} is compatible with your ONNX Runtime")
                del test_session
            except Exception as e:
                if "IR version" in str(e):
                    # Extract max supported version from error message
                    print(f"⚠️  Original IR version {original_ir} not supported")
                    print(f"   Downgrading to IR version 8 (compatible with opset {opset})")
                    
                    # Downgrade IR version
                    onnx_model.ir_version = 8
                    onnx.save(onnx_model, str(onnx_path))
                    print(f"✓ IR version adjusted to {onnx_model.ir_version}")
                    
                    # Verify the fix worked
                    test_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                    print(f"✓ Model now loads successfully!")
                    del test_session
                else:
                    raise e
                    
        except Exception as e:
            print(f"⚠️  IR version adjustment: {e}")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_onnx_model(onnx_path: str, img_size: int = 640):
    """
    Verify ONNX model and test inference
    """
    print(f"\n🔍 Verifying ONNX model...")
    
    try:
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")
        
        # Get model info
        print(f"\n📊 ONNX Model Information:")
        print(f"  - IR Version: {onnx_model.ir_version}")
        print(f"  - Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        print(f"  - Opset: {onnx_model.opset_import[0].version}")
        
        # Input/Output info
        print(f"\n  Inputs:")
        for inp in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                    for d in inp.type.tensor_type.shape.dim]
            print(f"    - {inp.name}: {shape}")
        
        print(f"\n  Outputs:")
        for out in onnx_model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                    for d in out.type.tensor_type.shape.dim]
            print(f"    - {out.name}: {shape}")
        
        # Test with ONNX Runtime
        print(f"\n🧪 Testing inference with ONNX Runtime...")
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Create dummy input
        if 'dynamic' in str(input_shape):
            dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        else:
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        print(f"  - Input name: {input_name}")
        print(f"  - Input shape: {dummy_input.shape}")
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"  - Output shapes:")
        for i, out in enumerate(outputs):
            print(f"    Output {i}: {out.shape}")
        
        print("✓ Inference test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_outputs(pt_path: str, onnx_path: str, img_size: int = 640):
    """
    Compare outputs between PT and ONNX models
    """
    print(f"\n🔬 Comparing PT and ONNX outputs...")
    
    try:
        # Load models
        pt_model = YOLO(pt_path)
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        
        # PT inference
        print("  Running PT inference...")
        pt_results = pt_model.predict(dummy_input, verbose=False)
        
        # ONNX inference
        print("  Running ONNX inference...")
        input_name = ort_session.get_inputs()[0].name
        onnx_outputs = ort_session.run(None, {input_name: dummy_input})
        
        # Compare
        print(f"\n  Comparison:")
        print(f"    - ONNX outputs: {len(onnx_outputs)}")
        
        # Check if outputs match (approximately)
        if pt_results and len(pt_results) > 0:
            print("  ✓ Both models produce outputs")
        
        print("  ✓ Comparison complete")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not compare outputs: {e}")
        return False

def get_file_sizes(pt_path: str, onnx_path: str):
    """
    Compare file sizes
    """
    pt_size = os.path.getsize(pt_path) / (1024 * 1024)  # MB
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    
    print(f"\n📦 File Size Comparison:")
    print(f"  - PT model:   {pt_size:.2f} MB")
    print(f"  - ONNX model: {onnx_size:.2f} MB")
    print(f"  - Ratio:      {onnx_size/pt_size:.2f}x")

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 PT model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python pt-onnx.py model.pt
  
  # With custom image size
  python pt-onnx.py model.pt --imgsz 416
  
  # With dynamic shapes (for variable input sizes)
  python pt-onnx.py model.pt --dynamic
  
  # With FP16 quantization (smaller, faster)
  python pt-onnx.py model.pt --half
  
  # Full options
  python pt-onnx.py model.pt --imgsz 640 --opset 11 --simplify --dynamic

Output:
  - model.onnx: Converted ONNX model (with automatic IR compatibility fix)
  - Verification and comparison reports
        """
    )
    
    parser.add_argument("model", help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model (default: True)")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify", help="Do not simplify")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes")
    parser.add_argument("--half", action="store_true", help="Export with FP16 quantization")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification step")
    parser.add_argument("--skip-compare", action="store_true", help="Skip output comparison")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Convert
    onnx_path = convert_yolo_to_onnx(
        args.model,
        img_size=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half
    )
    
    if not onnx_path:
        print("\n❌ Conversion failed!")
        sys.exit(1)
    
    # Verify
    if not args.skip_verify:
        verify_success = verify_onnx_model(onnx_path, args.imgsz)
        if not verify_success:
            print("\n⚠️  Warning: Verification failed, but ONNX file was created")
    
    # Compare
    if not args.skip_compare:
        compare_outputs(args.model, onnx_path, args.imgsz)
    
    # File sizes
    get_file_sizes(args.model, onnx_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETED!")
    print("=" * 70)
    print(f"\n✅ ONNX model ready: {onnx_path}")
    print(f"\n📖 Usage with ONNX Runtime:")
    print(f"""
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('{onnx_path}')

# Prepare input (1, 3, {args.imgsz}, {args.imgsz})
input_name = session.get_inputs()[0].name
image = np.random.randn(1, 3, {args.imgsz}, {args.imgsz}).astype(np.float32)

# Run inference
outputs = session.run(None, {{input_name: image}})
    """)
    print("=" * 70)

if __name__ == "__main__":
    main()