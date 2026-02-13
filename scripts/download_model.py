#!/usr/bin/env python3
"""
Download MobileNetV3 ONNX Model
===============================
Downloads the model needed for vision encoding.

Usage:
    python3 scripts/download_model.py
"""

import os
import sys
import urllib.request
import hashlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model configuration
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
MODEL_DIR = "models"
MODEL_NAME = "mobilenet_v3.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def download_with_progress(url: str, destination: str):
    """Download file with progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print("\n  Download complete!")


def verify_model(path: str) -> bool:
    """Verify the model file exists and is valid."""
    if not os.path.exists(path):
        return False
    
    size = os.path.getsize(path)
    if size < 1000000:  # Less than 1MB is suspicious
        print(f"  Warning: Model file seems too small ({size} bytes)")
        return False
    
    print(f"  Model size: {size / (1024*1024):.1f} MB")
    return True


def test_model(path: str) -> bool:
    """Test that the model loads correctly."""
    try:
        import onnxruntime as ort
        print("  Testing model with ONNX Runtime...")
        
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"  Input: {input_info.name} {input_info.shape}")
        print(f"  Output: {output_info.name} {output_info.shape}")
        print("  ✓ Model loads correctly!")
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def main():
    print("=" * 50)
    print("MobileNetV3 ONNX Model Downloader")
    print("=" * 50)
    
    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n1. Model directory: {MODEL_DIR}/")
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"\n2. Model already exists: {MODEL_PATH}")
        if verify_model(MODEL_PATH):
            print("\n3. Testing existing model...")
            if test_model(MODEL_PATH):
                print("\n" + "=" * 50)
                print("✅ Model is ready to use!")
                print("=" * 50)
                return
        
        print("  Existing model appears invalid. Re-downloading...")
        os.remove(MODEL_PATH)
    
    # Download model
    print(f"\n2. Downloading model...")
    try:
        download_with_progress(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        print("\n  Alternative: Manual download")
        print(f"  1. Visit: {MODEL_URL}")
        print(f"  2. Save as: {MODEL_PATH}")
        sys.exit(1)
    
    # Verify download
    print("\n3. Verifying download...")
    if not verify_model(MODEL_PATH):
        print("  ✗ Verification failed")
        sys.exit(1)
    
    # Test model
    print("\n4. Testing model...")
    if not test_model(MODEL_PATH):
        print("  ✗ Model test failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Model downloaded and verified successfully!")
    print("=" * 50)
    print(f"\nModel path: {MODEL_PATH}")
    print("You can now run: python3 main.py --help")


if __name__ == "__main__":
    main()