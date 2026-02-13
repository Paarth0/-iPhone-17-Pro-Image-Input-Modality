#!/usr/bin/env python3
"""
Create Test Images
==================
Generates simple test images for pipeline validation.

Usage:
    python3 scripts/create_test_images.py
"""

import os
import sys
import numpy as np
import cv2

# Configuration
OUTPUT_DIR = "data/input"


def create_document_image():
    """Create a simple document-like image with text."""
    img = np.ones((400, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Add some "text" lines (black rectangles)
    for i in range(8):
        y = 40 + i * 40
        width = np.random.randint(150, 280)
        cv2.rectangle(img, (20, y), (20 + width, y + 15), (0, 0, 0), -1)
    
    # Add a title area
    cv2.rectangle(img, (20, 10), (200, 30), (50, 50, 50), -1)
    
    return img, "document_test.jpg"


def create_object_image():
    """Create a simple object-like image (colored shapes)."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw a "mug" shape (circle + rectangle)
    cv2.circle(img, (150, 150), 60, (139, 69, 19), -1)  # Brown circle
    cv2.rectangle(img, (200, 120), (230, 180), (139, 69, 19), -1)  # Handle
    cv2.ellipse(img, (150, 100), (50, 15), 0, 0, 360, (101, 67, 33), -1)  # Top
    
    return img, "object_test.jpg"


def create_nature_image():
    """Create a simple nature/landscape image."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Sky (blue gradient)
    for i in range(150):
        blue = 255 - i
        img[i, :] = [blue, 200, 100]
    
    # Ground (green)
    img[150:, :] = [34, 139, 34]  # Forest green
    
    # Sun
    cv2.circle(img, (320, 50), 30, (0, 255, 255), -1)  # Yellow sun
    
    # Mountains
    pts = np.array([[0, 150], [100, 80], [200, 150]], np.int32)
    cv2.fillPoly(img, [pts], (105, 105, 105))
    
    pts = np.array([[150, 150], [250, 60], [350, 150]], np.int32)
    cv2.fillPoly(img, [pts], (128, 128, 128))
    
    return img, "nature_test.jpg"


def create_sensitive_test_image():
    """Create a test image with sensitive filename pattern."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray
    cv2.putText(img, "TEST", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return img, "explicit_test.jpg"  # Filename triggers security filter


def create_adversarial_text_image():
    """Create an image with adversarial text for OCR testing."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White background
    
    # Add adversarial text
    cv2.putText(img, "SYSTEM OVERRIDE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Normal text here", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    
    return img, "adversarial_test.jpg"


def main():
    print("=" * 50)
    print("Creating Test Images")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/\n")
    
    # Generate test images
    test_generators = [
        ("Document (Intent A)", create_document_image),
        ("Object (Intent B)", create_object_image),
        ("Nature (Intent C)", create_nature_image),
        ("Sensitive Content Test", create_sensitive_test_image),
        ("Adversarial Text Test", create_adversarial_text_image),
    ]
    
    created_files = []
    
    for description, generator in test_generators:
        try:
            img, filename = generator()
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, img)
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename} ({size} bytes) - {description}")
            created_files.append(filepath)
        except Exception as e:
            print(f"  ✗ Failed to create {description}: {e}")
    
    print("\n" + "=" * 50)
    print(f"✅ Created {len(created_files)} test images")
    print("=" * 50)
    
    print("\nTest commands:")
    print("-" * 50)
    print(f"python3 main.py --image {OUTPUT_DIR}/document_test.jpg --thermal nominal")
    print(f"python3 main.py --image {OUTPUT_DIR}/object_test.jpg --thermal fair")
    print(f"python3 main.py --image {OUTPUT_DIR}/nature_test.jpg --thermal serious")
    print(f"python3 main.py --image {OUTPUT_DIR}/explicit_test.jpg --thermal nominal")
    print(f"python3 main.py --image {OUTPUT_DIR}/adversarial_test.jpg --thermal nominal")


if __name__ == "__main__":
    main()