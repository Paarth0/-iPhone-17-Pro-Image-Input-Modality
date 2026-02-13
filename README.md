# iPhone 17 Pro Image Input Modality

[![Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A laptop-based validation environment for the iPhone 17 Pro's **Omni-Model Vision Router** - a real-time pipeline that processes visual input, adapts to thermal conditions, filters for security, and routes to AI agents based on detected intent.

## Purpose

This project validates the core architecture described in the iPhone 17 Pro Image Input Modality specification (Chapter 3) without requiring actual iPhone hardware.

### What It Validates

| Component | Spec Reference | Implementation |
|-----------|---------------|----------------|
| Dynamic Resolution | Table 8.2 | Thermal-aware resizing (768→256px) |
| Intent Classification | Section 3.4 | Document/Object/Nature routing |
| Blind Eye Security | Section 3.5 | Sensitive content detection |
| Injection Guard | Section 3.5 | Adversarial text masking |

## Quick Start

### Prerequisites

- Python 3.9+
- macOS, Linux, or Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/iphone17-pro-vision-sim.git
cd iphone17-pro-vision-sim

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the ML model
python3 scripts/download_model.py

# Create test images
python3 scripts/create_test_images.py


## 1A RUN THE PIPELINE

# Basic usage
python3 main.py --image data/input/document_test.jpg --thermal nominal

# With thermal throttling
python3 main.py --image data/input/document_test.jpg --thermal critical

# Test security filters
python3 main.py --image data/input/explicit_test.jpg --thermal nominal


## 1B RUN TESTS
python3 -m pytest tests/ -v


## 2A COMMAND EXAMPLES

# Process single image
python3 main.py --image photo.jpg --thermal nominal

# Batch process directory
python3 main.py --batch data/input/ --thermal fair --output results/

# Verbose output
python3 main.py --image test.jpg --thermal nominal --verbose

# Output to stdout (JSON)
python3 main.py --image test.jpg --thermal nominal --stdout --pretty


## 2B TESTING
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_resolution_scaler.py -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=html


Test Coverage:
Module	            Tests	Status
ResolutionScaler	18	    ✅ Pass
SecurityFilter	    14	    ✅ Pass
IntentRouter	    22	    ✅ Pass
Integration	        9	    ✅ Pass



**Save and exit.**

---

# Progress Summary

| Step | Status |
|------|--------|
| ✅ 1. Project Structure | Complete |
| ✅ 2. Core Modules | Complete |
| ✅ 3. Model Download | Complete |
| ✅ 4. Test Images | Complete |
| ✅ 5. Pipeline Testing | Complete |
| ✅ 6. Unit Tests (63 passed) | Complete |
| ✅ 7. Documentation | Complete |
| ✅ 8. Diagrams | Complete |

---

# Final Step: Verify Everything

```bash
# Check all files exist
ls -la *.md
ls -la docs/diagrams/

# Run tests one more time
python3 -m pytest tests/ -v

# Run pipeline test
python3 main.py --image data/input/document_test.jpg --thermal nominal

## RUN FINAL VALIDATION
python3 scripts/final_validation.py
