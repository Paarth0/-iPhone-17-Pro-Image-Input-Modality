#!/usr/bin/env python3
"""
iPhone 17 Pro Vision Router Simulation - CLI Entrypoint
=========================================================
Implements the Omni-Model Vision Router (Chapter 3) for laptop validation.

This CLI tool processes images through:
1. Thermal-aware resolution scaling
2. Security filtering (OCR injection guard + sensitive content)
3. Vision encoding (MobileNetV3 via ONNX)
4. Intent routing (Document/Object/Art classification)

Usage:
    python main.py --image data/input/test.jpg --thermal nominal
    python main.py --image data/input/doc.jpg --thermal critical --verbose

Reference: iPhone 17 Pro Image Input Modality Spec, Chapter 3
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

# Local module imports (to be implemented in subsequent steps)
from src.pipeline.resolution_scaler import ResolutionScaler, ThermalState
from src.pipeline.security_filter import SecurityFilter
from src.model.vision_encoder import VisionEncoder
from src.model.intent_router import IntentRouter, Intent
from src.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisionRouter")


class VisionRouterPipeline:
    """
    Main pipeline orchestrator for the Agentic Vision Router.
    
    Implements the secure vision pipeline architecture:
    FrameBuffer → ResolutionScaler → SecurityGate → EncoderEngine → IntentMapper
    
    Reference: Spec Section 3.2 - Pipeline Architecture
    """
    
    def __init__(self, model_path: str = "models/mobilenet_v3.onnx", verbose: bool = False):
        """
        Initialize all pipeline components.
        
        Args:
            model_path: Path to ONNX model file
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.timings: Dict[str, float] = {}
        
        logger.info("Initializing Vision Router Pipeline...")
        
        # Initialize components
        # Implements: CFR-1 (Resolution scaling based on thermal state)
        self.resolution_scaler = ResolutionScaler()
        
        # Implements: CFR-3 (Blind Eye) and CFR-4 (Injection Guard)
        self.security_filter = SecurityFilter()
        
        # Implements: Core vision encoding via MobileNetV3 proxy
        self.vision_encoder = VisionEncoder(model_path)
        
        # Implements: CFR-2 (Intent Classification)
        self.intent_router = IntentRouter()
        
        logger.info("Pipeline initialization complete.")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from disk (simulates camera frame buffer).
        
        Implements: CNFR-2 (Privacy) - Image loaded to memory only
        
        Args:
            image_path: Path to input image file
            
        Returns:
            Image as numpy array (BGR format)
            
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image cannot be decoded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")
        
        logger.info(f"Loaded image: {image_path} (Shape: {image.shape})")
        return image
    
    def process(
        self,
        image_path: str,
        thermal_state: ThermalState
    ) -> Dict[str, Any]:
        """
        Execute the full vision routing pipeline.
        
        Pipeline stages:
        1. Load image (InputHandler)
        2. Get resolution based on thermal state (StateEngine)
        3. Resize image (ResolutionScaler)
        4. Security scan and filter (SecurityFilter)
        5. Encode with vision model (VisionEncoder)
        6. Route to intent (IntentRouter)
        
        Args:
            image_path: Path to input image
            thermal_state: Simulated device thermal state
            
        Returns:
            Dictionary containing processing results and metadata
        """
        pipeline_start = time.time()
        result = {
            "filename": os.path.basename(image_path),
            "filepath": image_path,
            "timestamp": datetime.now().isoformat(),
            "thermal_state": thermal_state.value,
            "pipeline_stages": {},
            "security_scan": {},
            "classification": {},
            "routed_intent": None,
            "timings_ms": {},
            "success": False,
            "error": None
        }
        
        try:
            # Stage 1: Load Image
            stage_start = time.time()
            image = self.load_image(image_path)
            original_shape = image.shape[:2]
            result["original_resolution"] = f"{original_shape[1]}x{original_shape[0]}"
            self.timings["load"] = (time.time() - stage_start) * 1000
            
            # Stage 2: Determine target resolution based on thermal state
            # Reference: Spec Table 8.2 - Dynamic Resolution Logic
            stage_start = time.time()
            target_resolution = self.resolution_scaler.get_resolution(thermal_state)
            result["resolution_processed"] = f"{target_resolution[0]}x{target_resolution[1]}"
            result["pipeline_stages"]["resolution_decision"] = {
                "thermal_input": thermal_state.value,
                "target_width": target_resolution[0],
                "target_height": target_resolution[1]
            }
            
            # Stage 3: Resize image
            resized_image = self.resolution_scaler.resize_image(image, thermal_state)
            self.timings["resize"] = (time.time() - stage_start) * 1000
            logger.info(f"Resized image to {target_resolution} based on thermal state: {thermal_state.value}")
            
            # Stage 4: Security Filtering
            # Implements: CFR-3 (Blind Eye) and CFR-4 (Injection Guard)
            stage_start = time.time()
            security_result = self.security_filter.scan(resized_image, os.path.basename(image_path))
            
            result["security_scan"] = {
                "blur_applied": security_result.blur_applied,
                "ocr_mask_applied": security_result.ocr_mask_applied,
                "detected_threats": security_result.detected_threats,
                "threat_regions": security_result.threat_regions,
                "is_safe": security_result.is_safe
            }
            
            # Apply security transformations if needed
            processed_image = security_result.processed_image
            self.timings["security"] = (time.time() - stage_start) * 1000
            
            if security_result.blur_applied:
                logger.warning("Sensitive content detected - blur applied (Blind Eye)")
            if security_result.ocr_mask_applied:
                logger.warning(f"Adversarial text detected - masked: {security_result.detected_threats}")
            
            # Stage 5: Vision Encoding
            # Reference: Spec Section 3.3 - MobileNet-V5 Encoder (using V3 proxy)
            stage_start = time.time()
            encoding_result = self.vision_encoder.encode(processed_image)
            
            result["classification"] = {
                "raw_class_id": encoding_result.class_id,
                "raw_class_name": encoding_result.class_name,
                "confidence": round(encoding_result.confidence, 4),
                "top_5_predictions": encoding_result.top_5
            }
            self.timings["encode"] = (time.time() - stage_start) * 1000
            logger.info(f"Encoded image - Top class: {encoding_result.class_name} ({encoding_result.confidence:.2%})")
            
            # Stage 6: Intent Routing
            # Reference: Spec Section 3.4 - Strategic Intent Classification
            stage_start = time.time()
            intent = self.intent_router.route(encoding_result.class_id, encoding_result.class_name)
            
            result["routed_intent"] = intent.value
            result["pipeline_stages"]["routing"] = {
                "input_class": encoding_result.class_name,
                "mapped_intent": intent.value,
                "intent_description": self.intent_router.get_intent_description(intent)
            }
            self.timings["route"] = (time.time() - stage_start) * 1000
            logger.info(f"Routed to intent: {intent.value}")
            
            # Calculate total timing
            total_time = (time.time() - pipeline_start) * 1000
            self.timings["total"] = total_time
            
            result["timings_ms"] = {k: round(v, 2) for k, v in self.timings.items()}
            result["success"] = True
            
            # Performance check (CNFR requirement: should be <500ms on CPU)
            if total_time > 500:
                logger.warning(f"Pipeline exceeded 500ms target: {total_time:.2f}ms")
            else:
                logger.info(f"Pipeline completed in {total_time:.2f}ms (within 500ms target)")
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return result


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="iPhone 17 Pro Vision Router Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a document image with nominal thermal state
  python main.py --image data/input/document.jpg --thermal nominal

  # Process under thermal stress (critical state)
  python main.py --image data/input/test.jpg --thermal critical

  # Batch process all images in a directory
  python main.py --batch data/input/ --thermal fair --output results/

  # Verbose output with timing details
  python main.py --image test.jpg --thermal nominal --verbose

Thermal States:
  nominal  - Full resolution (768x768)
  fair     - Reduced resolution (512x512)
  serious  - Minimum resolution (256x256)
  critical - Minimum resolution (256x256)
        """
    )
    
    # Input options (mutually exclusive: single image or batch)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to single input image (JPEG/PNG)"
    )
    input_group.add_argument(
        "--batch", "-b",
        type=str,
        help="Path to directory containing multiple images"
    )
    
    # Thermal state simulation
    parser.add_argument(
        "--thermal", "-t",
        type=str,
        required=True,
        choices=["nominal", "fair", "serious", "critical"],
        help="Simulated device thermal state"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/output",
        help="Output directory for JSON results (default: data/output)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/mobilenet_v3.onnx",
        help="Path to ONNX model file (default: models/mobilenet_v3.onnx)"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Pretty print JSON output
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )
    
    # Suppress file output (print to stdout only)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output JSON to stdout instead of file"
    )
    
    return parser.parse_args()


def get_thermal_state(thermal_str: str) -> ThermalState:
    """
    Convert string argument to ThermalState enum.
    
    Args:
        thermal_str: Thermal state as string
        
    Returns:
        ThermalState enum value
    """
    thermal_map = {
        "nominal": ThermalState.NOMINAL,
        "fair": ThermalState.FAIR,
        "serious": ThermalState.SERIOUS,
        "critical": ThermalState.CRITICAL
    }
    return thermal_map[thermal_str.lower()]


def save_result(result: Dict[str, Any], output_dir: str, pretty: bool = False) -> str:
    """
    Save processing result to JSON file.
    
    Implements: CNFR-2 (Privacy) - Only metadata saved, not raw images
    
    Args:
        result: Processing result dictionary
        output_dir: Output directory path
        pretty: Whether to pretty-print JSON
        
    Returns:
        Path to saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(result["filename"])[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_result_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Write JSON
    with open(output_path, "w") as f:
        if pretty:
            json.dump(result, f, indent=2)
        else:
            json.dump(result, f)
    
    return output_path


def process_batch(
    pipeline: VisionRouterPipeline,
    input_dir: str,
    thermal_state: ThermalState,
    output_dir: str,
    pretty: bool = False
) -> Dict[str, Any]:
    """
    Process all images in a directory.
    
    Args:
        pipeline: Initialized pipeline instance
        input_dir: Directory containing input images
        thermal_state: Simulated thermal state
        output_dir: Output directory for results
        pretty: Whether to pretty-print JSON
        
    Returns:
        Summary dictionary with batch results
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]
    
    if not image_files:
        logger.warning(f"No supported images found in {input_dir}")
        return {"batch_size": 0, "results": []}
    
    logger.info(f"Processing batch of {len(image_files)} images...")
    
    batch_results = {
        "batch_size": len(image_files),
        "thermal_state": thermal_state.value,
        "input_directory": input_dir,
        "timestamp": datetime.now().isoformat(),
        "results": [],
        "summary": {
            "successful": 0,
            "failed": 0,
            "intent_distribution": {
                "INTENT_A_PRACTICAL_GUIDANCE": 0,
                "INTENT_B_DISCOVERY": 0,
                "INTENT_C_CREATIVE": 0
            }
        }
    }
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            result = pipeline.process(image_path, thermal_state)
            batch_results["results"].append(result)
            batch_results["summary"]["successful"] += 1
            
            if result["routed_intent"]:
                intent_key = result["routed_intent"]
                if intent_key in batch_results["summary"]["intent_distribution"]:
                    batch_results["summary"]["intent_distribution"][intent_key] += 1
                    
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            batch_results["results"].append({
                "filename": image_file,
                "success": False,
                "error": str(e)
            })
            batch_results["summary"]["failed"] += 1
    
    # Save batch summary
    summary_path = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(batch_results, f, indent=2 if pretty else None)
    
    logger.info(f"Batch processing complete. Summary saved to: {summary_path}")
    return batch_results


def main():
    """
    Main entry point for the Vision Router CLI.
    
    Implements the full pipeline orchestration as specified in
    the iPhone 17 Pro Image Input Modality validation plan.
    """
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Convert thermal state
    thermal_state = get_thermal_state(args.thermal)
    logger.info(f"Thermal state: {thermal_state.value}")
    
    # Check model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("Run 'python scripts/download_model.py' to download the model.")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = VisionRouterPipeline(
            model_path=args.model,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process based on input mode
    if args.image:
        # Single image processing
        try:
            result = pipeline.process(args.image, thermal_state)
            
            if args.stdout:
                # Output to stdout
                print(json.dumps(result, indent=2 if args.pretty else None))
            else:
                # Save to file
                output_path = save_result(result, args.output, args.pretty)
                logger.info(f"Result saved to: {output_path}")
                
                # Also print summary to console
                print("\n" + "=" * 60)
                print("VISION ROUTER RESULT SUMMARY")
                print("=" * 60)
                print(f"  File:        {result['filename']}")
                print(f"  Thermal:     {result['thermal_state']}")
                print(f"  Resolution:  {result['resolution_processed']}")
                print(f"  Class:       {result['classification'].get('raw_class_name', 'N/A')}")
                print(f"  Confidence:  {result['classification'].get('confidence', 0):.2%}")
                print(f"  Intent:      {result['routed_intent']}")
                print(f"  Security:    {'⚠️ Filtered' if result['security_scan'].get('blur_applied') or result['security_scan'].get('ocr_mask_applied') else '✅ Clean'}")
                print(f"  Time:        {result['timings_ms'].get('total', 0):.2f}ms")
                print("=" * 60)
                
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
            
    elif args.batch:
        # Batch processing
        if not os.path.isdir(args.batch):
            logger.error(f"Batch directory not found: {args.batch}")
            sys.exit(1)
            
        batch_results = process_batch(
            pipeline,
            args.batch,
            thermal_state,
            args.output,
            args.pretty
        )
        
        # Print batch summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total Images:  {batch_results['batch_size']}")
        print(f"  Successful:    {batch_results['summary']['successful']}")
        print(f"  Failed:        {batch_results['summary']['failed']}")
        print(f"  Intent Distribution:")
        for intent, count in batch_results['summary']['intent_distribution'].items():
            print(f"    - {intent}: {count}")
        print("=" * 60)


if __name__ == "__main__":
    main()