#!/usr/bin/env python3
"""
Fix script - Creates all module files with correct content.
Run from project root: python3 fix_all_files.py
"""

import os

# Ensure we're in the right directory
if not os.path.exists("src"):
    print("ERROR: Run this from the project root (iphone17-pro-vision-sim)")
    exit(1)

files_to_create = {}

# =============================================================================
# src/__init__.py
# =============================================================================
files_to_create["src/__init__.py"] = '''"""
iPhone 17 Pro Vision Router Simulation
"""

__version__ = "0.1.0"
__author__ = "Vision Router Team"
'''

# =============================================================================
# src/pipeline/__init__.py
# =============================================================================
files_to_create["src/pipeline/__init__.py"] = '''"""
Pipeline Module
"""

from .resolution_scaler import ResolutionScaler, ThermalState
from .security_filter import SecurityFilter, SecurityResult

__all__ = [
    "ResolutionScaler",
    "ThermalState",
    "SecurityFilter",
    "SecurityResult",
]
'''

# =============================================================================
# src/model/__init__.py
# =============================================================================
files_to_create["src/model/__init__.py"] = '''"""
Model Module
"""

from .vision_encoder import VisionEncoder, EncodingResult
from .intent_router import IntentRouter, Intent

__all__ = [
    "VisionEncoder",
    "EncodingResult",
    "IntentRouter",
    "Intent",
]
'''

# =============================================================================
# src/config/__init__.py
# =============================================================================
files_to_create["src/config/__init__.py"] = '''"""
Config Module
"""

from .settings import Settings, DEFAULT_SETTINGS

__all__ = ["Settings", "DEFAULT_SETTINGS"]
'''

# =============================================================================
# src/pipeline/resolution_scaler.py
# =============================================================================
files_to_create["src/pipeline/resolution_scaler.py"] = '''"""
Resolution Scaler Module
"""

from enum import Enum
from typing import Tuple
import numpy as np
import cv2


class ThermalState(Enum):
    """Device thermal states that affect processing resolution."""
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"


class ResolutionScaler:
    """Thermal-aware resolution scaler."""
    
    RESOLUTION_MAP = {
        ThermalState.NOMINAL: (768, 768),
        ThermalState.FAIR: (512, 512),
        ThermalState.SERIOUS: (256, 256),
        ThermalState.CRITICAL: (256, 256),
    }
    
    DEFAULT_RESOLUTION = (512, 512)
    
    def __init__(self):
        pass
    
    @classmethod
    def get_resolution(cls, thermal_state: ThermalState) -> Tuple[int, int]:
        """Get target resolution for given thermal state."""
        return cls.RESOLUTION_MAP.get(thermal_state, cls.DEFAULT_RESOLUTION)
    
    @classmethod
    def resize_image(cls, image: np.ndarray, thermal_state: ThermalState) -> np.ndarray:
        """Resize image based on thermal state."""
        target = cls.get_resolution(thermal_state)
        current_size = (image.shape[1], image.shape[0])
        
        if current_size == target:
            return image
        
        if current_size[0] > target[0] or current_size[1] > target[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        return cv2.resize(image, target, interpolation=interpolation)
    
    @classmethod
    def get_scale_factor(cls, original_size: Tuple[int, int], thermal_state: ThermalState) -> float:
        """Calculate the scale factor for a given thermal state."""
        target = cls.get_resolution(thermal_state)
        scale_w = target[0] / original_size[0]
        scale_h = target[1] / original_size[1]
        return min(scale_w, scale_h)
    
    @staticmethod
    def get_all_resolutions() -> dict:
        """Get all thermal state to resolution mappings."""
        return {state.value: res for state, res in ResolutionScaler.RESOLUTION_MAP.items()}
'''

# =============================================================================
# src/pipeline/security_filter.py
# =============================================================================
files_to_create["src/pipeline/security_filter.py"] = '''"""
Security Filter Module
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityResult:
    """Result of security scan on an image."""
    processed_image: np.ndarray
    blur_applied: bool = False
    ocr_mask_applied: bool = False
    detected_threats: List[str] = field(default_factory=list)
    threat_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    is_safe: bool = True


class SecurityFilter:
    """Security filter implementing Blind Eye and Injection Guard."""
    
    ADVERSARIAL_KEYWORDS = [
        "override", "system", "ignore", "bypass", "admin",
        "prompt", "instruction", "forget", "disregard",
    ]
    
    SENSITIVE_FILENAME_PATTERNS = [
        "explicit", "nsfw", "unsafe", "adult", "sensitive", "private",
    ]
    
    def __init__(self, use_ocr: bool = True):
        """Initialize security filter."""
        self.use_ocr = use_ocr
        self._ocr_reader = None
    
    @property
    def ocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._ocr_reader is None and self.use_ocr:
            try:
                import easyocr
                logger.info("Loading EasyOCR reader...")
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                logger.info("EasyOCR reader loaded")
            except ImportError:
                logger.warning("EasyOCR not installed. OCR disabled.")
                self.use_ocr = False
            except Exception as e:
                logger.warning(f"Failed to load EasyOCR: {e}")
                self.use_ocr = False
        return self._ocr_reader
    
    def scan(self, image: np.ndarray, filename: str = "") -> SecurityResult:
        """Scan image for security threats."""
        processed = image.copy()
        result = SecurityResult(processed_image=processed)
        
        if self._check_sensitive_content(filename):
            logger.warning(f"Sensitive content detected: {filename}")
            processed = self._apply_blur(processed)
            result.blur_applied = True
            result.is_safe = False
            result.detected_threats.append("SENSITIVE_CONTENT")
        
        if self.use_ocr:
            ocr_result = self._check_adversarial_text(processed)
            if ocr_result["found"]:
                logger.warning(f"Adversarial text: {ocr_result['keywords']}")
                processed = self._mask_text_regions(processed, ocr_result["regions"])
                result.ocr_mask_applied = True
                result.is_safe = False
                result.detected_threats.extend(ocr_result["keywords"])
                result.threat_regions.extend(ocr_result["regions"])
        
        result.processed_image = processed
        return result
    
    def _check_sensitive_content(self, filename: str) -> bool:
        """Check if filename indicates sensitive content."""
        if not filename:
            return False
        filename_lower = filename.lower()
        for pattern in self.SENSITIVE_FILENAME_PATTERNS:
            if pattern in filename_lower:
                return True
        return False
    
    def _check_adversarial_text(self, image: np.ndarray) -> dict:
        """Use OCR to detect adversarial keywords."""
        result = {"found": False, "keywords": [], "regions": []}
        
        if not self.use_ocr or self.ocr_reader is None:
            return result
        
        try:
            ocr_results = self.ocr_reader.readtext(image)
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                text_lower = text.lower()
                
                for keyword in self.ADVERSARIAL_KEYWORDS:
                    if keyword in text_lower:
                        result["found"] = True
                        result["keywords"].append(keyword.upper())
                        
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w = int(max(x_coords) - x)
                        h = int(max(y_coords) - y)
                        result["regions"].append((x, y, w, h))
                        break
                        
        except Exception as e:
            logger.error(f"OCR scan failed: {e}")
        
        return result
    
    def _apply_blur(self, image: np.ndarray, blur_strength: int = 51) -> np.ndarray:
        """Apply Gaussian blur to image."""
        if blur_strength % 2 == 0:
            blur_strength += 1
        return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    def _mask_text_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Mask detected text regions with black boxes."""
        masked = image.copy()
        for (x, y, w, h) in regions:
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return masked
    
    @classmethod
    def get_adversarial_keywords(cls) -> List[str]:
        """Get list of monitored adversarial keywords."""
        return cls.ADVERSARIAL_KEYWORDS.copy()
'''

# =============================================================================
# src/model/vision_encoder.py
# =============================================================================
files_to_create["src/model/vision_encoder.py"] = '''"""
Vision Encoder Module
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Result of vision encoding."""
    class_id: int
    class_name: str
    confidence: float
    top_5: List[Tuple[str, float]] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None


class VisionEncoder:
    """Vision encoder using MobileNetV3 ONNX model."""
    
    INPUT_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(self, model_path: str = "models/mobilenet_v3.onnx"):
        """Initialize the vision encoder."""
        self.model_path = model_path
        self._session = None
        self._labels = self._get_imagenet_labels()
    
    @property
    def session(self):
        """Lazy load ONNX Runtime session."""
        if self._session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}\\n"
                    f"Run python3 scripts/download_model.py to download it."
                )
            
            try:
                import onnxruntime as ort
                logger.info(f"Loading ONNX model: {self.model_path}")
                
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self._session = ort.InferenceSession(
                    self.model_path,
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                logger.info("ONNX model loaded")
                
            except ImportError:
                raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        
        return self._session
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        resized = cv2.resize(image, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD
        transposed = normalized.transpose(2, 0, 1)
        return np.expand_dims(transposed, axis=0)
    
    def encode(self, image: np.ndarray) -> EncodingResult:
        """Encode image and return classification results."""
        if not os.path.exists(self.model_path):
            logger.warning("Model not found, returning placeholder")
            return self._placeholder_result()
        
        try:
            input_tensor = self.preprocess(image)
            
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run([output_name], {input_name: input_tensor})
            
            logits = outputs[0][0]
            probabilities = self._softmax(logits)
            
            top_indices = np.argsort(probabilities)[::-1][:5]
            
            top_class_id = top_indices[0]
            top_confidence = float(probabilities[top_class_id])
            top_class_name = self._labels.get(top_class_id, f"class_{top_class_id}")
            
            top_5 = [
                (self._labels.get(idx, f"class_{idx}"), float(probabilities[idx]))
                for idx in top_indices
            ]
            
            return EncodingResult(
                class_id=int(top_class_id),
                class_name=top_class_name,
                confidence=top_confidence,
                top_5=top_5,
            )
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return self._placeholder_result()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _placeholder_result(self) -> EncodingResult:
        """Return placeholder when model unavailable."""
        return EncodingResult(
            class_id=0,
            class_name="placeholder",
            confidence=0.0,
            top_5=[("placeholder", 0.0)],
        )
    
    def _get_imagenet_labels(self) -> dict:
        """Get ImageNet class labels (subset)."""
        return {
            281: "tabby_cat", 282: "tiger_cat", 283: "persian_cat",
            284: "siamese_cat", 285: "egyptian_cat",
            446: "binder", 457: "book_jacket",
            504: "coffee_mug", 508: "computer_keyboard",
            620: "laptop", 671: "cell_phone",
            695: "notebook", 917: "three_ring_binder",
            970: "alp", 973: "coral_reef", 975: "lakeside",
            978: "seashore", 979: "valley", 980: "volcano",
        }
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if self._session is None and not os.path.exists(self.model_path):
            return {"loaded": False, "path": self.model_path}
        return {"loaded": True, "path": self.model_path}
'''

# =============================================================================
# src/model/intent_router.py
# =============================================================================
files_to_create["src/model/intent_router.py"] = '''"""
Intent Router Module
"""

from enum import Enum
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Strategic intent categories for vision routing."""
    INTENT_A_PRACTICAL_GUIDANCE = "INTENT_A_PRACTICAL_GUIDANCE"
    INTENT_B_DISCOVERY = "INTENT_B_DISCOVERY"
    INTENT_C_CREATIVE = "INTENT_C_CREATIVE"
    UNKNOWN = "UNKNOWN"


class IntentRouter:
    """Routes ImageNet classifications to strategic intents."""
    
    INTENT_DESCRIPTIONS = {
        Intent.INTENT_A_PRACTICAL_GUIDANCE: "Document/Text processing for practical guidance",
        Intent.INTENT_B_DISCOVERY: "Object/Product recognition for discovery",
        Intent.INTENT_C_CREATIVE: "Art/Nature appreciation for creative inspiration",
        Intent.UNKNOWN: "Unable to determine intent",
    }
    
    INTENT_A_CLASS_IDS: Set[int] = {
        446, 457, 550, 623, 624, 662, 695, 732, 737, 739, 765, 818, 917,
    }
    
    INTENT_B_CLASS_IDS: Set[int] = {
        281, 282, 283, 284, 285,
        504, 505, 508, 620, 621, 671, 720, 761, 762, 849, 850,
        0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    }
    
    INTENT_C_CLASS_IDS: Set[int] = {
        970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980,
    }
    
    INTENT_A_KEYWORDS = [
        "book", "paper", "document", "letter", "notebook", "binder",
        "menu", "magazine", "newspaper", "envelope", "library",
    ]
    
    INTENT_B_KEYWORDS = [
        "cat", "dog", "bird", "fish", "animal",
        "phone", "computer", "keyboard", "laptop",
        "cup", "mug", "bottle", "bowl",
    ]
    
    INTENT_C_KEYWORDS = [
        "mountain", "lake", "sea", "ocean", "beach",
        "forest", "tree", "flower", "sunset",
        "valley", "cliff", "waterfall", "volcano",
    ]
    
    def __init__(self):
        """Initialize intent router."""
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build lookup tables for routing."""
        self._class_id_to_intent: Dict[int, Intent] = {}
        
        for class_id in self.INTENT_A_CLASS_IDS:
            self._class_id_to_intent[class_id] = Intent.INTENT_A_PRACTICAL_GUIDANCE
        
        for class_id in self.INTENT_B_CLASS_IDS:
            self._class_id_to_intent[class_id] = Intent.INTENT_B_DISCOVERY
        
        for class_id in self.INTENT_C_CLASS_IDS:
            self._class_id_to_intent[class_id] = Intent.INTENT_C_CREATIVE
    
    def route(self, class_id: int, class_name: str) -> Intent:
        """Route a classification to an intent."""
        if class_id in self._class_id_to_intent:
            return self._class_id_to_intent[class_id]
        
        class_name_lower = class_name.lower().replace("_", " ")
        
        for keyword in self.INTENT_A_KEYWORDS:
            if keyword in class_name_lower:
                return Intent.INTENT_A_PRACTICAL_GUIDANCE
        
        for keyword in self.INTENT_C_KEYWORDS:
            if keyword in class_name_lower:
                return Intent.INTENT_C_CREATIVE
        
        for keyword in self.INTENT_B_KEYWORDS:
            if keyword in class_name_lower:
                return Intent.INTENT_B_DISCOVERY
        
        return Intent.INTENT_B_DISCOVERY
    
    def route_batch(self, predictions: List[tuple]) -> List[Intent]:
        """Route multiple predictions."""
        return [self.route(cid, name) for cid, name in predictions]
    
    def get_intent_description(self, intent: Intent) -> str:
        """Get human-readable description of an intent."""
        return self.INTENT_DESCRIPTIONS.get(intent, "Unknown intent")
    
    def get_all_intents(self) -> List[Intent]:
        """Get all possible intent values."""
        return [
            Intent.INTENT_A_PRACTICAL_GUIDANCE,
            Intent.INTENT_B_DISCOVERY,
            Intent.INTENT_C_CREATIVE,
        ]
    
    def get_routing_stats(self) -> dict:
        """Get routing configuration stats."""
        return {
            "intent_a_class_count": len(self.INTENT_A_CLASS_IDS),
            "intent_b_class_count": len(self.INTENT_B_CLASS_IDS),
            "intent_c_class_count": len(self.INTENT_C_CLASS_IDS),
        }
'''

# =============================================================================
# src/config/settings.py
# =============================================================================
files_to_create["src/config/settings.py"] = '''"""
Settings Module
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import os


@dataclass
class Settings:
    """Application settings and configuration."""
    
    MODEL_PATH: str = "models/mobilenet_v3.onnx"
    MODEL_INPUT_SIZE: int = 224
    DEFAULT_THERMAL_STATE: str = "nominal"
    MAX_PIPELINE_LATENCY_MS: float = 500.0
    
    ADVERSARIAL_KEYWORDS: Tuple[str, ...] = (
        "override", "system", "ignore", "bypass", "admin"
    )
    SENSITIVE_FILENAME_PATTERNS: Tuple[str, ...] = (
        "explicit", "nsfw", "unsafe", "adult", "sensitive"
    )
    
    OUTPUT_DIR: str = "data/output"
    INPUT_DIR: str = "data/input"
    SAVE_PROCESSED_IMAGES: bool = False
    
    RESOLUTION_MAP: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "nominal": (768, 768),
        "fair": (512, 512),
        "serious": (256, 256),
        "critical": (256, 256),
    })
    
    def __post_init__(self):
        """Validate settings."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.INPUT_DIR, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            MODEL_PATH=os.getenv("VISION_MODEL_PATH", cls.MODEL_PATH),
            OUTPUT_DIR=os.getenv("VISION_OUTPUT_DIR", cls.OUTPUT_DIR),
        )
    
    def get_resolution(self, thermal_state: str) -> Tuple[int, int]:
        """Get resolution for a thermal state."""
        return self.RESOLUTION_MAP.get(thermal_state.lower(), (512, 512))
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            "model_path": self.MODEL_PATH,
            "output_dir": self.OUTPUT_DIR,
            "default_thermal_state": self.DEFAULT_THERMAL_STATE,
        }


DEFAULT_SETTINGS = Settings()
'''

# =============================================================================
# Create all files
# =============================================================================
def main():
    print("=" * 50)
    print("Fixing all module files...")
    print("=" * 50)
    
    for filepath, content in files_to_create.items():
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Verify
        size = os.path.getsize(filepath)
        print(f"  ✓ {filepath} ({size} bytes)")
    
    print("=" * 50)
    print("All files created successfully!")
    print("=" * 50)
    print("\nNow run: python3 scripts/verify_setup.py")


if __name__ == "__main__":
    main()