"""
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
