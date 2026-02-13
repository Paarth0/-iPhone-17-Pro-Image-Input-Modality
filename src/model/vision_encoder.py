"""
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
                    f"Model not found: {self.model_path}\n"
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
