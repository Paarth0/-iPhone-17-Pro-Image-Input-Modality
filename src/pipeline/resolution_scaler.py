"""
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
