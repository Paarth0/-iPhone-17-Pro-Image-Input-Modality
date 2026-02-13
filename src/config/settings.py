"""
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
