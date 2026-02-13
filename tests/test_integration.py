#!/usr/bin/env python3
"""
Integration tests for the full vision router pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from src.pipeline.resolution_scaler import ResolutionScaler, ThermalState
from src.pipeline.security_filter import SecurityFilter, SecurityResult
from src.model.intent_router import IntentRouter, Intent
from src.config.settings import Settings


class TestPipelineFlow:
    """Test the complete pipeline flow."""
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy test image."""
        return np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    def test_full_pipeline_nominal(self, dummy_image):
        """Test full pipeline with nominal thermal state."""
        # Step 1: Resolution scaling
        resized = ResolutionScaler.resize_image(dummy_image, ThermalState.NOMINAL)
        assert resized.shape[:2] == (768, 768)
        
        # Step 2: Security filter
        security = SecurityFilter(use_ocr=False)
        security_result = security.scan(resized, "safe_image.jpg")
        assert security_result.is_safe is True
        
        # Step 3: Intent routing (simulated classification)
        router = IntentRouter()
        intent = router.route(695, "notebook")
        assert intent == Intent.INTENT_A_PRACTICAL_GUIDANCE
    
    def test_full_pipeline_critical(self, dummy_image):
        """Test full pipeline with critical thermal state."""
        # Step 1: Resolution scaling (should downscale to 256)
        resized = ResolutionScaler.resize_image(dummy_image, ThermalState.CRITICAL)
        assert resized.shape[:2] == (256, 256)
        
        # Step 2: Security filter
        security = SecurityFilter(use_ocr=False)
        security_result = security.scan(resized, "safe_image.jpg")
        assert security_result.is_safe is True
    
    def test_pipeline_with_sensitive_content(self, dummy_image):
        """Test pipeline handles sensitive content."""
        # Step 1: Resolution scaling
        resized = ResolutionScaler.resize_image(dummy_image, ThermalState.NOMINAL)
        
        # Step 2: Security filter should detect sensitive filename
        security = SecurityFilter(use_ocr=False)
        security_result = security.scan(resized, "explicit_content.jpg")
        
        assert security_result.blur_applied is True
        assert security_result.is_safe is False
        assert "SENSITIVE_CONTENT" in security_result.detected_threats


class TestThermalThrottling:
    """Test thermal throttling behavior."""
    
    @pytest.fixture
    def large_image(self):
        """Create a large test image."""
        return np.zeros((2000, 2000, 3), dtype=np.uint8)
    
    def test_nominal_no_throttle(self, large_image):
        """NOMINAL should use maximum resolution."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.NOMINAL)
        assert resized.shape[:2] == (768, 768)
    
    def test_fair_mild_throttle(self, large_image):
        """FAIR should reduce resolution."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.FAIR)
        assert resized.shape[:2] == (512, 512)
    
    def test_serious_heavy_throttle(self, large_image):
        """SERIOUS should minimize resolution."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.SERIOUS)
        assert resized.shape[:2] == (256, 256)
    
    def test_critical_maximum_throttle(self, large_image):
        """CRITICAL should use minimum resolution."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.CRITICAL)
        assert resized.shape[:2] == (256, 256)


class TestSettingsIntegration:
    """Test settings integration."""
    
    def test_settings_resolution_map(self):
        """Settings should have correct resolution mappings."""
        settings = Settings()
        
        assert settings.get_resolution("nominal") == (768, 768)
        assert settings.get_resolution("fair") == (512, 512)
        assert settings.get_resolution("serious") == (256, 256)
        assert settings.get_resolution("critical") == (256, 256)
    
    def test_settings_to_dict(self):
        """Settings should convert to dictionary."""
        settings = Settings()
        config = settings.to_dict()
        
        assert "model_path" in config
        assert "output_dir" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])