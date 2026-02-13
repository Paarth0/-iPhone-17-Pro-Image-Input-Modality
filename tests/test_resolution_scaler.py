#!/usr/bin/env python3
"""
Unit tests for ResolutionScaler.
Validates CFR-1: Thermal-aware image resizing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.pipeline.resolution_scaler import ResolutionScaler, ThermalState


class TestThermalStateEnum:
    """Test ThermalState enum values."""
    
    def test_nominal_value(self):
        assert ThermalState.NOMINAL.value == "nominal"
    
    def test_fair_value(self):
        assert ThermalState.FAIR.value == "fair"
    
    def test_serious_value(self):
        assert ThermalState.SERIOUS.value == "serious"
    
    def test_critical_value(self):
        assert ThermalState.CRITICAL.value == "critical"


class TestResolutionMapping:
    """Test thermal state to resolution mapping."""
    
    def test_nominal_returns_768(self):
        """NOMINAL thermal state should return 768x768."""
        result = ResolutionScaler.get_resolution(ThermalState.NOMINAL)
        assert result == (768, 768), f"Expected (768, 768), got {result}"
    
    def test_fair_returns_512(self):
        """FAIR thermal state should return 512x512."""
        result = ResolutionScaler.get_resolution(ThermalState.FAIR)
        assert result == (512, 512), f"Expected (512, 512), got {result}"
    
    def test_serious_returns_256(self):
        """SERIOUS thermal state should return 256x256."""
        result = ResolutionScaler.get_resolution(ThermalState.SERIOUS)
        assert result == (256, 256), f"Expected (256, 256), got {result}"
    
    def test_critical_returns_256(self):
        """CRITICAL thermal state should return 256x256."""
        result = ResolutionScaler.get_resolution(ThermalState.CRITICAL)
        assert result == (256, 256), f"Expected (256, 256), got {result}"


class TestImageResizing:
    """Test actual image resizing functionality."""
    
    @pytest.fixture
    def large_image(self):
        """Create a 1000x1000 test image."""
        return np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    @pytest.fixture
    def small_image(self):
        """Create a 100x100 test image."""
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def test_resize_to_768_nominal(self, large_image):
        """Large image should resize to 768x768 in NOMINAL state."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.NOMINAL)
        assert resized.shape[:2] == (768, 768)
    
    def test_resize_to_512_fair(self, large_image):
        """Large image should resize to 512x512 in FAIR state."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.FAIR)
        assert resized.shape[:2] == (512, 512)
    
    def test_resize_to_256_serious(self, large_image):
        """Large image should resize to 256x256 in SERIOUS state."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.SERIOUS)
        assert resized.shape[:2] == (256, 256)
    
    def test_resize_to_256_critical(self, large_image):
        """Large image should resize to 256x256 in CRITICAL state."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.CRITICAL)
        assert resized.shape[:2] == (256, 256)
    
    def test_upscale_small_image(self, small_image):
        """Small image should upscale to target resolution."""
        resized = ResolutionScaler.resize_image(small_image, ThermalState.NOMINAL)
        assert resized.shape[:2] == (768, 768)
    
    def test_preserves_channels(self, large_image):
        """Resizing should preserve 3 color channels."""
        resized = ResolutionScaler.resize_image(large_image, ThermalState.NOMINAL)
        assert resized.shape[2] == 3


class TestScaleFactor:
    """Test scale factor calculations."""
    
    def test_scale_factor_downscale(self):
        """Scale factor should be < 1 for downscaling."""
        factor = ResolutionScaler.get_scale_factor((1000, 1000), ThermalState.NOMINAL)
        assert factor < 1.0
        assert factor == 768 / 1000
    
    def test_scale_factor_upscale(self):
        """Scale factor should be > 1 for upscaling."""
        factor = ResolutionScaler.get_scale_factor((100, 100), ThermalState.NOMINAL)
        assert factor > 1.0
        assert factor == 768 / 100


class TestGetAllResolutions:
    """Test the get_all_resolutions utility method."""
    
    def test_returns_all_states(self):
        """Should return mappings for all thermal states."""
        resolutions = ResolutionScaler.get_all_resolutions()
        assert "nominal" in resolutions
        assert "fair" in resolutions
        assert "serious" in resolutions
        assert "critical" in resolutions
    
    def test_correct_values(self):
        """Should return correct resolution values."""
        resolutions = ResolutionScaler.get_all_resolutions()
        assert resolutions["nominal"] == (768, 768)
        assert resolutions["fair"] == (512, 512)
        assert resolutions["serious"] == (256, 256)
        assert resolutions["critical"] == (256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])