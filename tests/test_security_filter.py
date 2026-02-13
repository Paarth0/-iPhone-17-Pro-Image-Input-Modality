#!/usr/bin/env python3
"""
Unit tests for SecurityFilter.
Validates CFR-3 (Blind Eye) and CFR-4 (Injection Guard).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.pipeline.security_filter import SecurityFilter, SecurityResult


class TestSecurityResult:
    """Test SecurityResult dataclass."""
    
    def test_default_values(self):
        """SecurityResult should have correct defaults."""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = SecurityResult(processed_image=dummy_image)
        
        assert result.blur_applied is False
        assert result.ocr_mask_applied is False
        assert result.detected_threats == []
        assert result.threat_regions == []
        assert result.is_safe is True


class TestSensitiveContentDetection:
    """Test Blind Eye - sensitive content detection."""
    
    @pytest.fixture
    def security_filter(self):
        """Create SecurityFilter without OCR for faster tests."""
        return SecurityFilter(use_ocr=False)
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy test image."""
        return np.zeros((256, 256, 3), dtype=np.uint8)
    
    def test_safe_filename_not_flagged(self, security_filter, dummy_image):
        """Normal filenames should not trigger blur."""
        result = security_filter.scan(dummy_image, "normal_photo.jpg")
        assert result.blur_applied is False
        assert result.is_safe is True
    
    def test_explicit_filename_flagged(self, security_filter, dummy_image):
        """'explicit' in filename should trigger blur."""
        result = security_filter.scan(dummy_image, "explicit_content.jpg")
        assert result.blur_applied is True
        assert result.is_safe is False
        assert "SENSITIVE_CONTENT" in result.detected_threats
    
    def test_nsfw_filename_flagged(self, security_filter, dummy_image):
        """'nsfw' in filename should trigger blur."""
        result = security_filter.scan(dummy_image, "nsfw_image.png")
        assert result.blur_applied is True
        assert result.is_safe is False
    
    def test_unsafe_filename_flagged(self, security_filter, dummy_image):
        """'unsafe' in filename should trigger blur."""
        result = security_filter.scan(dummy_image, "unsafe_test.jpg")
        assert result.blur_applied is True
    
    def test_adult_filename_flagged(self, security_filter, dummy_image):
        """'adult' in filename should trigger blur."""
        result = security_filter.scan(dummy_image, "adult_material.jpg")
        assert result.blur_applied is True
    
    def test_sensitive_filename_flagged(self, security_filter, dummy_image):
        """'sensitive' in filename should trigger blur."""
        result = security_filter.scan(dummy_image, "sensitive_data.jpg")
        assert result.blur_applied is True
    
    def test_case_insensitive(self, security_filter, dummy_image):
        """Detection should be case insensitive."""
        result = security_filter.scan(dummy_image, "EXPLICIT_IMAGE.JPG")
        assert result.blur_applied is True
    
    def test_empty_filename_safe(self, security_filter, dummy_image):
        """Empty filename should be considered safe."""
        result = security_filter.scan(dummy_image, "")
        assert result.blur_applied is False
        assert result.is_safe is True


class TestBlurApplication:
    """Test blur functionality."""
    
    @pytest.fixture
    def security_filter(self):
        return SecurityFilter(use_ocr=False)
    
    def test_blur_changes_image(self, security_filter):
        """Blur should modify the image."""
        # Create image with some content
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[100:150, 100:150] = 255  # White square
        
        result = security_filter.scan(image, "explicit_test.jpg")
        
        # Blurred image should be different from original
        assert not np.array_equal(result.processed_image, image)
    
    def test_blur_preserves_shape(self, security_filter):
        """Blur should preserve image dimensions."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        result = security_filter.scan(image, "explicit_test.jpg")
        
        assert result.processed_image.shape == image.shape


class TestAdversarialKeywords:
    """Test adversarial keyword list."""
    
    def test_keywords_exist(self):
        """Should have adversarial keywords defined."""
        keywords = SecurityFilter.get_adversarial_keywords()
        assert len(keywords) > 0
    
    def test_expected_keywords_present(self):
        """Should include expected keywords."""
        keywords = SecurityFilter.get_adversarial_keywords()
        assert "override" in keywords
        assert "system" in keywords
        assert "ignore" in keywords
        assert "bypass" in keywords


class TestOCRDisabled:
    """Test behavior when OCR is disabled."""
    
    def test_no_ocr_scan_when_disabled(self):
        """Should not perform OCR when disabled."""
        filter = SecurityFilter(use_ocr=False)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        result = filter.scan(image, "test.jpg")
        
        assert result.ocr_mask_applied is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])