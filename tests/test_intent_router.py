#!/usr/bin/env python3
"""
Unit tests for IntentRouter.
Validates CFR-2: Intent Classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.model.intent_router import IntentRouter, Intent


class TestIntentEnum:
    """Test Intent enum values."""
    
    def test_intent_a_value(self):
        assert Intent.INTENT_A_PRACTICAL_GUIDANCE.value == "INTENT_A_PRACTICAL_GUIDANCE"
    
    def test_intent_b_value(self):
        assert Intent.INTENT_B_DISCOVERY.value == "INTENT_B_DISCOVERY"
    
    def test_intent_c_value(self):
        assert Intent.INTENT_C_CREATIVE.value == "INTENT_C_CREATIVE"
    
    def test_unknown_value(self):
        assert Intent.UNKNOWN.value == "UNKNOWN"


class TestClassIDRouting:
    """Test routing by ImageNet class ID."""
    
    @pytest.fixture
    def router(self):
        return IntentRouter()
    
    def test_binder_routes_to_intent_a(self, router):
        """Class 446 (binder) should route to Intent A."""
        intent = router.route(446, "binder")
        assert intent == Intent.INTENT_A_PRACTICAL_GUIDANCE
    
    def test_notebook_routes_to_intent_a(self, router):
        """Class 695 (notebook) should route to Intent A."""
        intent = router.route(695, "notebook")
        assert intent == Intent.INTENT_A_PRACTICAL_GUIDANCE
    
    def test_cat_routes_to_intent_b(self, router):
        """Class 281 (tabby_cat) should route to Intent B."""
        intent = router.route(281, "tabby_cat")
        assert intent == Intent.INTENT_B_DISCOVERY
    
    def test_laptop_routes_to_intent_b(self, router):
        """Class 620 (laptop) should route to Intent B."""
        intent = router.route(620, "laptop")
        assert intent == Intent.INTENT_B_DISCOVERY
    
    def test_volcano_routes_to_intent_c(self, router):
        """Class 980 (volcano) should route to Intent C."""
        intent = router.route(980, "volcano")
        assert intent == Intent.INTENT_C_CREATIVE
    
    def test_seashore_routes_to_intent_c(self, router):
        """Class 978 (seashore) should route to Intent C."""
        intent = router.route(978, "seashore")
        assert intent == Intent.INTENT_C_CREATIVE


class TestKeywordRouting:
    """Test routing by class name keywords."""
    
    @pytest.fixture
    def router(self):
        return IntentRouter()
    
    def test_book_keyword_routes_to_intent_a(self, router):
        """Class name with 'book' should route to Intent A."""
        intent = router.route(9999, "comic_book")
        assert intent == Intent.INTENT_A_PRACTICAL_GUIDANCE
    
    def test_document_keyword_routes_to_intent_a(self, router):
        """Class name with 'document' should route to Intent A."""
        intent = router.route(9999, "legal_document")
        assert intent == Intent.INTENT_A_PRACTICAL_GUIDANCE
    
    def test_cat_keyword_routes_to_intent_b(self, router):
        """Class name with 'cat' should route to Intent B."""
        intent = router.route(9999, "wild_cat")
        assert intent == Intent.INTENT_B_DISCOVERY
    
    def test_phone_keyword_routes_to_intent_b(self, router):
        """Class name with 'phone' should route to Intent B."""
        intent = router.route(9999, "smart_phone")
        assert intent == Intent.INTENT_B_DISCOVERY
    
    def test_mountain_keyword_routes_to_intent_c(self, router):
        """Class name with 'mountain' should route to Intent C."""
        intent = router.route(9999, "snow_mountain")
        assert intent == Intent.INTENT_C_CREATIVE
    
    def test_sunset_keyword_routes_to_intent_c(self, router):
        """Class name with 'sunset' should route to Intent C."""
        intent = router.route(9999, "beautiful_sunset")
        assert intent == Intent.INTENT_C_CREATIVE


class TestDefaultRouting:
    """Test default routing behavior."""
    
    @pytest.fixture
    def router(self):
        return IntentRouter()
    
    def test_unknown_class_defaults_to_intent_b(self, router):
        """Unknown classes should default to Intent B (Discovery)."""
        intent = router.route(9999, "random_unknown_class")
        assert intent == Intent.INTENT_B_DISCOVERY


class TestBatchRouting:
    """Test batch routing functionality."""
    
    @pytest.fixture
    def router(self):
        return IntentRouter()
    
    def test_batch_route_multiple(self, router):
        """Should route multiple predictions correctly."""
        predictions = [
            (446, "binder"),
            (281, "tabby_cat"),
            (980, "volcano"),
        ]
        
        intents = router.route_batch(predictions)
        
        assert len(intents) == 3
        assert intents[0] == Intent.INTENT_A_PRACTICAL_GUIDANCE
        assert intents[1] == Intent.INTENT_B_DISCOVERY
        assert intents[2] == Intent.INTENT_C_CREATIVE


class TestIntentDescriptions:
    """Test intent description functionality."""
    
    @pytest.fixture
    def router(self):
        return IntentRouter()
    
    def test_intent_a_has_description(self, router):
        """Intent A should have a description."""
        desc = router.get_intent_description(Intent.INTENT_A_PRACTICAL_GUIDANCE)
        assert isinstance(desc, str)
        assert len(desc) > 0
    
    def test_intent_b_has_description(self, router):
        """Intent B should have a description."""
        desc = router.get_intent_description(Intent.INTENT_B_DISCOVERY)
        assert isinstance(desc, str)
        assert len(desc) > 0
    
    def test_intent_c_has_description(self, router):
        """Intent C should have a description."""
        desc = router.get_intent_description(Intent.INTENT_C_CREATIVE)
        assert isinstance(desc, str)
        assert len(desc) > 0


class TestRoutingStats:
    """Test routing statistics."""
    
    def test_stats_returns_dict(self):
        """get_routing_stats should return a dictionary."""
        router = IntentRouter()
        stats = router.get_routing_stats()
        
        assert isinstance(stats, dict)
        assert "intent_a_class_count" in stats
        assert "intent_b_class_count" in stats
        assert "intent_c_class_count" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])