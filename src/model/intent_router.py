"""
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
