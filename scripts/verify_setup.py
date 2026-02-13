#!/usr/bin/env python3
"""
Verify project setup is correct.
"""

import sys
import os
from pathlib import Path

# Fix: Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify():
    """Verify project structure and imports."""
    print("Verifying project setup...\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("⚠️  Warning: Python 3.10+ recommended (3.9 should work)")
    else:
        print("✓ Python version OK")
    
    print(f"\nProject root: {PROJECT_ROOT}")
    
    # Check directory structure
    print("\nChecking directories...")
    required_dirs = [
        "src", "src/pipeline", "src/model", "src/config",
        "models", "data/input", "data/output", "tests", "scripts"
    ]
    
    all_dirs_ok = True
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            all_dirs_ok = False
    
    # Check __init__.py files exist
    print("\nChecking __init__.py files...")
    init_files = [
        "src/__init__.py",
        "src/pipeline/__init__.py",
        "src/model/__init__.py",
        "src/config/__init__.py",
    ]
    
    for init_file in init_files:
        init_path = PROJECT_ROOT / init_file
        if init_path.exists():
            print(f"  ✓ {init_file}")
        else:
            print(f"  ✗ {init_file} - MISSING (creating...)")
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.touch()
            print(f"    → Created {init_file}")
    
    # Check imports
    print("\nChecking imports...")
    
    import_tests = [
        ("ResolutionScaler", "from src.pipeline.resolution_scaler import ResolutionScaler, ThermalState"),
        ("SecurityFilter", "from src.pipeline.security_filter import SecurityFilter, SecurityResult"),
        ("VisionEncoder", "from src.model.vision_encoder import VisionEncoder, EncodingResult"),
        ("IntentRouter", "from src.model.intent_router import IntentRouter, Intent"),
        ("Settings", "from src.config.settings import Settings"),
    ]
    
    import_success = True
    for name, import_statement in import_tests:
        try:
            exec(import_statement)
            print(f"  ✓ {name} imported")
        except ImportError as e:
            print(f"  ✗ {name} import failed: {e}")
            import_success = False
        except Exception as e:
            print(f"  ✗ {name} error: {e}")
            import_success = False
    
    if not import_success:
        print("\n⚠️  Import errors detected. Checking module files...")
        module_files = [
            "src/pipeline/resolution_scaler.py",
            "src/pipeline/security_filter.py",
            "src/model/vision_encoder.py",
            "src/model/intent_router.py",
            "src/config/settings.py",
        ]
        for mod_file in module_files:
            mod_path = PROJECT_ROOT / mod_file
            if mod_path.exists():
                size = mod_path.stat().st_size
                print(f"    {mod_file}: {size} bytes")
            else:
                print(f"    {mod_file}: MISSING")
        return False
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    try:
        from src.pipeline.resolution_scaler import ResolutionScaler, ThermalState
        result = ResolutionScaler.get_resolution(ThermalState.NOMINAL)
        assert result == (768, 768), f"Expected (768, 768), got {result}"
        print("  ✓ ResolutionScaler.get_resolution() works")
    except Exception as e:
        print(f"  ✗ ResolutionScaler test failed: {e}")
    
    try:
        from src.model.intent_router import IntentRouter, Intent
        router = IntentRouter()
        desc = router.get_intent_description(Intent.INTENT_A_PRACTICAL_GUIDANCE)
        assert len(desc) > 0, "Description should not be empty"
        print("  ✓ IntentRouter.get_intent_description() works")
    except Exception as e:
        print(f"  ✗ IntentRouter test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Setup verification complete!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)