"""
Unified test runner for all ML components:
- Preprocessing tests
- Feature extraction & patient leakage tests
- Model evaluation & metrics tests
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    ml_root = PROJECT_ROOT / "ml"
    suite.addTests(loader.discover(str(ml_root / "preprocessing" / "tests"), top_level_dir=str(PROJECT_ROOT), pattern="test_*.py"))
    suite.addTests(loader.discover(str(ml_root / "features" / "tests"), top_level_dir=str(PROJECT_ROOT), pattern="test_*.py"))
    suite.addTests(loader.discover(str(ml_root / "evaluation" / "tests"), top_level_dir=str(PROJECT_ROOT), pattern="test_*.py"))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    if not result.wasSuccessful():
        exit(1)
