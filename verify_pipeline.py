#!/usr/bin/env python
"""
Root runner for pipeline verification and smoke test.
Delegates to scripts/verify_pipeline.py.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_pipeline import run_smoke_test

if __name__ == "__main__":
    run_smoke_test()

