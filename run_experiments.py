#!/usr/bin/env python
"""
Root runner for VGG16 experiments.
Delegates to scripts/run_experiments.py.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiments import main

if __name__ == "__main__":
    main()

