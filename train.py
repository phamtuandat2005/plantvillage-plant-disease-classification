#!/usr/bin/env python
"""
Root entry point for PlantVillage VGG16 Training CLI.
Delegates to scripts/train.py.
"""
import os
import sys
from pathlib import Path

# Suppress TF verbose C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import main

if __name__ == "__main__":
    main()

