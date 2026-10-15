#!/usr/bin/env python3
"""Convenience root launcher for python/plot_validation.py"""
import sys
from pathlib import Path

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).parent))

from python.plot_validation import main

if __name__ == "__main__":
    main()
