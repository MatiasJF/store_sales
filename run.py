#!/usr/bin/env python3
"""
Store Sales Forecasting — Autonomous Pipeline
Run: python run.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_pipeline


if __name__ == "__main__":
    try:
        submission = run_pipeline()
        print(f"\nDone. {len(submission)} predictions written.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
