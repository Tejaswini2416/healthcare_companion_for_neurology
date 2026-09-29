#!/usr/bin/env python3
"""
Root Workspace Launcher for Glioma Digital Twin System
Directly executes glioma_digital_twin/run_demo.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sub_path = os.path.join(current_dir, "glioma_digital_twin", "run_demo.py")

if __name__ == "__main__":
    if os.path.exists(sub_path):
        import runpy
        sys.path.insert(0, os.path.join(current_dir, "glioma_digital_twin"))
        sys.path.insert(0, current_dir)
        runpy.run_path(sub_path, run_name="__main__")
    else:
        print(f"Error: Could not locate {sub_path}")
