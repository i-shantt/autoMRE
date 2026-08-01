#!/usr/bin/env python3
"""
autoMRE: Automated Bug Reproduction Minimization

Main entry point for the tool.
"""

import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "automre" / "src"))

from cli import main

if __name__ == '__main__':
    sys.exit(main())
