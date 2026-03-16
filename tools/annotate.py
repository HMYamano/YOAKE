#!/usr/bin/env python3
"""
YOAKE Annotation Tool – launcher
Usage:
    python tools/annotate.py [video_path] [--load annotations.json]
"""
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from htrtdetr.annotator import main

if __name__ == "__main__":
    main()
