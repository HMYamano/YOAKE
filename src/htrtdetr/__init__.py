"""
htrtdetr — Hierarchical Temporal RT-DETR
動物行動解析アルゴリズム

Author: YOAKE Research Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "YOAKE Research Team"

from .config import HTRTDETRConfig, get_stage1_config, get_stage2_config, get_stage3_config, get_stage4_config

try:
    from .models import HTRTDETR, build_model
except ImportError:
    pass  # torch not installed

__all__ = [
    "HTRTDETR",
    "build_model",
    "HTRTDETRConfig",
    "get_stage1_config",
    "get_stage2_config",
    "get_stage3_config",
    "get_stage4_config",
]
