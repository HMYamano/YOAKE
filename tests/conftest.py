"""共通フィクスチャ"""
import sys
from pathlib import Path

# リポジトリルートを sys.path に追加
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
