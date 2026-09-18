"""
``python -m htrtdetr.gui`` のエントリポイント。

README が案内する 3 つの起動方法 — ``yoake gui`` / ``python -m htrtdetr.gui`` /
``gui`` — はどれも ``htrtdetr.gui.main`` (= ``hub.main`` への遅延ディスパッチ) に
行き着く。``__main__.py`` が無いと ``python -m htrtdetr.gui`` だけが
「No module named htrtdetr.gui.__main__」で落ちるので、ここで揃える。

dearpygui の import は ``main()`` の中で起きるため、このファイルを import する
だけでは GUI 依存を要求しない (torch なし・dearpygui なしの環境でも
``import htrtdetr.gui`` は通る、という既存の方針を壊さない)。
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
