"""
htrtdetr.gui — YOAKE GUI スイート (DearPyGui)
=============================================

YORU (github.com/Kamikouchi-lab/YORU) の情報設計を参考にした統一ハブ。
単一ウィンドウの左ナビから Labeling / Training / Video Analysis / Evaluation を切替。

このパッケージは 2 種類のモジュールに分かれる:

- **純モジュール (dearpygui 非依存・単体テスト可)**:
  ``labeling_ops`` (bbox 補間) / ``monitor`` (results.csv・metrics_latest.json パーサ) /
  ``predictions`` (predictions.json ローダ) / ``commands`` (yoake コマンド生成)。
- **UI モジュール (dearpygui 依存)**:
  ``hub`` / ``canvas`` / ``training_panel`` / ``analysis_panel`` / ``runner``。

``htrtdetr.gui`` を import しただけで dearpygui を要求しないよう、UI モジュールは
トップレベルでは読み込まない (``main()`` から遅延 import する)。
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> None:
    """統一ハブ GUI を起動する (``htrtdetr.gui.hub:main`` への遅延ディスパッチ)。"""
    from .hub import main as _main

    _main(argv)
