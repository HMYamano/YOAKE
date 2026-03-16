# Design Assumptions

本実装で採用した仮定・設計判断を記録する。

## データ

1. **アノテーション形式**: 独自 JSON 形式をベースとし、MOT 形式への変換コンバーターを提供。
   COCO 形式は将来対応。

2. **個体数**: 同一フレームに最大 50 個体を前提。
   `max_ids=50` で変更可能。

3. **動画 fps**: デフォルト 30fps。低速撮影 (100fps+) の場合は `window_size` と
   branch の `num_frames` を調整すること。

4. **action_id=-1**: アノテーションなしフレームを ignore。
   損失計算時は `ignore_index=-1` でスキップ。

## モデル

5. **Backbone**: ResNet-18 (ImageNet pretrained)。BSD ライセンス (商用利用可)。
   ResNet-50 への切り替えは `BackboneConfig.name="resnet50"` のみ。

6. **DETR Decoder**: 標準 Multi-head Attention を使用。
   公式 RT-DETR の Deformable Attention は未実装。
   → 精度向上が必要な場合は `models/detector/rtdetr_wrapper.py` の `TransformerDecoderLayer` を
     Deformable Attention に差し替え可能。

7. **Positional Encoding**: Sinusoidal 2D PE を encoder 側に使用。
   Decoder の query には使わない (DETR 原著の設計に準拠)。

8. **NMS**: DETR は本来 NMS 不要だが、精度保険として score threshold のみ使用。
   必要に応じて `utils/misc.py` の `nms()` を推論時に追加可能。

9. **ID Memory の GRU**: 単純な GRUCell を使用。
   BiGRU / Attention-based Memory は将来拡張として設計余地あり。

10. **Metric Loss**: Triplet loss を実装。
    Contrastive loss / ArcFace は将来オプション。

## 学習

11. **Hungarian Matching**: scipy の `linear_sum_assignment` を使用。
    scipy がない場合は greedy fallback (精度低下あり)。

12. **AMP**: torch の native AMP を使用 (apex 不要)。

13. **GT ID matching**: Stage 3/4 の学習時に、detection result と GT の
    proper な対応付け (Hungarian) が必要。現実装では `-1` (ignore) を返す
    placeholder になっており、**実データ使用前に要実装**。
    → `training/losses.py` の `_collect_gt_actions()` / `_collect_gt_track_ids()` を参照。

## 評価

14. **IDF1 の簡略化**: 厳密な CLEAR MOT metrics の代わりに簡略版を実装。
    motmetrics ライブラリへの差し替えで厳密計算が可能。

15. **AP 計算**: 11-point interpolation (PASCAL VOC 方式)。
    COCO 方式 (101-point) への変更は `evaluation/evaluator.py` を参照。

## 推論

16. **Memory はシーケンス間でリセット**: 動画ごとに IdentityMemory を初期化。
    シーン切り替え検出による自動リセットは optional。

17. **ONNX Export**: 現時点では `torch.onnx.export()` で直接エクスポート可能な設計。
    GRUCell / MultiheadAttention は標準 ONNX op でサポートされている。
    一部動的シェイプ (可変個体数) は処理が必要。

## ライセンス

18. **torchvision (BSD License)**: ResNet を torchvision 経由で使用。
    商用利用可。

19. **scipy (BSD License)**: Hungarian matching に使用。商用利用可。

20. **OpenCV (Apache-2.0)**: 推論時の動画 I/O に使用。商用利用可。
    optional 依存として実装 (なければ PIL + fallback)。
