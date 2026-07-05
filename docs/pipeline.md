# YOAKE レイヤードパイプライン (pipeline / stages)

物体検出を土台に、その上へ時系列解析 (L1→L5) を段階的に積み重ねられる宣言的
パイプライン。**2プレーン構成**:

- **Compute プレーン** (`src/htrtdetr/models/`): 既存の学習済みモデル群。各段の「中身」。
- **Orchestration プレーン** (`src/htrtdetr/pipeline/` + `src/htrtdetr/stages/`): 新規。
  L1〜L5 を capability トークンで結び、依存解決・差し替え・スキップを担う。

## レイヤー

| 段 | tier | 役割 | provides | 代表バックエンド |
|---|---|---|---|---|
| L1 検出 | 1 | 物体検出 | boxes, class_id, det_score (+dense_features) | `rtdetr`, `yolo`, `fused_htrtdetr` |
| L1b 安定化 | 1 | ID なしで検出を時系列安定化 | temporal_stable | `temporal_nms` |
| L2 静的関係 | 2 | 近接/重なり/上下 | static_relation | `geometric` |
| L3 追跡 | 3 | ID 付与 | track_id (+appearance_embedding) | `memory_id`, `bytetrack`, `iou` |
| L4 運動 | 4 | 速度/向き/軌跡 | kinematics | `kinematic` |
| pose (任意) | 4 | キーポイント/体軸 | pose | `keypoint` |
| L5 相互作用 | 5 | 誰が誰に (有向) | directed_interaction | `heuristic`, `learned_pair`, `gnn` |

各段は `requires` / `provides` (capability 文字列) と `tier` / `after` (順序) を宣言する。
`resolver` がトポロジカルソートで実行順を決め、依存を検証する:

- **strict**: 満たされない capability があれば実行前にエラー (無効な組合せを事前検出)
- **lenient**: 満たせない段と、その下流を自動スキップ

## 解析モード = config の差し替えだけ

| モード | 使う config | 段 |
|---|---|---|
| ① 物体検出のみ | `detect_only.yaml` | L1 |
| ② 時系列を加味した検出 | `temporal_detect.yaml` | L1 + L1b |
| ③ 検出 + ID 追跡 | `detect_track.yaml` | L1(fused) + L1b |
| ④ フル解析 | `full_offline.yaml` | L1〜L5 |
| 差し替え例 | `yolo_bytetrack.yaml` | YOLO × ByteTrack × 運動 × heuristic |

```bash
yoake pipeline config=configs/pipeline/full_offline.yaml source=data/videos/clip.mp4 \
      output_dir=runs/pipeline
# → runs/pipeline/predictions.json (per-frame 検出/track_id/relations)
#    runs/pipeline/interactions.json (有向相互作用 A→B)
```

## 差し替えの実例 (要件1+2 の実証)

- `L1=yolo` (dense_features を出さない) × `L3=memory_id` (dense_features 必須) は
  **strict なら実行前にエラー**、**lenient なら memory_id と下流を自動スキップ**。
- `L3` を `bytetrack` (boxes だけで動く) に替えれば YOLO と問題なく繋がり L4/L5 まで到達。

## L1b (時系列検出安定化) と L3 (追跡) の違い

L1b は ID を振らず、窓内の使い捨て IoU 連結で **誤検出抑制 (min_persist)**・
**短ギャップ補間 (`interpolated=True`)**・**スコア平滑化 (`raw_score` に原値)** のみ行う。
速度/向きは出さない (それは L4)。L3 の前に置くと、安定化済み検出で偽トラック/ID切替が減る。

## 学習済み統合モデルとの両立

`fused_htrtdetr` バックエンドは既存 `HTRTDETR` を sliding window で回し、
`inference/inferencer.py` と同じ最終フレーム推論を再現して boxes/class/score/dense/track_id を
一括供給する (JSONパリティ経路)。個別段 (`rtdetr` + `bytetrack` + `kinematic` …) と
config で相互に切り替えられる。

## 拡張

- 新しい段: `StageBase` を継承し `@register("Lx_layer.backend")` で登録、
  `requires/provides/tier/after` を宣言するだけ。`stages/__init__.py` に import を追加。
- capability トークンは `pipeline/schema.py` の `KNOWN_CAPABILITIES` に追記
  (未知トークンは登録時に警告)。
