"""
convert_mot17.py — MOT17 データセット → YOAKE annotation JSON 変換

MOT17 フォルダ構造:
  MOT17/
    train/
      MOT17-02-DPM/
        gt/gt.txt          # ground truth: frame,id,x,y,w,h,conf,class,visibility
        img1/              # フレーム画像 (000001.jpg, 000002.jpg, ...)
        seqinfo.ini        # シーケンス情報 (fps, width, height, etc.)
      MOT17-04-DPM/
        ...
    test/
      ...

使い方:
  # train セットを変換 (train/val に分割)
  python tools/convert_mot17.py \\
      mot17_dir=data/MOT17/train \\
      output_dir=data/mot17_yoake \\
      val_ratio=0.2

  # test セットを変換 (アノテーションなし → 推論用)
  python tools/convert_mot17.py \\
      mot17_dir=data/MOT17/test \\
      output_dir=data/mot17_yoake \\
      split=test
"""

from __future__ import annotations

import configparser
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htrtdetr.data.annotation import (
    DatasetAnno, VideoAnno, FrameAnno, ObjectAnno,
    save_annotation, validate_annotation, print_annotation_stats,
)


# MOT17 gt.txt の class_id 定義
# 1=Pedestrian, 2=Person on vehicle, 3=Car, 4=Motorcycle, ...
# 検証のため pedestrian (1) のみ使用
MOT17_PEDESTRIAN_CLASS = 1


def parse_seqinfo(seqinfo_path: Path) -> dict:
    """seqinfo.ini からシーケンス情報を読み込む"""
    config = configparser.ConfigParser()
    config.read(str(seqinfo_path))
    seq = config["Sequence"]
    return {
        "name": seq.get("name", seqinfo_path.parent.name),
        "fps": float(seq.get("frameRate", "25")),
        "width": int(seq.get("imWidth", "1920")),
        "height": int(seq.get("imHeight", "1080")),
        "length": int(seq.get("seqLength", "0")),
        "img_dir": seq.get("imDir", "img1"),
        "img_ext": seq.get("imExt", ".jpg"),
    }


def load_mot17_sequence(
    seq_dir: Path,
    pedestrian_only: bool = True,
    min_visibility: float = 0.3,
) -> Optional[VideoAnno]:
    """
    MOT17 シーケンスディレクトリを読み込み VideoAnno を返す。

    Args:
        seq_dir: MOT17-XX-DPM などのディレクトリ
        pedestrian_only: True の場合 class_id=1 (pedestrian) のみ使用
        min_visibility: 可視度のフィルタ閾値
    """
    gt_path = seq_dir / "gt" / "gt.txt"
    seqinfo_path = seq_dir / "seqinfo.ini"

    if not gt_path.exists():
        print(f"  [SKIP] gt.txt not found: {gt_path}")
        return None

    # シーケンス情報
    if seqinfo_path.exists():
        info = parse_seqinfo(seqinfo_path)
    else:
        print(f"  [WARN] seqinfo.ini not found, using defaults")
        info = {"name": seq_dir.name, "fps": 25.0, "width": 1920, "height": 1080,
                "length": 0, "img_dir": "img1", "img_ext": ".jpg"}

    video_id = info["name"]
    img_dir = seq_dir / info["img_dir"]
    img_ext = info["img_ext"]

    print(f"  Loading: {video_id} ({info['width']}x{info['height']}, {info['fps']}fps)")

    # gt.txt 読み込み
    # 形式: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
    frames_map: Dict[int, List[ObjectAnno]] = defaultdict(list)
    obj_id_counter = 0

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            frame_idx = int(parts[0]) - 1  # MOT17 は 1-indexed → 0-indexed
            track_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            conf = int(float(parts[6])) if len(parts) > 6 else 1
            class_id_mot = int(parts[7]) if len(parts) > 7 else 1
            visibility = float(parts[8]) if len(parts) > 8 else 1.0

            # フィルタリング
            if conf == 0:  # conf=0 は ignore 領域
                continue
            if pedestrian_only and class_id_mot != MOT17_PEDESTRIAN_CLASS:
                continue
            if visibility < min_visibility:
                continue

            x1 = bb_left
            y1 = bb_top
            x2 = bb_left + bb_width
            y2 = bb_top + bb_height

            # 画像範囲内にクリップ
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(info["width"]), x2)
            y2 = min(float(info["height"]), y2)

            if x2 <= x1 or y2 <= y1:
                continue

            obj_id_counter += 1
            frames_map[frame_idx].append(ObjectAnno(
                object_id=obj_id_counter,
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                class_id=0,        # YOAKE では 0 = "person"
                track_id=track_id,
                action_id=-1,      # MOT17 に行動ラベルなし → unannotated
            ))

    if not frames_map:
        print(f"  [SKIP] No valid annotations in {video_id}")
        return None

    # FrameAnno 構築
    frame_annos = []
    for fi in sorted(frames_map.keys()):
        # 画像パスの構築
        img_filename = f"{fi + 1:06d}{img_ext}"  # MOT17 は 1-indexed ファイル名
        img_path = str(img_dir / img_filename)

        frame_annos.append(FrameAnno(
            frame_index=fi,
            image_path=img_path,
            width=info["width"],
            height=info["height"],
            objects=frames_map[fi],
        ))

    num_frames = len(frame_annos)
    num_objects = sum(len(f.objects) for f in frame_annos)
    print(f"    Frames: {num_frames}, Objects: {num_objects}, "
          f"Avg: {num_objects/max(num_frames,1):.1f} obj/frame")

    return VideoAnno(
        video_id=video_id,
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        num_frames=num_frames,
        frames=frame_annos,
    )


def convert_mot17(
    mot17_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    split: str = "train",  # "train" or "test"
    seed: int = 42,
    pedestrian_only: bool = True,
    min_visibility: float = 0.3,
    detector: str = "",  # "" = all detectors, "DPM", "FRCNN", "SDP"
) -> None:
    """
    MOT17 ディレクトリ全体を変換し、train/val JSON を生成する。

    Args:
        mot17_dir: MOT17/train または MOT17/test ディレクトリ
        output_dir: 出力先ディレクトリ
        val_ratio: validation に使用するシーケンスの割合
        split: "train" (gt.txt あり) or "test" (gt.txt なし)
        detector: 使用するディテクタ ("DPM", "FRCNN", "SDP", "" = 重複排除のため DPM のみ)
    """
    mot17_dir = Path(mot17_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MOT17 はシーケンスごとに DPM/FRCNN/SDP の3種類があるが
    # ground truth は同じなので 1 種類だけ使う
    use_detector = detector if detector else "DPM"

    print(f"Scanning: {mot17_dir}")
    seq_dirs = sorted([
        d for d in mot17_dir.iterdir()
        if d.is_dir() and (use_detector in d.name or not use_detector)
    ])

    if not seq_dirs:
        # detector フィルタなしで全シーケンス
        seq_dirs = sorted([d for d in mot17_dir.iterdir() if d.is_dir()])

    print(f"Found {len(seq_dirs)} sequences: {[d.name for d in seq_dirs]}")

    # 各シーケンスを読み込み
    videos = []
    for seq_dir in seq_dirs:
        video_anno = load_mot17_sequence(
            seq_dir,
            pedestrian_only=pedestrian_only,
            min_visibility=min_visibility,
        )
        if video_anno:
            videos.append(video_anno)

    if not videos:
        print("ERROR: No valid sequences loaded.")
        return

    print(f"\nLoaded {len(videos)} sequences")

    # class_names: MOT17 は pedestrian のみ使用
    # action_names: MOT17 に行動ラベルなし → "unknown" 1クラス (dummy)
    # 注意: Stage 2 (行動分類) は -1 (unannotated) のため学習できない
    #       Stage 1 (検出) と Stage 3 (追跡) のみ有効
    class_names = ["person"]
    action_names = ["unknown"]  # dummy — action_id は全て -1

    if split == "train":
        # train/val 分割
        rng = random.Random(seed)
        rng.shuffle(videos)
        n_val = max(1, int(len(videos) * val_ratio))
        val_videos = videos[:n_val]
        train_videos = videos[n_val:]

        # train annotation
        train_anno = DatasetAnno(
            meta={
                "version": "1.1",
                "description": "MOT17 pedestrian tracking (converted for YOAKE)",
                "source": "MOT Challenge - https://motchallenge.net/data/MOT17/",
                "split": "train",
                "notes": "action_id=-1 (unannotated). Use for Stage1 (detection) + Stage3 (tracking).",
            },
            class_names=class_names,
            action_names=action_names,
            videos=train_videos,
        )
        train_path = output_dir / "annotations_train.json"
        save_annotation(train_anno, str(train_path))
        print(f"\nTrain: {len(train_videos)} sequences → {train_path}")
        print_annotation_stats(train_anno)

        # validate
        val_result = validate_annotation(train_anno)
        if not val_result.valid:
            print(f"Validation errors: {val_result.errors}")
        else:
            print(f"Validation: OK (warnings: {len(val_result.warnings)})")

        # val annotation
        val_anno = DatasetAnno(
            meta={
                "version": "1.1",
                "description": "MOT17 pedestrian tracking (converted for YOAKE)",
                "source": "MOT Challenge - https://motchallenge.net/data/MOT17/",
                "split": "val",
            },
            class_names=class_names,
            action_names=action_names,
            videos=val_videos,
        )
        val_path = output_dir / "annotations_val.json"
        save_annotation(val_anno, str(val_path))
        print(f"\nVal:   {len(val_videos)} sequences → {val_path}")

    else:  # test (gt なし、推論のみ)
        # test セットには gt.txt がないため、フレームのみ登録
        frame_annos_all = []
        for seq_dir in seq_dirs:
            seqinfo_path = seq_dir / "seqinfo.ini"
            if not seqinfo_path.exists():
                continue
            info = parse_seqinfo(seqinfo_path)
            img_dir = seq_dir / info["img_dir"]
            frames = []
            for img_path in sorted(img_dir.glob(f"*{info['img_ext']}")):
                fi = int(img_path.stem) - 1
                frames.append(FrameAnno(
                    frame_index=fi,
                    image_path=str(img_path),
                    objects=[],
                ))
            videos.append(VideoAnno(
                video_id=info["name"],
                fps=info["fps"],
                width=info["width"],
                height=info["height"],
                num_frames=len(frames),
                frames=frames,
            ))

        test_anno = DatasetAnno(
            meta={"version": "1.1", "description": "MOT17 test (no gt)", "split": "test"},
            class_names=class_names,
            action_names=action_names,
            videos=videos,
        )
        test_path = output_dir / "annotations_test.json"
        save_annotation(test_anno, str(test_path))
        print(f"\nTest: {len(videos)} sequences → {test_path}")

    print("\nConversion complete!")
    print(f"Output dir: {output_dir}")
    print("\nNOTE: MOT17 has no action labels (action_id=-1).")
    print("      You can use Stage 1 (detection) and Stage 3 (tracking) normally.")
    print("      For Stage 2 (action), use config: model.action_head.num_actions=1")


def parse_args(argv: List[str]) -> dict:
    args = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            args[k] = v
    return args


if __name__ == "__main__":
    from typing import List
    args = parse_args(sys.argv[1:])

    mot17_dir = args.get("mot17_dir", "")
    output_dir = args.get("output_dir", "data/mot17_yoake")

    if not mot17_dir:
        print("Usage: python tools/convert_mot17.py mot17_dir=<path> output_dir=<path>")
        print("  mot17_dir : path to MOT17/train directory")
        print("  output_dir: output directory for YOAKE JSON files")
        print("  val_ratio : fraction of sequences for validation (default: 0.2)")
        print("  detector  : DPM | FRCNN | SDP (default: DPM)")
        sys.exit(1)

    convert_mot17(
        mot17_dir=mot17_dir,
        output_dir=output_dir,
        val_ratio=float(args.get("val_ratio", "0.2")),
        split=args.get("split", "train"),
        seed=int(args.get("seed", "42")),
        detector=args.get("detector", "DPM"),
        min_visibility=float(args.get("min_visibility", "0.3")),
    )
