"""
smoke_test.py — 動作確認用スモークテスト

実データなしで YOAKE の全パイプラインが動作することを確認する。
GPU が不要で、CPU 上で完結する。

使い方:
    python examples/smoke_test.py
    python examples/smoke_test.py stage=all
    python examples/smoke_test.py stage=1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from htrtdetr.config import (
    get_stage1_config, HTRTDETRConfig,
    ModelConfig, DetectorConfig, DetectorHeadConfig
)
from htrtdetr.models import build_model, HTRTDETR
from htrtdetr.data import DummyDataset, get_collate_fn
from htrtdetr.utils import count_parameters, get_device


def test_model_forward() -> None:
    """モデルの forward パスのテスト"""
    print("\n" + "="*50)
    print("Test: Model Forward Pass")
    print("="*50)

    # 小さい設定でテスト
    cfg = HTRTDETRConfig()
    cfg.model.detector.backbone.pretrained = False  # テスト時は pretrained を使わない
    cfg.model.detector.head.num_queries = 20
    cfg.model.detector.head.num_decoder_layers = 2
    cfg.model.id_head.max_ids = 10
    cfg.model.action_head.num_actions = 5

    model = build_model(cfg.model)
    model.eval()

    device = torch.device("cpu")
    model = model.to(device)

    params = model.num_parameters()
    print(f"Parameters:")
    for name, n in params.items():
        print(f"  {name}: {n:,}")

    # Stage 1 forward (single frame)
    model.set_stage(1)
    B, C, H, W = 2, 3, 256, 256
    images = torch.randn(B, C, H, W)

    with torch.no_grad():
        out = model.forward_single_frame(images)

    print(f"\nStage 1 output:")
    print(f"  pred_logits: {out.pred_logits.shape}")  # (B, Q, C+1)
    print(f"  pred_boxes:  {out.pred_boxes.shape}")   # (B, Q, 4)
    print(f"  query_feats: {out.query_features.shape}")
    assert out.pred_logits.shape[0] == B
    assert not torch.isnan(out.pred_logits).any(), "NaN in pred_logits!"
    print("  [PASS] Stage 1 forward")

    # Stage 2 forward (sequence)
    model.set_stage(2)
    T = 8
    images_seq = torch.randn(B, T, C, H, W)

    with torch.no_grad():
        out2 = model(images_seq)

    print(f"\nStage 2 output:")
    print(f"  pred_logits:    {out2.pred_logits.shape}")
    print(f"  action_logits:  {out2.action_logits.shape if out2.action_logits is not None else None}")
    assert not (out2.action_logits is not None and torch.isnan(out2.action_logits).any()), "NaN in action_logits!"
    print("  [PASS] Stage 2 forward")

    # Stage 3 forward (ID)
    model.set_stage(3)
    with torch.no_grad():
        out3 = model(images_seq)
    print(f"\nStage 3 output:")
    print(f"  id_logits:   {out3.id_logits.shape if out3.id_logits is not None else None}")
    print("  [PASS] Stage 3 forward")

    print("\n[PASS] All forward passes complete!")


def test_loss_computation() -> None:
    """損失計算のテスト"""
    print("\n" + "="*50)
    print("Test: Loss Computation")
    print("="*50)

    from htrtdetr.training.losses import DetectionLoss, ActionLoss, IDLoss
    from htrtdetr.config import LossConfig

    loss_cfg = LossConfig()
    det_loss = DetectionLoss(loss_cfg, num_classes=1)
    act_loss = ActionLoss(loss_cfg, num_actions=5)
    id_loss = IDLoss(loss_cfg, max_ids=10)

    # Detection loss テスト
    B, Q = 2, 20
    pred_logits = torch.randn(B, Q, 2)  # 1 class + background
    pred_boxes = torch.sigmoid(torch.randn(B, Q, 4))

    targets = []
    for b in range(B):
        n = 3
        boxes = torch.rand(n, 4)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:].clamp(min=0.1)
        boxes = boxes.clamp(0, 1)
        targets.append({
            "boxes": boxes,
            "class_ids": torch.zeros(n, dtype=torch.long),
        })

    try:
        losses = det_loss(pred_logits, pred_boxes, targets)
        print(f"Detection losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")
        print("  [PASS] Detection loss")
    except ImportError as e:
        print(f"  [SKIP] scipy not available: {e}")

    # Action loss テスト
    N = 6
    action_logits = torch.randn(N, 5)
    gt_actions = torch.randint(0, 5, (N,))
    gt_actions[0] = -1  # ignore test

    loss_a = act_loss(action_logits, gt_actions)
    print(f"Action loss: {loss_a.item():.4f}")
    print("  [PASS] Action loss")

    # ID loss テスト
    id_logits = torch.randn(N, 11)  # max_ids=10 + 1
    gt_tracks = torch.randint(0, 10, (N,))
    result = id_loss(id_logits, gt_tracks)
    print(f"ID loss: { {k: f'{v.item():.4f}' for k, v in result.items()} }")
    print("  [PASS] ID loss")


def test_dataset() -> None:
    """Dataset / DataLoader のテスト"""
    print("\n" + "="*50)
    print("Test: Dataset and DataLoader")
    print("="*50)

    from torch.utils.data import DataLoader

    # Single frame
    ds_single = DummyDataset(n_samples=10, mode="single", image_size=(256, 256))
    item = ds_single[0]
    print(f"SingleFrame item keys: {list(item.keys())}")
    print(f"  image: {item['image'].shape}")
    print(f"  boxes: {item['boxes'].shape}")

    loader_single = DataLoader(
        ds_single, batch_size=2, collate_fn=get_collate_fn("single")
    )
    batch = next(iter(loader_single))
    print(f"  batch images: {batch['images'].shape}")
    print("  [PASS] SingleFrameDataset")

    # Sequence
    ds_seq = DummyDataset(n_samples=10, mode="sequence", window_size=8, image_size=(256, 256))
    item = ds_seq[0]
    print(f"\nSequence item keys: {list(item.keys())}")
    print(f"  images: {item['images'].shape}")

    loader_seq = DataLoader(
        ds_seq, batch_size=2, collate_fn=get_collate_fn("sequence")
    )
    batch = next(iter(loader_seq))
    print(f"  batch images: {batch['images'].shape}")
    print("  [PASS] SlidingWindowDataset (DummyDataset)")


def main() -> None:
    args = sys.argv[1:]
    stage_arg = "all"
    for arg in args:
        if arg.startswith("stage="):
            stage_arg = arg.split("=")[1]

    print("YOAKE Smoke Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {get_device()}")

    try:
        test_dataset()
    except Exception as e:
        print(f"[FAIL] Dataset test: {e}")
        import traceback; traceback.print_exc()

    try:
        test_model_forward()
    except Exception as e:
        print(f"[FAIL] Model forward test: {e}")
        import traceback; traceback.print_exc()

    try:
        test_loss_computation()
    except Exception as e:
        print(f"[FAIL] Loss test: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*50)
    print("Smoke Test Complete")
    print("="*50)


if __name__ == "__main__":
    main()
