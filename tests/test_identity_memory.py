"""
test_identity_memory.py — ID subsystem の責務と memory の生存を固定する

検証する 3 点:

1. **GRU memory が実際に次フレームへ引き継がれる** (P1-6)。
   保存だけして毎フレームゼロから計算し直していると、``IdentityMemory`` は
   名前だけの飾りになる。同じ入力を与え続けても hidden が変化することで、
   前フレームの状態が計算に効いていることを確かめる。
2. **track が消えたら memory も消える** (memory lifetime)。
3. **ID assignment authority がひとつ** (P1-7)。
   ``OnlineTracker`` が final track ID を決め、``MemoryIDHead`` は embedding を
   作って確定 ID で memory を進めるだけ。
"""

from __future__ import annotations

import pytest
import torch

from htrtdetr.config.config import MemoryIDConfig
from htrtdetr.models.id_head import MemoryIDHead


@pytest.fixture
def head() -> MemoryIDHead:
    torch.manual_seed(0)
    return MemoryIDHead(MemoryIDConfig()).eval()


def _det(head: MemoryIDHead, n: int = 1):
    feats = torch.randn(n, head.cfg.feature_dim)
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]]).repeat(n, 1)
    return feats, boxes


class TestGruMemoryIsCarriedForward:
    def test_hidden_state_changes_across_frames_with_identical_input(self, head):
        """
        入力が毎フレーム同じでも hidden が変わる = 前フレームの hidden を読んでいる。

        修正前は ``hidden_states = torch.zeros(...)`` から毎回始めていたので、
        同じ入力に対して hidden は常に同じ値になっていた (保存した値は二度と
        読まれない dead state)。
        """
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head)

        hiddens = []
        for _ in range(4):
            out = head.forward_inference(feats, boxes, mem)
            tid = int(out["track_ids"][0])
            hiddens.append(mem.tracks[tid]["gru_hidden"].clone())

        assert len(hiddens) == 4
        for i in range(1, len(hiddens)):
            assert not torch.allclose(hiddens[i], hiddens[i - 1]), (
                f"frame {i} の hidden が frame {i-1} と同一 — "
                "前フレームの memory が使われていない"
            )

    def test_same_track_keeps_its_own_id_across_frames(self, head):
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head)
        ids = [
            int(head.forward_inference(feats, boxes, mem)["track_ids"][0])
            for _ in range(3)
        ]
        assert len(set(ids)) == 1, f"同じ個体に別 ID が付いた: {ids}"

    def test_gather_previous_state_returns_zero_for_unknown_track(self, head):
        mem = head.create_memory(torch.device("cpu"))
        hidden, vel = mem.gather_previous_state(
            [-1, 999], head.cfg.memory_dim, torch.device("cpu")
        )
        assert hidden.shape == (2, head.cfg.memory_dim)
        assert torch.count_nonzero(hidden) == 0
        assert torch.count_nonzero(vel) == 0

    def test_memory_is_dropped_when_track_expires(self, head):
        """TTL 切れで track が消えたら hidden も消える (再出現は新しい track)。"""
        cfg = MemoryIDConfig()
        cfg.memory_ttl = 2
        h = MemoryIDHead(cfg).eval()
        mem = h.create_memory(torch.device("cpu"))
        feats, boxes = _det(h)

        first = int(h.forward_inference(feats, boxes, mem)["track_ids"][0])
        assert first in mem.tracks

        empty_f = torch.zeros(0, cfg.feature_dim)
        empty_b = torch.zeros(0, 4)
        for _ in range(cfg.memory_ttl + 1):
            h.forward_inference(empty_f, empty_b, mem)
        assert first not in mem.tracks, "TTL 切れの track が memory に残っている"

    def test_reset_clears_memory_and_external_links(self, head):
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head)
        head.commit_memory(mem, [7], feats, boxes)
        assert mem.tracks and mem.external_id_map
        mem.reset()
        assert not mem.tracks and not mem.external_id_map


class TestEmbedAndCommitSeparation:
    """P1-7: embedding 生成と ID 決定を分ける経路。"""

    def test_embed_does_not_touch_memory(self, head):
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head, n=3)
        emb = head.embed(feats, boxes)
        assert emb.shape == (3, head.cfg.embedding_dim)
        assert torch.allclose(emb.norm(dim=-1), torch.ones(3), atol=1e-5)
        assert mem.tracks == {}, "embed() が memory を書き換えている"

    def test_embed_is_empty_safe(self, head):
        emb = head.embed(torch.zeros(0, head.cfg.feature_dim), torch.zeros(0, 4))
        assert emb.shape == (0, head.cfg.embedding_dim)

    def test_commit_memory_uses_external_track_ids(self, head):
        """外部 (tracker) の ID で呼べば、同じ ID は同じ memory slot を使い続ける。"""
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head, n=2)

        head.commit_memory(mem, [11, 22], feats, boxes)
        assert set(mem.external_id_map) == {11, 22}
        slots = dict(mem.external_id_map)

        head.commit_memory(mem, [11, 22], feats, boxes)
        assert mem.external_id_map == slots, "同じ tracker ID が別 slot に移った"
        for mid in slots.values():
            assert mem.tracks[mid]["frame_count"] == 2

    def test_commit_memory_carries_hidden_for_external_ids(self, head):
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head)

        hiddens = []
        for _ in range(3):
            head.commit_memory(mem, [5], feats, boxes)
            mid = mem.external_id_map[5]
            hiddens.append(mem.tracks[mid]["gru_hidden"].clone())

        for i in range(1, len(hiddens)):
            assert not torch.allclose(hiddens[i], hiddens[i - 1]), (
                "commit_memory が前フレームの hidden を使っていない"
            )

    def test_commit_memory_ignores_unassigned_detections(self, head):
        mem = head.create_memory(torch.device("cpu"))
        feats, boxes = _det(head, n=2)
        head.commit_memory(mem, [-1, 4], feats, boxes)
        assert set(mem.external_id_map) == {4}
        assert len(mem.tracks) == 1

    def test_commit_memory_empty_frame_is_safe(self, head):
        mem = head.create_memory(torch.device("cpu"))
        head.commit_memory(
            mem, [], torch.zeros(0, head.cfg.feature_dim), torch.zeros(0, 4)
        )
        assert mem.tracks == {}


class TestStreamingUsesTrackerAsAuthority:
    """streaming 推論の final track ID は OnlineTracker のものであること。"""

    def test_streaming_calls_embed_not_forward_inference(self):
        import inspect

        from htrtdetr.inference import streaming

        src = inspect.getsource(streaming.StreamingInferencer.process_frame)
        assert "id_head.embed(" in src
        assert "id_head.forward_inference(" not in src, (
            "streaming が ID head 側でも ID を採番している (二重 authority)"
        )
        assert "id_head.commit_memory(" in src
