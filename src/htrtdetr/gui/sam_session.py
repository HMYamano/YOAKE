"""
sam_session.py — クリックセグメンテーション用サイドカーのライフサイクル

annotator.py から切り出した、SAM / SAM2 を別プロセス (tools/sam_sidecar.py)
で動かすための責務。起動・能力の問い合わせ・リクエスト送信・返信の取り込み・
停止までを 1 か所にまとめる。

=== なぜ別プロセスなのか ===

GUI の Python は torch-free でよい (pip install -e ".[gui]" は PyTorch を
入れない)。SAM は torch を要るので、**torch のある別の Python** を探して
そこで動かし、GUI とは標準入出力の JSON でやり取りする。GUI プロセスに torch を
持ち込まないための境界がこのモジュールで、だから annotator 本体から切り離せる。

=== スレッドの約束 ===

* 起動 (インタプリタ探索 + import torch) は数秒かかるのでワーカースレッドで行う。
* dearpygui はスレッドセーフではないので、**UI に触るのは必ずメインスレッド**。
  ワーカーは結果を持ち帰るだけで、反映は _finish_sam_start / _drain_sam
  がフレームコールバック上で行う。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

try:
    from . import click_segment
    from . import sidecar as sidecar_mod
except ImportError:  # pragma: no cover
    from htrtdetr.gui import click_segment
    from htrtdetr.gui import sidecar as sidecar_mod

#: フレーム移動から投機 encode を投げるまでの猶予 (秒)。シークバーを
#: ドラッグしている間ずっと投げ続けないための遅延 (docs/CLICK_SEGMENT.md §6.4)。
SAM_EMBED_DELAY = 0.2


def repo_root() -> Path:
    """リポジトリのルート。子プロセスの cwd と PYTHONPATH の基準になる。

    src/htrtdetr/gui/sam_session.py から 3 つ上 (gui → htrtdetr → src → root)。
    インストール済みでも src-layout のチェックアウトでも同じ答えになる。
    """
    return Path(__file__).resolve().parents[3]


class SamSessionMixin:
    """SamSessionMixin — annotator.py から切り出したミックスイン。

    ``AnnotationApp`` の属性 (``self._zoom`` など) をそのまま使う。
    メソッドは移動しただけで中身は変えていない。
    """

    def _sam_ready(self) -> bool:
        return self._sam_client is not None and self._sam_client.is_running()

    def set_click_backend(self, name: str) -> None:
        """フィットに使うバックエンドを切り替える。

        ``local`` へは常に戻れます。これは UI の約束でもあって、SAM を選んで
        失敗しても行き止まりにならないことを保証しています。
        """
        if name not in ("auto", *click_segment.BACKENDS):
            name = "local"
        if name == "sam" and not self._sam_ready():
            # The start is asynchronous now, so whether it worked is not known
            # here. Selecting the backend optimistically keeps the window
            # responsive; the fallback to Local happens when the answer comes.
            self._start_sam(on_done=self._sam_backend_started)
        if name == "model" and not self._predictions:
            self._status(
                "No model predictions are loaded. Use File > Load Predictions "
                "for Click to Box... first, or keep the Local (OpenCV) fit."
            )
            name = "local"
        self._click_backend = name
        if name not in ("sam", "auto"):
            self._stop_sam()
        self._refresh_backend_ui()
        self._status(
            f"Click animal now uses {click_segment.BACKEND_LABELS.get(name, 'Automatic (best available)')}."
        )

    def _sam_backend_started(self, ok: bool) -> None:
        """Keep the promise that choosing SAM is never a dead end.

        ``local`` needs nothing installed, so a helper that could not start
        drops the selection back to it rather than leaving Click animal
        pointing at a backend that is not there.
        """
        if not ok and self._click_backend == "sam":
            self.set_click_backend("local")

    def _start_sam(self, on_done: Optional[Callable[[bool], None]] = None) -> bool:
        """SAM ヘルパーを遅延起動する。使うと言われて初めて探索します。

        probe も起動も**この瞬間まで一切走りません**。torch のあるインタプリタを
        探すだけで数秒かかることがあり、SAM を使わない人にその代金を払わせる
        理由がないからです。

        探索と起動は**ワーカースレッドで**走ります。両方が UI スレッドに乗って
        いた頃、``_status`` に書いた進捗は 1 文字も画面に出ませんでした。
        dearpygui はこのコールバックが返るまでフレームを描き直さないので、
        最後に描かれた "Looking for a Python with PyTorch..." のまま窓が固まり、
        Windows が「応答なし」を出します。実測でも探索 1.2 秒 + 起動 2.7 秒、
        コールドな初回は子プロセスの ``import torch`` だけで数十秒かかります。

        戻り値は**いま使えるか**であって、起動が成功するかではありません。
        まだなら ``on_done(ok)`` が UI スレッドで後から呼ばれるので、押された
        操作はそこでやり直します — 二度押しを覚えてもらう代わりに。
        """
        if self._sam_ready():
            if on_done is not None:
                on_done(True)
            return True
        if on_done is not None:
            self._sam_on_ready.append(on_done)
        if self._sam_starting:
            return False
        self._sam_starting = True
        self._sam_start_token += 1
        token = self._sam_start_token
        self._status("Looking for a Python with PyTorch...")
        threading.Thread(
            target=self._sam_start_worker, args=(token,), daemon=True
        ).start()
        return False

    def _sam_start_worker(self, token: int) -> None:
        """Probe and launch, off the UI thread. Touches no dearpygui item."""
        cap = self._sam_cap
        if cap is None:
            try:
                cap = sidecar_mod.probe_interpreters()
            except Exception as exc:  # noqa: BLE001
                cap = sidecar_mod.SidecarCapability(
                    usable=False, reason=f"Could not probe for PyTorch: {exc}"
                )
        if not cap.usable:
            self._post(lambda: self._finish_sam_start(token, cap, None, ""))
            return
        spec = sidecar_mod.SidecarSpec(
            python=cap.interpreter.python,
            repo_root=str(repo_root()),
            backend=cap.backends[0],
            # The helper picks the device, because only the helper has torch.
            device="auto",
        )
        self._post(
            lambda: self._status(f"Starting the segmentation helper ({spec.backend})...")
        )
        client: Any = None
        error = ""
        try:
            client = sidecar_mod.SidecarClient(spec, on_log=self._sam_log)
            client.start(timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            client, error = None, f"The segmentation helper did not start: {exc}"
        self._post(lambda: self._finish_sam_start(token, cap, client, error))

    def _finish_sam_start(self, token: int, cap, client, error: str) -> None:
        """Install the helper on the UI thread, then replay what was waiting."""
        self._sam_cap = cap
        if token != self._sam_start_token:
            # _stop_sam ran while this start was in flight. The user's last
            # word wins, and a helper nobody asked for any more is a stray
            # process holding the GPU.
            if client is not None:
                try:
                    client.stop()
                except Exception:  # noqa: BLE001
                    pass
            return
        self._sam_starting = False
        waiting, self._sam_on_ready = self._sam_on_ready, []
        if client is None:
            self._sam_client = None
            reason = error or cap.reason
            self._status(
                (reason.splitlines() or ["The segmentation helper did not start."])[0]
            )
        else:
            self._sam_client = client
            self._arm_sam_pump()
            if self._current_bgr is not None:
                self._sam_embed_due = time.monotonic()
            hello = getattr(client, "hello", None) or {}
            backend = str(hello.get("backend") or "")
            device = str(hello.get("device") or "")
            detail = ", ".join(part for part in (backend, device) if part)
            self._status(
                f"Segmentation helper ready ({detail})."
                if detail
                else "Segmentation helper ready."
            )
        ok = client is not None
        for callback in waiting:
            try:
                callback(ok)
            except Exception as exc:  # noqa: BLE001
                self._status(f"Could not resume after starting: {type(exc).__name__}: {exc}")

    def _stop_sam(self) -> None:
        client, self._sam_client = self._sam_client, None
        # Invalidate any start still in flight: without this a helper the user
        # has just turned off installs itself seconds later, and the operation
        # waiting on it runs after they said no.
        self._sam_start_token += 1
        self._sam_starting = False
        self._sam_on_ready = []
        self._sam_jobs.clear()
        self._sam_waiting = False
        self._sam_embed_due = None
        if client is not None:
            try:
                client.stop()
            except Exception:  # noqa: BLE001
                pass

    def _sam_log(self, text: str) -> None:
        # ヘルパーの stderr。ログ枠がまだ無いので黙って捨てるのではなく、
        # 最後の 1 行だけ status に出す (無言で壊れているのが最悪なので)。
        if "Traceback" in text or "Error" in text:
            self._post(lambda message=text.strip()[:160]: self._status(message))

    def _arm_sam_pump(self) -> None:
        """毎フレーム 1 回だけ自分を呼び直すコールバックを仕掛ける。

        ワーカースレッドは dpg に触れず queue に積むだけで、取り出しは必ず
        メインスレッドのここ。GUI_DESIGN P9 が求める形そのものです。
        """
        if self._sam_pump_armed:
            return
        self._sam_pump_armed = True
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self._pump_sidecar)
        except Exception:  # noqa: BLE001 — dpg 未起動 (テスト) では何もしない
            self._sam_pump_armed = False

    def _pump_sidecar(self) -> None:
        self._sam_pump_armed = False
        client = self._sam_client
        if client is None:
            return
        try:
            self._drain_sam(client)
            self._maybe_embed(client)
            self._tick_sam_status(client)
        finally:
            if self._sam_client is not None:
                self._arm_sam_pump()

    def _drain_sam(self, client) -> None:
        for msg in client.poll():
            job = self._sam_jobs.pop(int(msg.get("id", -1)), None)
            if job is None or job["kind"] == "embed":
                continue
            self._sam_waiting = False
            if job["kind"] == "concept":
                if job["frame"] != self.current_frame_idx:
                    self._status("Frame changed; the search was discarded.")
                elif not msg.get("ok"):
                    self._status(str(msg.get("detail") or "The search failed."))
                else:
                    self._apply_concept_reply(msg, job)
                continue
            # 世代チェック: 依頼したときのフレームから動いていたら捨てる。
            # 黙って捨てるが、ユーザが待たされていたことは 1 行だけ知らせる。
            if job["frame"] != self.current_frame_idx:
                self._status("Frame changed; the segmentation was discarded.")
                continue
            self._apply_click_result(
                sidecar_mod.result_from_reply(msg, f"sam:{client.backend_name}")
            )

    def _maybe_embed(self, client) -> None:
        """フレーム移動から少し置いて、投機的に encode を始める。

        CPU では encode が秒オーダーで、click ごとの decode は数十 ms。
        ユーザが個体を探している 1-3 秒のあいだに終わらせてしまえば、
        クリックは即答に見えます。シーク中に投げ続けないための遅延です。
        """
        if self._sam_embed_due is None or self._current_bgr is None:
            return
        if time.monotonic() < self._sam_embed_due:
            return
        self._sam_embed_due = None
        try:
            req_id = client.embed(self._sam_frame_id(self.current_frame_idx), self._current_bgr)
        except Exception:  # noqa: BLE001
            return
        self._sam_jobs[req_id] = {"kind": "embed", "frame": self.current_frame_idx}

    def _tick_sam_status(self, client) -> None:
        if not self._sam_waiting:
            return
        age = client.oldest_pending_age()
        if age is not None and age > 0.4:
            # 進捗は取れないので経過時間だけ出す (GUI_DESIGN P20)。
            self._status(f"Segmenting... {age:.1f} s")

    def _request_sam_point(self, req) -> None:
        client = self._sam_client
        frame_index = self.current_frame_idx
        try:
            path = client.frames.put(self._sam_frame_id(frame_index), self._current_bgr)
            req_id = client.send(
                "point",
                frame=self._sam_frame_id(frame_index),
                path=str(path),
                points=[[float(req.x), float(req.y), 1 if req.polarity else 0]],
            )
        except Exception as exc:  # noqa: BLE001
            self._status(f"The segmentation helper stopped: {exc}")
            self._stop_sam()
            self._click_backend = "local"
            self._refresh_backend_ui()
            return
        self._sam_jobs[req_id] = {"kind": "point", "frame": frame_index}
        self._sam_waiting = True
        self._arm_sam_pump()

    def _on_frame_changed_for_sam(self) -> None:
        """フレームが変わったときに呼ぶ。投機 encode の予約だけ置く。"""
        if self._click_backend != "sam" or not self._sam_ready():
            return
        self._sam_embed_due = time.monotonic() + SAM_EMBED_DELAY
        self._arm_sam_pump()
