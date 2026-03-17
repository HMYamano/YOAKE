"""
visualize_dataset.py — YOAKE アノテーション データビジュアライザ

画像 + bbox + Track ID をGUIで確認する。

使い方:
  # ファイルダイアログで選択
  python tools/visualize_dataset.py

  # 直接パス指定
  python tools/visualize_dataset.py anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/train/annotations.json

キーボード:
  ← / → : フレーム移動
  Space  : 再生 / 一時停止
  Home   : 最初のフレームへ
  End    : 最後のフレームへ

依存:
  pip install Pillow
"""

from __future__ import annotations

import json
import sys
import colorsys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ---------------------------------------------------------------------------
# カラーユーティリティ
# ---------------------------------------------------------------------------

def _id_to_rgb(track_id: int):
    """track_id → 視認性の高い RGB (黄金比ハッシュ)"""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ---------------------------------------------------------------------------
# メインアプリケーション
# ---------------------------------------------------------------------------

class DatasetVisualizer:

    PLAY_INTERVAL_MS = 33   # 約30fps

    def __init__(self, root: tk.Tk, anno_path: Optional[str] = None):
        self.root = root
        self.root.title("YOAKE Dataset Visualizer")

        # 状態
        self.annotation: Optional[dict] = None
        self.cur_vid_idx = 0
        self.cur_frm_idx = 0
        self._photo = None       # GC防止
        self._playing = False
        self._play_job = None
        self._slider_dragging = False

        self._build_ui()
        self._bind_keys()

        if anno_path and Path(anno_path).exists():
            self._load_annotation(anno_path)

    # ------------------------------------------------------------------
    # UI構築
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.configure(bg="#2b2b2b")

        # メニューバー
        menubar = tk.Menu(self.root)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label="アノテーション JSON を開く...", command=self._open_file)
        fm.add_separator()
        fm.add_command(label="終了", command=self.root.quit)
        menubar.add_cascade(label="ファイル", menu=fm)
        self.root.config(menu=menubar)

        # ---- 全体レイアウト ----
        # 上段: [左ペイン|キャンバス|右ペイン]
        body = tk.Frame(self.root, bg="#2b2b2b")
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 左ペイン: シーケンス一覧
        left = tk.Frame(body, bg="#2b2b2b", width=185)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_left_pane(left)

        # 中央: キャンバス
        center = tk.Frame(body, bg="#2b2b2b")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_center(center)

        # 右ペイン: 物体一覧
        right = tk.Frame(body, bg="#2b2b2b", width=215)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)
        self._build_right_pane(right)

    def _lbl(self, parent, text, **kw):
        """ダークテーマ用ラベル"""
        return tk.Label(parent, text=text, bg="#2b2b2b", fg="#cccccc", **kw)

    def _build_left_pane(self, parent):
        self._lbl(parent, "シーケンス", font=("", 10, "bold")).pack(anchor="w", pady=(0, 2))

        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.BOTH, expand=True)

        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.seq_lb = tk.Listbox(
            frame, yscrollcommand=sb.set, selectmode=tk.SINGLE,
            bg="#1e1e1e", fg="#cccccc", selectbackground="#3a7bd5",
            activestyle="none", bd=0, highlightthickness=0,
        )
        self.seq_lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.seq_lb.yview)
        self.seq_lb.bind("<<ListboxSelect>>", self._on_seq_select)

        self.seq_info_var = tk.StringVar(value="")
        tk.Label(
            parent, textvariable=self.seq_info_var,
            bg="#2b2b2b", fg="#888888", justify=tk.LEFT, wraplength=175,
        ).pack(anchor="w", pady=(6, 0))

    def _build_center(self, parent):
        # キャンバス
        self.canvas = tk.Canvas(parent, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render())

        # コントロールバー
        ctrl = tk.Frame(parent, bg="#2b2b2b")
        ctrl.pack(fill=tk.X, pady=(4, 0))

        # 再生ボタン
        self.play_btn = tk.Button(
            ctrl, text="▶ 再生", width=7, command=self._toggle_play,
            bg="#3a7bd5", fg="white", relief=tk.FLAT, activebackground="#2a5bc5",
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(ctrl, text="|◀", width=3, command=self._goto_first,
                  bg="#3c3c3c", fg="#cccccc", relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(ctrl, text="◀", width=3, command=self._prev_frame,
                  bg="#3c3c3c", fg="#cccccc", relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(ctrl, text="▶", width=3, command=self._next_frame,
                  bg="#3c3c3c", fg="#cccccc", relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(ctrl, text="▶|", width=3, command=self._goto_last,
                  bg="#3c3c3c", fg="#cccccc", relief=tk.FLAT).pack(side=tk.LEFT)

        # フレームスライダー
        self.slider = ttk.Scale(ctrl, orient=tk.HORIZONTAL, from_=0, to=0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.slider.bind("<ButtonPress-1>",   lambda e: self._set_dragging(True))
        self.slider.bind("<ButtonRelease-1>", lambda e: self._on_slider_release())
        self.slider.bind("<Motion>",          self._on_slider_motion)

        # フレーム番号 & ジャンプ
        self.frame_var = tk.StringVar(value="0 / 0")
        tk.Label(ctrl, textvariable=self.frame_var,
                 bg="#2b2b2b", fg="#cccccc", width=12).pack(side=tk.LEFT)

        self.jump_var = tk.StringVar()
        jump = tk.Entry(ctrl, textvariable=self.jump_var, width=6,
                        bg="#1e1e1e", fg="#cccccc", insertbackground="white")
        jump.pack(side=tk.LEFT, padx=(0, 2))
        jump.bind("<Return>", self._on_jump)
        self._lbl(ctrl, "へ").pack(side=tk.LEFT)

        # 表示オプション
        opts = tk.Frame(parent, bg="#2b2b2b")
        opts.pack(fill=tk.X, pady=(2, 0))

        self.show_bbox_var = tk.BooleanVar(value=True)
        self.show_id_var   = tk.BooleanVar(value=True)
        self.show_fill_var = tk.BooleanVar(value=True)

        def _chk(parent, text, var):
            return tk.Checkbutton(
                parent, text=text, variable=var, command=self._render,
                bg="#2b2b2b", fg="#cccccc", selectcolor="#2b2b2b",
                activebackground="#2b2b2b", activeforeground="#ffffff",
            )

        _chk(opts, "Bbox表示", self.show_bbox_var).pack(side=tk.LEFT, padx=4)
        _chk(opts, "ID表示",   self.show_id_var  ).pack(side=tk.LEFT, padx=4)
        _chk(opts, "塗りつぶし", self.show_fill_var).pack(side=tk.LEFT, padx=4)

        # スピード
        self._lbl(opts, "再生速度:").pack(side=tk.LEFT, padx=(12, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_combo = ttk.Combobox(opts, textvariable=self.speed_var,
                                   values=[0.25, 0.5, 1.0, 2.0, 4.0], width=5, state="readonly")
        speed_combo.set(1.0)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind("<<ComboboxSelected>>", lambda e: None)

        # ステータスバー
        self.status_var = tk.StringVar(value="アノテーション JSON を開いてください")
        tk.Label(parent, textvariable=self.status_var,
                 bg="#1a1a1a", fg="#888888", anchor="w",
                 relief=tk.FLAT, padx=4).pack(fill=tk.X, pady=(4, 0))

    def _build_right_pane(self, parent):
        self._lbl(parent, "物体一覧", font=("", 10, "bold")).pack(anchor="w", pady=(0, 2))

        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.BOTH, expand=True)

        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.obj_lb = tk.Listbox(
            frame, yscrollcommand=sb.set, selectmode=tk.SINGLE,
            bg="#1e1e1e", fg="#cccccc", selectbackground="#3a7bd5",
            activestyle="none", bd=0, highlightthickness=0,
            font=("Courier", 9),
        )
        self.obj_lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.obj_lb.yview)

        self.stats_var = tk.StringVar(value="")
        tk.Label(
            parent, textvariable=self.stats_var,
            bg="#2b2b2b", fg="#888888", justify=tk.LEFT, wraplength=205,
        ).pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------
    # キーバインド
    # ------------------------------------------------------------------

    def _bind_keys(self):
        self.root.bind("<Left>",  lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Home>",  lambda e: self._goto_first())
        self.root.bind("<End>",   lambda e: self._goto_last())

    # ------------------------------------------------------------------
    # ファイル読み込み
    # ------------------------------------------------------------------

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="アノテーション JSON を選択",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home()),
        )
        if path:
            self._load_annotation(path)

    def _load_annotation(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.annotation = json.load(f)
        except Exception as e:
            messagebox.showerror("読み込みエラー", str(e))
            return

        self.root.title(f"YOAKE Dataset Visualizer  —  {Path(path).name}")
        self._populate_seq_list()

        if self.annotation.get("videos"):
            self.cur_vid_idx = 0
            self.cur_frm_idx = 0
            self.seq_lb.selection_clear(0, tk.END)
            self.seq_lb.selection_set(0)
            self.seq_lb.see(0)
            self._on_seq_loaded()

        self.status_var.set(str(path))

    def _populate_seq_list(self):
        self.seq_lb.delete(0, tk.END)
        for v in self.annotation.get("videos", []):
            self.seq_lb.insert(tk.END, v["video_id"])

    # ------------------------------------------------------------------
    # シーケンス選択
    # ------------------------------------------------------------------

    def _on_seq_select(self, event):
        sel = self.seq_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self.cur_vid_idx and self.annotation:
            return
        self._stop_play()
        self.cur_vid_idx = idx
        self.cur_frm_idx = 0
        self._on_seq_loaded()

    def _on_seq_loaded(self):
        video = self._cur_video()
        if not video:
            return

        frames = video.get("frames", [])
        n = len(frames)
        self.slider.configure(from_=0, to=max(0, n - 1))
        self.slider.set(0)

        fps  = video.get("fps", 25.0)
        w    = video.get("width", "?")
        h    = video.get("height", "?")
        self.seq_info_var.set(f"{video['video_id']}\n{w}×{h}  {fps} fps\n{n} フレーム")
        self._render()

    # ------------------------------------------------------------------
    # フレーム移動
    # ------------------------------------------------------------------

    def _cur_video(self):
        if not self.annotation:
            return None
        vids = self.annotation.get("videos", [])
        return vids[self.cur_vid_idx] if 0 <= self.cur_vid_idx < len(vids) else None

    def _n_frames(self):
        v = self._cur_video()
        return len(v["frames"]) if v else 0

    def _goto_first(self):
        self._stop_play()
        self.cur_frm_idx = 0
        self.slider.set(0)
        self._render()

    def _goto_last(self):
        self._stop_play()
        n = self._n_frames()
        if n:
            self.cur_frm_idx = n - 1
            self.slider.set(n - 1)
            self._render()

    def _prev_frame(self):
        if self.cur_frm_idx > 0:
            self.cur_frm_idx -= 1
            self.slider.set(self.cur_frm_idx)
            self._render()

    def _next_frame(self):
        n = self._n_frames()
        if self.cur_frm_idx < n - 1:
            self.cur_frm_idx += 1
            self.slider.set(self.cur_frm_idx)
            self._render()

    def _on_slider_motion(self, event):
        if self._slider_dragging:
            val = int(float(self.slider.get()))
            if val != self.cur_frm_idx:
                self.cur_frm_idx = val
                self._render()

    def _set_dragging(self, v: bool):
        self._slider_dragging = v

    def _on_slider_release(self):
        self._slider_dragging = False
        val = int(float(self.slider.get()))
        self.cur_frm_idx = val
        self._render()

    def _on_jump(self, event):
        try:
            idx = int(self.jump_var.get())
        except ValueError:
            return
        n = self._n_frames()
        if n == 0:
            return
        self.cur_frm_idx = max(0, min(idx, n - 1))
        self.slider.set(self.cur_frm_idx)
        self._render()
        self.jump_var.set("")

    # ------------------------------------------------------------------
    # 再生
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self._playing = True
        self.play_btn.config(text="⏸ 停止", bg="#c0392b")
        self._schedule_next()

    def _stop_play(self):
        self._playing = False
        self.play_btn.config(text="▶ 再生", bg="#3a7bd5")
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _schedule_next(self):
        if not self._playing:
            return
        speed = float(self.speed_var.get())
        interval = max(10, int(self.PLAY_INTERVAL_MS / speed))
        self._play_job = self.root.after(interval, self._play_step)

    def _play_step(self):
        n = self._n_frames()
        if self.cur_frm_idx >= n - 1:
            self._stop_play()
            return
        self.cur_frm_idx += 1
        self.slider.set(self.cur_frm_idx)
        self._render()
        self._schedule_next()

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------

    def _render(self):
        video = self._cur_video()
        if not video:
            return

        frames = video.get("frames", [])
        n = len(frames)
        if n == 0:
            return

        fi = max(0, min(self.cur_frm_idx, n - 1))
        fd = frames[fi]

        self.frame_var.set(f"{fi:04d} / {n - 1:04d}")

        # 画像ロード
        img_path = fd.get("image_path", "")
        img_w    = fd.get("width") or video.get("width", 640)
        img_h    = fd.get("height") or video.get("height", 480)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (img_w, img_h), (28, 28, 28))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "画像が見つかりません:", fill=(200, 80, 80))
            draw.text((10, 30), img_path[:80], fill=(180, 60, 60))

        # アノテーション描画
        objects       = fd.get("objects", [])
        class_names   = self.annotation.get("class_names", ["?"])
        action_names  = self.annotation.get("action_names", [])

        if self.show_bbox_var.get():
            img = self._draw_boxes(img, objects, class_names, action_names)

        # キャンバスに合わせてリサイズ
        cw = self.canvas.winfo_width()  or 960
        ch = self.canvas.winfo_height() or 540
        img_fit = self._fit(img, cw, ch)

        self._photo = ImageTk.PhotoImage(img_fit)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)

        # 右ペイン更新
        self._update_obj_list(objects, class_names, action_names)

        # ステータスバー
        p = Path(img_path)
        self.status_var.set(f"{p.parent.name}/{p.name}  |  物体: {len(objects)}個")

    def _draw_boxes(self, img, objects, class_names, action_names):
        draw  = ImageDraw.Draw(img, "RGBA")
        iw, ih = img.size

        for obj in objects:
            tid  = obj.get("track_id", -1)
            r, g, b = _id_to_rgb(tid if tid >= 0 else 0)

            x1, y1, x2, y2 = obj["bbox"]
            x1 = max(0.0, min(float(x1), iw))
            y1 = max(0.0, min(float(y1), ih))
            x2 = max(0.0, min(float(x2), iw))
            y2 = max(0.0, min(float(y2), ih))

            # 塗りつぶし
            if self.show_fill_var.get():
                draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 35))

            # 枠線
            draw.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=2)

            # ラベル
            if self.show_id_var.get():
                cid = obj.get("class_id", 0)
                cls = class_names[cid] if 0 <= cid < len(class_names) else "?"
                aid = obj.get("action_id", -1)
                act = action_names[aid] if 0 <= aid < len(action_names) else ""

                parts = []
                if tid >= 0:
                    parts.append(f"ID:{tid}")
                parts.append(cls)
                if act and act != "unknown":
                    parts.append(act)
                label = " ".join(parts)

                lw = len(label) * 7 + 4
                lh = 14
                ty = max(0.0, y1 - lh)
                draw.rectangle([x1, ty, x1 + lw, ty + lh], fill=(r, g, b, 210))
                draw.text((x1 + 2, ty + 1), label, fill=(255, 255, 255))

        return img

    def _fit(self, img, cw, ch):
        if cw <= 1 or ch <= 1:
            return img
        iw, ih = img.size
        scale  = min(cw / iw, ch / ih)
        return img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)

    def _update_obj_list(self, objects, class_names, action_names):
        self.obj_lb.delete(0, tk.END)
        self.obj_lb.config(fg="#cccccc")

        for obj in objects:
            tid  = obj.get("track_id", -1)
            cid  = obj.get("class_id", 0)
            cls  = class_names[cid] if 0 <= cid < len(class_names) else "?"
            x1, y1, x2, y2 = obj["bbox"]
            bw   = int(x2 - x1)
            bh   = int(y2 - y1)
            line = f"ID:{tid:>3}  {cls:<8}  {bw:>4}×{bh:<4}"
            self.obj_lb.insert(tk.END, line)

        # 統計
        n   = len(objects)
        ids = sorted(set(
            o.get("track_id", -1) for o in objects if o.get("track_id", -1) >= 0
        ))
        id_str = ", ".join(str(i) for i in ids[:15])
        if len(ids) > 15:
            id_str += " ..."
        self.stats_var.set(
            f"物体数: {n}\n"
            f"ユニークID: {len(ids)} 個\n"
            f"{id_str}"
        )


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def _parse(argv):
    args = {}
    for a in argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            args[k] = v
        else:
            args.setdefault("anno", a)
    return args


def main():
    if not _PIL_OK:
        print("ERROR: Pillow が必要です。")
        print("  pip install Pillow")
        sys.exit(1)

    args = _parse(sys.argv)
    anno_path = args.get("anno", None)

    root = tk.Tk()
    root.geometry("1400x800")
    root.minsize(900, 600)

    DatasetVisualizer(root, anno_path=anno_path)
    root.mainloop()


if __name__ == "__main__":
    main()
