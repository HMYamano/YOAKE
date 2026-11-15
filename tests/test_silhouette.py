"""
tests/test_silhouette.py — シルエット幾何 (analysis/silhouette.py)

合成した「ハエ」(楕円の体、必要なら翅と脚) は真の角度・大きさ・動きが分かっているので、
回帰はあいまいな見た目ではなく数値で現れる。torch は使わない。

設計: docs/SILHOUETTE_GEOMETRY.md
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from htrtdetr.analysis import silhouette as S
from htrtdetr.gui import click_segment as cs

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "src" / "htrtdetr" / "analysis" / "silhouette.py"

_HAS_CV2 = importlib.util.find_spec("cv2") is not None
needs_cv2 = pytest.mark.skipif(not _HAS_CV2, reason="OpenCV is not installed")

FPS = 30.0
BODY = (16, 6)          # 楕円の半径 → 体長 33px、幅 13px
BOX = (34.0, 14.0)      # detector が出しそうな箱 (体より少し大きい)


# --------------------------------------------------------------------------- #
# 合成データ
# --------------------------------------------------------------------------- #

def _draw_fly(img, cx, cy, angle_deg, *, wing=None, leg=False, body=BODY):
    import cv2

    cv2.ellipse(img, (int(round(cx)), int(round(cy))), body, angle_deg, 0, 360,
                (40, 40, 40), -1)
    if wing is not None:
        # 体の中心から wing 度の方向へ開いた翅 (幅があるので opening では消えない)
        a = math.radians(angle_deg + wing)
        centre = (int(round(cx + 10 * math.cos(a))), int(round(cy + 10 * math.sin(a))))
        cv2.ellipse(img, centre, (10, 3), angle_deg + wing, 0, 360, (90, 90, 90), -1)
    if leg:
        cv2.line(img, (int(cx - 8), int(cy)), (int(cx - 8), int(cy + 13)), (40, 40, 40), 2)
    return img


def _frame(flies, H=240, W=320, **kw):
    img = np.full((H, W, 3), 210, np.uint8)
    for fly in flies:
        _draw_fly(img, *fly, **kw)
    return img


def _corners(flies, box=BOX):
    return np.array([cs.obb_corners((cx, cy, box[0], box[1], math.radians(a)))
                     for cx, cy, a in flies])


def _boxes(corners):
    c = np.asarray(corners)
    return np.stack([c[..., 0].min(1), c[..., 1].min(1), c[..., 0].max(1), c[..., 1].max(1)], 1)


def _update(analyzer, flies, frame_index, track_ids=None, **kw):
    corners = _corners(flies)
    rows = analyzer.update(_frame(flies, **kw), frame_index, _boxes(corners),
                           corners=corners,
                           track_ids=track_ids if track_ids is not None
                           else list(range(1, len(flies) + 1)))
    return S.rows_as_dicts(rows)


def _gray(img):
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _axis_error_deg(measured_rad, true_deg):
    d = (math.degrees(measured_rad) - true_deg) % 180.0
    return min(d, 180.0 - d)


def _angle_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


# --------------------------------------------------------------------------- #
# 依存の契約
# --------------------------------------------------------------------------- #

def test_module_never_imports_torch():
    """幾何は GUI と同じ torch の無い Python でも動かす (AST で確かめる)。"""
    imported = set()
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


def test_module_scope_needs_only_numpy():
    """OpenCV は使う関数の中で読む: cv2 の無い環境でも import は通る。"""
    top = set()
    for node in ast.parse(MODULE_PATH.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "contextlib", "csv", "math", "time", "dataclasses",
                   "typing", "numpy"}


def test_fields_are_unique_and_the_meta_names_them():
    assert len(set(S.GEOMETRY_FIELDS)) == len(S.GEOMETRY_FIELDS)
    meta = S.geometry_meta()
    assert meta["fields"] == list(S.GEOMETRY_FIELDS)
    assert meta["version"] == S.GEOMETRY_VERSION
    assert meta["params"]["roi_scale"] == S.SilhouetteParams().roi_scale


def test_nonsense_params_are_refused():
    with pytest.raises(ValueError):
        S.SilhouetteParams(roi_scale=1.0)
    with pytest.raises(ValueError):
        S.SilhouetteParams(velocity_smoothing=0.0)


# --------------------------------------------------------------------------- #
# 1 フレームの測定
# --------------------------------------------------------------------------- #

@needs_cv2
@pytest.mark.parametrize("angle", [0.0, 30.0, 75.0, 120.0, 165.0])
def test_measure_recovers_the_axis_and_the_body_size(angle):
    fly = (160.0, 120.0, angle)
    sil, why = S.measure(_gray(_frame([fly])), _corners([fly])[0])
    assert why == "" and sil is not None
    assert _axis_error_deg(math.atan2(sil.axis[1], sil.axis[0]), angle) < 2.0
    assert sil.length == pytest.approx(2 * BODY[0] + 1, abs=2.0)
    assert sil.width == pytest.approx(2 * BODY[1] + 1, abs=2.0)
    assert sil.solidity > 0.9


@needs_cv2
def test_the_axis_comes_from_the_animal_not_from_the_box():
    """AABB detector の箱 (θ=0) からでも、斜めの個体の本当の体軸が出る。

    既定の detector は軸平行箱なので、行動ブランチの orientation 特徴は θ=0 に
    張り付く。この測定はその穴を埋めるためにある。
    """
    fly = (160.0, 120.0, 40.0)
    upright = cs.obb_corners((160.0, 120.0, 40.0, 40.0, 0.0))
    sil, why = S.measure(_gray(_frame([fly])), upright)
    assert why == ""
    assert _axis_error_deg(math.atan2(sil.axis[1], sil.axis[0]), 40.0) < 2.0


@needs_cv2
def test_an_extended_wing_shows_as_lateral_asymmetry_not_as_a_round_body():
    folded = S.rows_as_dicts(S.SilhouetteAnalyzer().update(
        _frame([(160, 120, 30)]), 0, _boxes(_corners([(160, 120, 30)])),
        corners=_corners([(160, 120, 30)]), track_ids=[1]))[0]
    extended = S.rows_as_dicts(S.SilhouetteAnalyzer().update(
        _frame([(160, 120, 30)], wing=90), 0, _boxes(_corners([(160, 120, 30)])),
        corners=_corners([(160, 120, 30)]), track_ids=[1]))[0]
    assert folded["lateral_asym"] < 0.05
    assert extended["lateral_asym"] > folded["lateral_asym"] + 0.2
    assert extended["solidity"] < folded["solidity"] - 0.1
    # 体の芯で測る細長さは、翅を開いても頭尾の判定を止めるほど丸くならない
    assert extended["elongation"] >= S.SilhouetteParams().min_elongation
    assert _axis_error_deg(extended["axis"], 30.0) < 3.0


@needs_cv2
def test_a_neighbour_inside_the_roi_is_cut_out_by_its_own_box():
    flies = [(150.0, 120.0, 0.0), (182.0, 120.0, 0.0)]
    rows = _update(S.SilhouetteAnalyzer(), flies, 0)
    for row, (cx, cy, _a) in zip(rows, flies):
        assert row["valid"] == 1 and row["twin_core"] == 0
        assert row["cx"] == pytest.approx(cx, abs=2.0)
        assert row["cy"] == pytest.approx(cy, abs=2.0)
        assert row["nn_dist"] == pytest.approx(32.0 / row["length"], abs=0.1)


@needs_cv2
def test_two_animals_under_one_box_are_measured_and_flagged():
    """交尾中のように 2 個体が 1 つの blob になり検出も 1 つなら、芯が 2 つあると言う。

    Click to Box はここで箱を作るのを拒む (学習データを汚すから) が、測定では
    「2 個体がくっついている」こと自体が知りたい値になる。
    """
    import cv2

    img = _frame([(140.0, 120.0, 0.0), (180.0, 120.0, 0.0)])
    cv2.line(img, (156, 120), (164, 120), (40, 40, 40), 2)   # 細い首でつながる
    box = cs.obb_corners((160.0, 120.0, 76.0, 14.0, 0.0))
    sil, why = S.measure(_gray(img), box)
    assert why == "" and sil is not None
    assert sil.twin_core


@needs_cv2
def test_nothing_found_is_reported_not_guessed():
    rows = S.SilhouetteAnalyzer().update(
        np.full((240, 320, 3), 210, np.uint8), 0, np.array([[100, 100, 134, 114]]),
        track_ids=[1])
    row = S.rows_as_dicts(rows)[0]
    assert row["valid"] == 0 and row["diagnosis"] in cs.DIAGNOSES
    assert all(row[k] is None for k in ("cx", "axis", "heading", "length", "motion_energy"))


@needs_cv2
def test_a_large_roi_is_downscaled_and_measured_in_frame_pixels():
    fly = (250.0, 200.0, 25.0)
    corners = [cs.obb_corners((250.0, 200.0, 132.0, 52.0, math.radians(25.0)))]
    img = _frame([fly], H=400, W=500, body=(64, 24))
    sil, why = S.measure(_gray(img), corners[0], params=S.SilhouetteParams(max_roi_side=96))
    assert why == ""
    assert sil.length == pytest.approx(129.0, abs=6.0)
    assert sil.width == pytest.approx(49.0, abs=4.0)
    assert sil.centre[0] == pytest.approx(250.0, abs=2.0)
    assert _axis_error_deg(math.atan2(sil.axis[1], sil.axis[0]), 25.0) < 2.0


# --------------------------------------------------------------------------- #
# 頭尾 (2π の向き)
# --------------------------------------------------------------------------- #

def _walk(analyzer, start_frame, x0, steps, dx, angle=0.0, y=120.0):
    """体軸を angle に保ったまま、x 方向に dx px/frame で歩かせる。最後の行と位置を返す。"""
    x = x0
    row = None
    for k in range(steps):
        x += dx
        row = _update(analyzer, [(x, y, angle)], start_frame + k)[0]
    return row, x


@needs_cv2
def test_heading_stays_null_until_the_animal_moves():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    for f in range(8):
        row = _update(analyzer, [(160.0, 120.0, 30.0)], f)[0]
        assert row["heading"] is None and row["heading_conf"] == 0.0
        assert row["valid"] == 1


@needs_cv2
def test_heading_follows_motion_then_continuity():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    row, x = _walk(analyzer, 0, 60.0, 12, 2.0)
    assert row["heading"] is not None
    assert _angle_error(row["heading"], 0.0) < math.radians(3)
    walking_conf = row["heading_conf"]
    assert walking_conf > 0.5

    # 立ち止まっても向きは保たれ、確信度はゆっくりしか下がらない
    row, _ = _walk(analyzer, 12, x, 10, 0.0)
    assert _angle_error(row["heading"], 0.0) < math.radians(3)
    assert 0.0 < row["heading_conf"] < walking_conf

    # その場で向きを変えると、連続性で同じ側 (頭) を追い続ける
    for k, angle in enumerate(range(10, 70, 10)):
        row = _update(analyzer, [(x, 120.0, float(angle))], 22 + k)[0]
    assert _angle_error(row["heading"], math.radians(60)) < math.radians(4)


@needs_cv2
def test_backing_up_briefly_does_not_flip_but_a_sustained_reversal_does():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    _row, x = _walk(analyzer, 0, 60.0, 12, 2.0)
    row, x = _walk(analyzer, 12, x, 5, -2.0)        # 短い後退
    assert _angle_error(row["heading"], 0.0) < math.radians(3)
    row, x = _walk(analyzer, 17, x, 12, -2.0)       # 逆向きの移動が続く
    assert _angle_error(row["heading"], math.pi) < math.radians(3)


@needs_cv2
def test_untracked_detections_get_an_axis_but_no_heading():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    for f in range(6):
        row = _update(analyzer, [(60.0 + 2 * f, 120.0, 0.0)], f, track_ids=[-1])[0]
    assert row["valid"] == 1 and row["axis"] is not None
    assert row["heading"] is None and row["motion_energy"] is None


@needs_cv2
def test_facing_uses_the_resolved_heading():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    xa = 60.0
    for f in range(12):
        xa += 2.0
        rows = _update(analyzer, [(xa, 120.0, 0.0), (250.0, 120.0, 90.0)], f)
    walker, sitter = rows
    assert walker["facing_nn"] < 0.2            # 歩いている方は相手を向いている
    assert sitter["faced_by_nn"] < 0.2          # 座っている方は向かれている
    assert sitter["heading"] is None and sitter["facing_nn"] is None


# --------------------------------------------------------------------------- #
# 残差運動
# --------------------------------------------------------------------------- #

@needs_cv2
def test_motion_energy_separates_limb_motion_from_whole_body_motion():
    still = S.SilhouetteAnalyzer(fps=FPS)
    for f in range(5):
        row = _update(still, [(160.0, 120.0, 0.0)], f)[0]
    assert row["motion_energy"] == pytest.approx(0.0, abs=0.01)

    walking = S.SilhouetteAnalyzer(fps=FPS)
    energies = []
    for f in range(8):
        row = _update(walking, [(60.0 + 1.5 * f, 120.0, 5.0 * f)], f)[0]
        if row["motion_energy"] is not None:
            energies.append(row["motion_energy"])
    assert max(energies) < 0.1                  # 並進と回転は打ち消される

    grooming = S.SilhouetteAnalyzer(fps=FPS)
    flicker = []
    for f in range(6):
        row = _update(grooming, [(160.0, 120.0, 0.0)], f, leg=(f % 2 == 1))[0]
        if row["motion_energy"] is not None:
            flicker.append(row["motion_energy"])
    assert min(flicker) > 0.2                   # 脚だけが動けば残る


@needs_cv2
def test_motion_energy_needs_the_previous_frame():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    _update(analyzer, [(160.0, 120.0, 0.0)], 0)
    row = _update(analyzer, [(160.0, 120.0, 0.0)], 2)[0]    # 欠番の後
    assert row["motion_energy"] is None
    row = _update(analyzer, [(160.0, 120.0, 0.0)], 3)[0]
    assert row["motion_energy"] is not None


# --------------------------------------------------------------------------- #
# 入出力
# --------------------------------------------------------------------------- #

@needs_cv2
def test_rows_follow_detection_order_and_the_csv_has_one_line_each(tmp_path):
    flies = [(80.0, 60.0, 0.0), (200.0, 160.0, 45.0)]
    corners = _corners(flies)
    rows = S.SilhouetteAnalyzer().update(_frame(flies), 7, _boxes(corners),
                                         corners=corners, track_ids=[5, 9])
    assert [round(r[S.GEOMETRY_FIELDS.index("cx")]) for r in rows] == [80, 200]

    table = S.GeometryCSV(str(tmp_path / "geometry.csv"))
    table.write(7, [5, 9], rows)
    table.close()
    with open(tmp_path / "geometry.csv", newline="", encoding="utf-8") as handle:
        lines = list(csv.reader(handle))
    assert lines[0] == ["frame", "detection", "track_id", *S.GEOMETRY_FIELDS]
    assert [line[:3] for line in lines[1:]] == [["7", "0", "5"], ["7", "1", "9"]]


@needs_cv2
def test_draw_geometry_marks_the_heading():
    analyzer = S.SilhouetteAnalyzer(fps=FPS)
    rows = None
    for f in range(10):
        corners = _corners([(60.0 + 2 * f, 120.0, 0.0)])
        rows = analyzer.update(_frame([(60.0 + 2 * f, 120.0, 0.0)]), f, _boxes(corners),
                               corners=corners, track_ids=[1])
    canvas = np.zeros((240, 320, 3), np.uint8)
    S.draw_geometry(canvas, rows)
    assert canvas.any()


def test_mismatched_inputs_are_refused():
    analyzer = S.SilhouetteAnalyzer()
    with pytest.raises(ValueError):
        analyzer.update(np.zeros((10, 10, 3), np.uint8), 0, np.zeros((2, 4)), track_ids=[1])
    with pytest.raises(ValueError):
        analyzer.update(np.zeros((10, 10, 3), np.uint8), 0, np.zeros((2, 4)),
                        corners=np.zeros((1, 4, 2)))
