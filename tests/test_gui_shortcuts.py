"""
htrtdetr.gui.shortcuts のユニットテスト.

Pure pytest: no display, no viewport, and dearpygui is never imported at module
scope. The two tests that need real dearpygui key constants import it lazily via
``pytest.importorskip`` so the suite still runs on a machine without it.

What these lock down
--------------------
* The keymap is internally consistent (no over-bound combination, no action
  without a key, no help row missing).
* Every key suffix in the table actually resolves against the installed
  dearpygui. This is the test that catches ``mvKey_Backspace`` /
  ``mvKey_LBracket`` / ``mvKey_Space``, none of which exist in 2.3.1 — writing
  one is an ``AttributeError`` at handler registration, i.e. the app does not
  start.
* The LabelImg-familiar four (``W`` create, ``Esc`` cancel, ``Delete`` delete,
  ``Ctrl+S`` save) and the legacy bindings users already have in muscle memory.
* Every user-visible description is English ASCII, so it renders before a font
  is loaded and never smuggles CJK into the GUI.
"""

from __future__ import annotations

import unicodedata

import pytest

from htrtdetr.gui import shortcuts


# ─── structural integrity ─────────────────────────────────────────────────
def _imported_modules() -> set:
    """Every module name shortcuts.py imports, anywhere in the file.

    Parsed with :mod:`ast` rather than matched as text, so the usage examples
    in the module docstring are not mistaken for real imports.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(shortcuts))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_module_imports_neither_dearpygui_nor_torch():
    """shortcuts must stay a pure data module: no dpg, no torch, anywhere.

    Both are checked because ``htrtdetr.gui.*`` has to stay import-safe on a
    machine with no display and no torch. The import list is asserted exactly,
    so a transitive dependency cannot sneak in later either.
    """
    imported = _imported_modules()
    for banned in ("dearpygui", "dearpygui.dearpygui", "torch", "cv2", "numpy"):
        assert not any(name == banned or name.startswith(banned + ".")
                       for name in imported), \
            f"shortcuts.py imports {banned}"
    assert imported <= {"sys", "dataclasses", "typing", "__future__"}, \
        f"shortcuts.py grew an unexpected dependency: {sorted(imported)}"


def test_no_duplicate_bindings():
    """No (scope, key, mods) may appear twice."""
    assert shortcuts.collisions() == []
    assert shortcuts.conflicts() == []

    seen = set()
    for b in shortcuts.BINDINGS:
        ident = (b.scope,) + b.combo
        assert ident not in seen, f"{b.label} is bound twice in scope {b.scope}"
        seen.add(ident)


def test_conflict_detection_actually_detects():
    """A deliberate duplicate must be reported, or the check is worthless."""
    dup = shortcuts.Binding(action="test.fake", key="W", mods=(), scope="annotator",
                            group="Tools", help="A deliberate duplicate")
    original = shortcuts.BINDINGS
    try:
        shortcuts.BINDINGS = original + (dup,)
        report = shortcuts.collisions()
        assert report, "a duplicate W binding was not detected"
        assert "annotator: W" in report[0]
        assert "tool.draw_box" in report[0] and "test.fake" in report[0]
    finally:
        shortcuts.BINDINGS = original
    assert shortcuts.collisions() == []


def test_scopes_are_known_and_nonempty():
    for scope in shortcuts.SCOPES:
        assert shortcuts.for_scope(scope), f"scope {scope} has no bindings"
    assert {b.scope for b in shortcuts.BINDINGS} == set(shortcuts.SCOPES)


def test_every_binding_has_a_known_group_and_context():
    for b in shortcuts.BINDINGS:
        assert b.group in shortcuts.GROUP_ORDER, f"{b.action}: unknown group {b.group!r}"
        assert b.context in shortcuts.CONTEXTS, f"{b.action}: unknown context {b.context!r}"


# ─── dearpygui key constants ──────────────────────────────────────────────
def test_every_key_constant_exists():
    """Every stored suffix must exist as ``mvKey_<suffix>`` on this dearpygui.

    This is the test that catches ``mvKey_Backspace`` (the constant is
    ``mvKey_Back``), ``mvKey_Space`` (it is ``mvKey_Spacebar``) and
    ``mvKey_LBracket`` / ``mvKey_RBracket`` (they are ``mvKey_Open_Brace`` /
    ``mvKey_Close_Brace``).
    """
    dpg = pytest.importorskip("dearpygui.dearpygui")
    for b in shortcuts.BINDINGS:
        assert hasattr(dpg, "mvKey_" + b.key), (
            f"{b.action}: dearpygui has no mvKey_{b.key}"
        )


def test_resolve_returns_an_int_for_every_binding():
    dpg = pytest.importorskip("dearpygui.dearpygui")
    for b in shortcuts.BINDINGS:
        assert isinstance(shortcuts.resolve(dpg, b), int)


def test_resolve_uses_the_fallback_chain():
    """A dearpygui that only knows the old spelling must still resolve."""

    class OldDpg:
        mvKey_LBracket = 219

    b = shortcuts.binding_for("obb.rotate_ccw")
    assert shortcuts.resolve(OldDpg, b) == 219


def test_resolve_names_what_it_tried_when_it_fails():
    class EmptyDpg:
        pass

    b = shortcuts.binding_for("edit.propagate")
    with pytest.raises(AttributeError) as excinfo:
        shortcuts.resolve(EmptyDpg, b)
    msg = str(excinfo.value)
    assert "mvKey_Spacebar" in msg and "edit.propagate" in msg


def test_modifier_constants_exist():
    """mvKey_Control does not exist in 2.x; mods_held must use mvKey_Mod*."""
    dpg = pytest.importorskip("dearpygui.dearpygui")
    used = {m for b in shortcuts.BINDINGS for m in b.mods}
    for mod in used:
        assert hasattr(dpg, "mvKey_Mod" + mod), f"dearpygui has no mvKey_Mod{mod}"


def test_mods_held_checks_every_modifier():
    class FakeDpg:
        mvKey_ModCtrl = 1
        mvKey_ModShift = 2
        down: set = set()

        @classmethod
        def is_key_down(cls, k):
            return k in cls.down

    save = shortcuts.binding_for("file.save")            # Ctrl+S
    save_as = shortcuts.binding_for("file.save_as")      # Ctrl+Shift+S
    draw = shortcuts.binding_for("tool.draw_box")        # bare W

    FakeDpg.down = set()
    assert shortcuts.mods_held(FakeDpg, draw) is True    # unmodified: always true
    assert shortcuts.mods_held(FakeDpg, save) is False

    FakeDpg.down = {FakeDpg.mvKey_ModCtrl}
    assert shortcuts.mods_held(FakeDpg, save) is True
    assert shortcuts.mods_held(FakeDpg, save_as) is False   # Shift still missing

    FakeDpg.down = {FakeDpg.mvKey_ModCtrl, FakeDpg.mvKey_ModShift}
    assert shortcuts.mods_held(FakeDpg, save_as) is True


# ─── the contracts users feel ─────────────────────────────────────────────
def test_labelimg_contract():
    """W creates, Esc cancels, Delete deletes, Ctrl+S saves."""
    draw = shortcuts.binding_for("tool.draw_box")
    assert draw.key == "W" and draw.mods == ()

    cancel = shortcuts.binding_for("tool.cancel")
    assert cancel.key == "Escape" and cancel.mods == ()

    delete = shortcuts.binding_for("edit.delete")
    assert delete.key == "Delete" and delete.mods == ()

    save = shortcuts.binding_for("file.save")
    assert save.key == "S" and save.mods == ("Ctrl",)


def test_review_navigation_exists():
    """The review loop is the product; without these, candidates are unreviewable."""
    nxt = shortcuts.binding_for("review.next_pending_frame")
    assert nxt.key == "Right" and nxt.mods == ("Ctrl",)

    prv = shortcuts.binding_for("review.prev_pending_frame")
    assert prv.key == "Left" and prv.mods == ("Ctrl",)

    assert shortcuts.binding_for("review.accept_all").key == "Y"
    assert shortcuts.binding_for("review.discard_all").key == "N"
    assert shortcuts.binding_for("review.cycle").key == "Q"
    for action in ("review.accept_all", "review.discard_all", "review.cycle"):
        assert shortcuts.binding_for(action).mods == ()


def test_legacy_bindings_preserved():
    """Guards against re-churning muscle memory that already exists."""
    prop = shortcuts.binding_for("edit.propagate")
    assert prop.mods == ()
    assert prop.key == "Spacebar"        # mvKey_Space does NOT exist
    assert prop.label == "Space"         # ...but the user still reads "Space"

    ccw = shortcuts.binding_for("obb.rotate_ccw")
    cw = shortcuts.binding_for("obb.rotate_cw")
    assert (ccw.key, ccw.mods, ccw.label) == ("Open_Brace", (), "[")
    assert (cw.key, cw.mods, cw.label) == ("Close_Brace", (), "]")


def test_every_tool_has_a_binding():
    for tool in ("select", "draw_box", "draw_rotated", "pan", "delete"):
        b = shortcuts.binding_for(f"tool.{tool}")
        assert b is not None, f"tool.{tool} has no key"
        assert b.mods == (), f"tool.{tool} must be a bare key"


def test_rebound_keys_match_the_changelog():
    """A is prev frame, N discards, R picks the rotated tool, Ctrl+T auto-tracks."""
    assert "nav.prev" in [b.action for b in shortcuts.by_key("A")]
    assert "review.discard_all" in [b.action for b in shortcuts.by_key("N")]
    assert "tool.draw_rotated" in [b.action for b in shortcuts.by_key("R")]
    assert "auto.toggle_autotrack" in [b.action for b in shortcuts.by_key("Ctrl+T")]
    assert shortcuts.by_key("P") == ()          # P is deliberately free now


def test_actions_cover_zero_through_nine():
    for i in range(10):
        b = shortcuts.binding_for(f"action.set_{i}")
        assert b is not None and b.key == str(i) and b.mods == ()


# ─── lookup round-trips ───────────────────────────────────────────────────
def test_lookup_round_trips():
    for b in shortcuts.BINDINGS:
        assert b in shortcuts.for_action(b.action, b.scope)
        assert b in shortcuts.by_key(b.key, b.mods, b.scope)
        assert b in shortcuts.by_key(b.label, scope=b.scope)
        assert b in shortcuts.for_scope(b.scope)
        assert b in shortcuts.for_group(b.group, b.scope)
        assert b in shortcuts.for_context(b.context, b.scope)


def test_by_key_accepts_friendly_spellings():
    assert shortcuts.by_key("Space") == shortcuts.by_key("Spacebar")
    assert shortcuts.by_key("[") == shortcuts.by_key("Open_Brace")
    assert shortcuts.by_key("]") == shortcuts.by_key("Close_Brace")
    assert shortcuts.by_key("Esc") == shortcuts.by_key("Escape")
    assert shortcuts.by_key("Backspace")[0].action == "edit.delete"
    assert shortcuts.by_key("Ctrl+Shift+S")[0].action == "file.save_as"
    assert shortcuts.by_key("+")[0].action == "view.zoom_in"


def test_canonical_key_normalises():
    assert shortcuts.canonical_key("Space") == "Spacebar"
    assert shortcuts.canonical_key("LBracket") == "Open_Brace"
    assert shortcuts.canonical_key("rbracket") == "Close_Brace"
    assert shortcuts.canonical_key("W") == "W"        # unknown passes through


def test_unbound_action_lookups_are_empty_not_exceptions():
    assert shortcuts.binding_for("does.not.exist") is None
    assert shortcuts.for_action("does.not.exist") == ()
    assert shortcuts.label_for("does.not.exist") == ""
    assert shortcuts.help_for("does.not.exist") == ""
    assert shortcuts.hint_for("does.not.exist") == ""
    assert shortcuts.with_key("Run", "does.not.exist") == "Run"


def test_help_shortcuts_is_scoped_per_window():
    """F1 exists in both scopes; binding_for must honour the scope argument."""
    assert shortcuts.binding_for("help.shortcuts", "annotator").group == "File"
    assert shortcuts.binding_for("help.shortcuts", "hub").group == "Hub"
    assert len(shortcuts.for_action("help.shortcuts")) == 2


def test_actions_are_unique_and_complete():
    acts = shortcuts.actions()
    assert len(acts) == len(set(acts))
    assert set(acts) == {b.action for b in shortcuts.BINDINGS}
    assert set(shortcuts.actions("hub")).isdisjoint(
        {"tool.select", "nav.next", "file.save"}
    )


# ─── rendering helpers ────────────────────────────────────────────────────
def test_help_rows_cover_every_binding():
    """Documentation cannot go stale: one row per binding, both scopes."""
    for scope in shortcuts.SCOPES:
        assert len(shortcuts.help_rows(scope)) == len(shortcuts.for_scope(scope))


def test_help_groups_group_correctly():
    for scope in shortcuts.SCOPES:
        groups = shortcuts.help_groups(scope)
        names = [g for g, _ in groups]

        assert names == sorted(names, key=shortcuts.GROUP_ORDER.index), \
            "help_groups must follow GROUP_ORDER"
        assert len(names) == len(set(names)), "a group appears twice"
        assert all(rows for _, rows in groups), "an empty group was emitted"

        flat = [row for _, rows in groups for row in rows]
        assert flat == shortcuts.help_rows(scope), \
            "help_groups and help_rows disagree about content or order"


def test_annotator_groups_are_the_expected_sections():
    names = [g for g, _ in shortcuts.help_groups("annotator")]
    assert names == ["Tools", "View", "Navigation", "Review", "Edit", "File"]
    assert [g for g, _ in shortcuts.help_groups("hub")] == ["Hub"]


def test_context_groups_partition_every_binding():
    groups = shortcuts.context_groups()
    names = [c for c, _ in groups]
    assert names == [c for c in shortcuts.CONTEXTS if c in names]
    total = sum(len(rows) for _, rows in groups)
    assert total == len(shortcuts.BINDINGS)


def test_with_key_formatting():
    assert shortcuts.with_key("Save", "file.save") == "Save (Ctrl+S)"
    assert shortcuts.with_key("Fit", "view.fit") == "Fit (F)"
    assert shortcuts.with_key("Accept all", "review.accept_all") == "Accept all (Y)"
    assert shortcuts.with_key("Track forward", "auto.track_forward") == \
        "Track forward (T)"
    # unbound -> unchanged, so a control never grows an empty "()"
    assert shortcuts.with_key("Whatever", "nope") == "Whatever"


def test_hint_for_lists_every_alternate():
    assert shortcuts.hint_for("nav.next") == "Next frame  (D / Right)"
    assert shortcuts.hint_for("nav.prev") == "Previous frame  (A / Left)"
    assert shortcuts.hint_for("edit.delete") == \
        "Delete the selected box  (Delete / Backspace)"
    # single-binding actions still render cleanly
    assert shortcuts.hint_for("view.fit") == "Fit the frame to the window  (F)"


def test_binding_row_and_aliases():
    b = shortcuts.binding_for("file.save")
    assert b.row() == ("Ctrl+S", b.help)
    assert b.description == b.help
    assert b.category == b.group == "File"
    assert b.keys == ("Ctrl", "S")
    assert shortcuts.Shortcut is shortcuts.Binding


def test_changed_rows_are_three_columns():
    rows = shortcuts.changed_rows()
    assert len(rows) == 4
    for row in rows:
        assert len(row) == 3
        assert all(isinstance(cell, str) and cell for cell in row)
        assert all(cell.isascii() for cell in row)
    assert all(isinstance(s, str) and s.isascii()
               for s in shortcuts.UNCHANGED_ON_PURPOSE)


# ─── key-name formatting ──────────────────────────────────────────────────
def test_format_key_is_readable():
    assert shortcuts.format_key("Open_Brace") == "["
    assert shortcuts.format_key("Close_Brace") == "]"
    assert shortcuts.format_key("Spacebar") == "Space"
    assert shortcuts.format_key("Back") == "Backspace"
    assert shortcuts.format_key("Escape") == "Esc"
    assert shortcuts.format_key("Plus") == "+"
    assert shortcuts.format_key("Minus") == "-"
    assert shortcuts.format_key("F1") == "F1"
    assert shortcuts.format_key("W") == "W"


def test_modifier_names_are_platform_correct():
    win = shortcuts.modifier_names("win32")
    mac = shortcuts.modifier_names("darwin")
    lin = shortcuts.modifier_names("linux")
    assert win["Ctrl"] == "Ctrl" and win["Alt"] == "Alt"
    assert lin == win
    assert mac["Ctrl"] == "Cmd" and mac["Alt"] == "Option"
    assert mac["Shift"] == win["Shift"] == "Shift"


def test_format_binding_orders_modifiers_consistently():
    assert shortcuts.format_binding("S", ("Ctrl",), "win32") == "Ctrl+S"
    assert shortcuts.format_binding("S", ("Ctrl", "Shift"), "win32") == "Ctrl+Shift+S"
    # reversed input must produce the same label
    assert shortcuts.format_binding("S", ("Shift", "Ctrl"), "win32") == "Ctrl+Shift+S"
    assert shortcuts.format_binding("S", ("Ctrl", "Shift"), "darwin") == "Cmd+Shift+S"
    assert shortcuts.format_binding("Open_Brace", (), "win32") == "["


def test_labels_are_auto_generated_and_match_the_formatter():
    for b in shortcuts.BINDINGS:
        assert b.label == shortcuts.format_binding(b.key, b.mods)
        assert b.label, f"{b.action} has an empty label"


def test_explicit_label_is_not_overwritten():
    b = shortcuts.Binding(action="test.custom", key="W", mods=("Ctrl",),
                          label="Custom", help="Something")
    assert b.label == "Custom"


def test_binding_normalises_its_inputs():
    b = shortcuts.Binding(action="test.norm", key="space", mods=("ctrl",),
                          group="Edit", help="Normalised")
    assert b.key == "Spacebar"
    assert b.mods == ("Ctrl",)
    assert b.label == shortcuts.format_binding("Spacebar", ("Ctrl",))
    assert b.context == "Annotator canvas"      # derived from the group


# ─── English copy ─────────────────────────────────────────────────────────
def test_every_binding_has_a_nonempty_english_description():
    for b in shortcuts.BINDINGS:
        assert b.help, f"{b.action} has no description"
        assert b.help.strip() == b.help, f"{b.action}: padded description"
        assert b.help[0].isupper(), f"{b.action}: description is not sentence case"
        assert not b.help.endswith("."), \
            f"{b.action}: a short description takes no final period"


def test_descriptions_contain_no_cjk_and_stay_ascii():
    """GUI-visible strings must be English ASCII.

    ASCII-only is the stronger guarantee: it rules out CJK, and it also rules
    out the em dashes and typographic quotes that render as blanks before a
    font is loaded.
    """
    for b in shortcuts.BINDINGS:
        for field_name in ("action", "key", "label", "help", "group", "context", "scope"):
            value = getattr(b, field_name)
            assert value.isascii(), (
                f"{b.action}.{field_name} contains non-ASCII: {value!r} "
                f"({[unicodedata.name(c, 'UNKNOWN') for c in value if not c.isascii()]})"
            )


def test_no_description_is_duplicated_within_a_group():
    """Two different actions in one section must not read identically."""
    for scope in shortcuts.SCOPES:
        for group, rows in shortcuts.help_groups(scope):
            by_help: dict[str, set] = {}
            for b in shortcuts.for_group(group, scope):
                by_help.setdefault(b.help, set()).add(b.action)
            for text, acts in by_help.items():
                assert len(acts) == 1, \
                    f"{scope}/{group}: {sorted(acts)} share the description {text!r}"


# ─── typing guard contract ────────────────────────────────────────────────
def test_only_modified_bindings_are_always_allowed():
    """A bare letter must never bypass the typing guard - it would type itself."""
    assert shortcuts.ALWAYS_ALLOWED
    for action in shortcuts.ALWAYS_ALLOWED:
        b = shortcuts.binding_for(action)
        assert b is not None, f"ALWAYS_ALLOWED names an unbound action: {action}"
        assert b.mods, f"{action} is unmodified and must not bypass the typing guard"


def test_editing_types_look_like_dpg_item_types():
    assert shortcuts.EDITING_TYPES
    for t in shortcuts.EDITING_TYPES:
        assert t.startswith("mv"), f"{t!r} is not a dearpygui item-type suffix"
    assert "mvInputText" in shortcuts.EDITING_TYPES
    assert "mvButton" not in shortcuts.EDITING_TYPES
