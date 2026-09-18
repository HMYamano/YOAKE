"""The console log handler, against the codepage that used to break it.

`trainer.py` writes Japanese log lines, and one of them carried U+2014 EM DASH.
cp932 - the Japanese Windows console codepage - cannot encode it, so
`StreamHandler.emit` raised, `handleError` turned that into a traceback on
stderr, and the log line was dropped. The utf-8 file handler kept it, so the
loss showed up only on screen. These tests pin both halves: the line survives
the console, and the file still gets the real character.
"""

from __future__ import annotations

import io
import logging

import pytest

from htrtdetr.utils.logging import _SafeStreamHandler, get_logger

EM_DASH = "—"


def cp932_stream() -> io.TextIOWrapper:
    """A stream that behaves like the Japanese Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp932", write_through=True)


def test_cp932_really_cannot_encode_an_em_dash():
    """The premise. Without this the rest of the file proves nothing."""
    with pytest.raises(UnicodeEncodeError):
        EM_DASH.encode("cp932")


def test_an_em_dash_does_not_drop_the_log_line(caplog):
    stream = cp932_stream()
    handler = _SafeStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    errors = []
    handler.handleError = lambda record: errors.append(record)

    logger = logging.getLogger("yoake.test.emdash")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.warning("DUMMY MODE %s metrics are meaningless", EM_DASH)

    written = stream.buffer.getvalue().decode("cp932")
    assert errors == [], "handleError ran, so the line was dropped"
    assert "DUMMY MODE" in written
    assert "metrics are meaningless" in written
    # The em dash itself cannot survive cp932; what matters is that its
    # absence costs one character rather than the whole line.
    assert EM_DASH not in written


def test_an_encodable_line_is_written_unchanged():
    """The fast path must stay byte-exact - Japanese is fine in cp932."""
    stream = cp932_stream()
    handler = _SafeStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("yoake.test.japanese")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    message = "メトリクスは無意味です"
    logger.info(message)

    assert stream.buffer.getvalue().decode("cp932").strip() == message


def test_get_logger_installs_the_safe_handler_and_a_utf8_file(tmp_path):
    log_file = tmp_path / "run.log"
    logger = get_logger("yoake.test.wiring", log_file=str(log_file))
    try:
        console = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
        assert console, "no console handler"
        assert all(isinstance(h, _SafeStreamHandler) for h in console)

        files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(files) == 1
        logger.info("DUMMY MODE %s test", EM_DASH)
        files[0].flush()
        # The file is utf-8, so it keeps the character the console could not.
        assert EM_DASH in log_file.read_text(encoding="utf-8")
    finally:
        for h in list(logger.handlers):
            h.close()
        logger.handlers = []
