from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

LOGGER = logging.getLogger(__name__)


class DesktopTrayIcon:
    """Optional Windows tray menu for controlling the desktop host."""

    def __init__(
        self,
        *,
        title: str,
        on_show: Callable[[], None],
        on_stop_listening: Callable[[], None],
        on_close: Callable[[], None],
        logger: logging.Logger | None = None,
    ) -> None:
        self._title = title
        self._on_show = on_show
        self._on_stop_listening = on_stop_listening
        self._on_close = on_close
        self._logger = logger or LOGGER
        self._icon: Any | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._thread is not None:
            return True
        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError as exc:
            self._logger.warning("desktop_tray_unavailable reason=%s", exc)
            return False

        image = Image.new("RGBA", (64, 64), (5, 7, 11, 255))
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [(32, 6), (56, 20), (56, 44), (32, 58), (8, 44), (8, 20)],
            outline=(125, 249, 255, 255),
            fill=(8, 13, 20, 255),
            width=3,
        )
        draw.ellipse((24, 24, 40, 40), fill=(74, 222, 128, 255))

        self._icon = pystray.Icon(
            self._title,
            image,
            self._title,
            menu=pystray.Menu(
                pystray.MenuItem("Show Window", self._menu_show, default=True),
                pystray.MenuItem("Stop Listening", self._menu_stop_listening),
                pystray.MenuItem("Close Program", self._menu_close),
            ),
        )
        self._thread = threading.Thread(
            target=self._run_icon,
            name="whispertome-tray",
            daemon=True,
        )
        self._thread.start()
        self._logger.info("desktop_tray_started")
        return True

    def stop(self) -> None:
        icon = self._icon
        self._icon = None
        if icon is not None:
            try:
                icon.stop()
            except Exception as exc:
                self._logger.warning("desktop_tray_stop_failed error=%s", exc)
        self._thread = None

    def _run_icon(self) -> None:
        assert self._icon is not None
        try:
            self._icon.run()
        except Exception as exc:
            self._logger.warning("desktop_tray_failed error=%s", exc)

    def _menu_show(self, _icon: Any, _item: Any) -> None:
        self._on_show()

    def _menu_stop_listening(self, _icon: Any, _item: Any) -> None:
        self._on_stop_listening()

    def _menu_close(self, _icon: Any, _item: Any) -> None:
        self._on_close()
