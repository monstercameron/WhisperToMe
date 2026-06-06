from __future__ import annotations

import time
import unittest

from whispertome.runtime.events import (
    NULL_BUS,
    EventBus,
    RuntimeEvent,
    attach_speech_reactions,
)


class _RecorderUi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def status(self, text: str, detail: str = "") -> None:
        self.calls.append(("status", text, detail))

    def activity(self, value: float) -> None:
        self.calls.append(("activity", round(value, 2)))

    def user_text(self, text: str) -> None:
        self.calls.append(("user_text", text))


class EventBusTests(unittest.TestCase):
    def test_publish_before_start_dispatches_inline(self) -> None:
        bus = EventBus()
        seen: list[RuntimeEvent] = []
        bus.subscribe(seen.append)
        bus.publish(RuntimeEvent(source="stt", kind="start"))
        self.assertEqual(len(seen), 1)  # inline, never dropped

    def test_async_dispatch_after_start(self) -> None:
        bus = EventBus()
        seen: list[RuntimeEvent] = []
        bus.subscribe(seen.append)
        bus.start()
        try:
            bus.publish(RuntimeEvent(source="stt", kind="final", text="hi"))
            deadline = time.time() + 1.0
            while not seen and time.time() < deadline:
                time.sleep(0.01)
        finally:
            bus.stop()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "hi")

    def test_listener_errors_are_isolated(self) -> None:
        bus = EventBus()
        good: list[RuntimeEvent] = []
        bus.subscribe(lambda e: (_ for _ in ()).throw(ValueError("boom")))
        bus.subscribe(good.append)
        bus.publish(RuntimeEvent(source="stt", kind="start"))  # must not raise
        self.assertEqual(len(good), 1)

    def test_null_bus_drops_silently(self) -> None:
        NULL_BUS.publish(RuntimeEvent(source="stt", kind="start"))  # no-op, no error

    def test_speech_reactions_map_events_to_ui(self) -> None:
        bus = EventBus()
        ui = _RecorderUi()
        attach_speech_reactions(bus, ui)
        bus.publish(RuntimeEvent(source="stt", kind="start", value=0.65))
        bus.publish(RuntimeEvent(source="stt", kind="final", text="hello world"))
        # non-stt events are ignored
        bus.publish(RuntimeEvent(source="tts", kind="start"))
        kinds = [c[0] for c in ui.calls]
        self.assertIn("status", kinds)
        self.assertIn("activity", kinds)
        self.assertIn(("user_text", "hello world"), ui.calls)
        self.assertEqual(ui.calls[0], ("status", "transcribing", ""))
        self.assertEqual(ui.calls[-1], ("activity", 0.0))


if __name__ == "__main__":
    unittest.main()
