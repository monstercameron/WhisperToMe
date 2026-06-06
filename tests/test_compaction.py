from __future__ import annotations

import unittest

from whispertome.llm.base import LlmResponse
from whispertome.llm.compaction import ConversationCompactionManager
from whispertome.llm.conversation_tools import ConversationController


def _resp(input_tokens, cached_tokens):
    return LlmResponse(
        text="ok", latency_ms=1.0, model="m", response_id=None,
        input_tokens=input_tokens, cached_tokens=cached_tokens, output_tokens=10,
    )


class _FakeController:
    def __init__(self):
        self.calls = 0

    def auto_compact(self):
        self.calls += 1
        return {"ok": True, "action": "auto_compacted"}


class _FakeResponder:
    def __init__(self, *, text="summary", raise_on_generate=False):
        self._text = text
        self._raise = raise_on_generate
        self.compacted = None
        self.reset = False

    def generate(self, user_text, **_kw):
        if self._raise:
            raise RuntimeError("offline")
        return LlmResponse(text=self._text, latency_ms=1.0, model="m", response_id=None)

    def reset_conversation(self):
        self.reset = True

    def compact_conversation(self, summary):
        self.compacted = summary


def _mgr(controller, **kw):
    opts = dict(
        controller=controller, idle_enabled=False, usage_enabled=True,
        context_window_tokens=1000, threshold_pct=0.75,
    )
    opts.update(kw)
    return ConversationCompactionManager(**opts)


class UsageTriggerTests(unittest.TestCase):
    def test_below_threshold_does_not_compact(self):
        fc = _FakeController()
        _mgr(fc).note_turn(_resp(700, 0))  # 70% < 75%
        self.assertEqual(fc.calls, 0)

    def test_above_threshold_compacts(self):
        fc = _FakeController()
        mgr = _mgr(fc)
        mgr.note_turn(_resp(700, 0))   # 70% -> no
        mgr.note_turn(_resp(900, 50))  # effective 850 -> 85% -> yes
        self.assertEqual(fc.calls, 1)

    def test_cached_tokens_are_excluded(self):
        # huge input but all cached -> effective 0 -> no compaction ("context % less the cache")
        fc = _FakeController()
        _mgr(fc).note_turn(_resp(5000, 5000))
        self.assertEqual(fc.calls, 0)

    def test_usage_disabled(self):
        fc = _FakeController()
        _mgr(fc, usage_enabled=False).note_turn(_resp(5000, 0))
        self.assertEqual(fc.calls, 0)

    def test_no_window_disables_usage(self):
        fc = _FakeController()
        _mgr(fc, context_window_tokens=0).note_turn(_resp(5000, 0))
        self.assertEqual(fc.calls, 0)

    def test_usage_compact_is_reentrant_under_turn_lock(self):
        # note_turn runs inside a turn that already holds turn_lock; compaction must not
        # self-deadlock (turn_lock is reentrant). Would hang with a plain Lock.
        from threading import RLock

        fc = _FakeController()
        lock = RLock()
        mgr = _mgr(fc, turn_lock=lock)
        with lock:  # simulate being inside the conversation turn
            mgr.note_turn(_resp(900, 0))  # 90% -> compact
        self.assertEqual(fc.calls, 1)

    def test_compact_resets_turn_counter(self):
        fc = _FakeController()
        mgr = _mgr(fc)
        mgr.note_turn(_resp(900, 0))  # compact (calls 1), turns reset
        # a fresh idle compaction now would be a no-op (no turns since)
        mgr._compact("idle")
        self.assertEqual(fc.calls, 1)


class AutoCompactTests(unittest.TestCase):
    def test_summarizes_and_compacts(self):
        ctrl = ConversationController()
        responder = _FakeResponder(text="carryover summary")
        ctrl.bind(responder)
        result = ctrl.auto_compact()
        self.assertEqual(result["action"], "auto_compacted")
        self.assertEqual(responder.compacted, "carryover summary")
        self.assertFalse(responder.reset)

    def test_falls_back_to_reset_on_failure(self):
        ctrl = ConversationController()
        responder = _FakeResponder(raise_on_generate=True)
        ctrl.bind(responder)
        result = ctrl.auto_compact()
        self.assertEqual(result["action"], "auto_reset")
        self.assertTrue(responder.reset)
        self.assertIsNone(responder.compacted)


if __name__ == "__main__":
    unittest.main()
