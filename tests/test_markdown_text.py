from __future__ import annotations

import unittest

from whispertome.text.markdown import parse_spoken_markdown


class MarkdownTextTests(unittest.TestCase):
    def test_fenced_script_is_not_read_as_tts_text(self) -> None:
        parsed = parse_spoken_markdown(
            "Here is the script:\n"
            "```script\n"
            "print('hello')\n"
            "```\n"
            "Run it when ready."
        )

        self.assertNotIn("```", parsed.speech_text)
        self.assertNotIn("print('hello')", parsed.speech_text)
        self.assertIn("I put the script on screen.", parsed.speech_text)
        self.assertEqual(len(parsed.code_blocks), 1)
        self.assertEqual(parsed.code_blocks[0].language, "script")
        self.assertEqual(parsed.code_blocks[0].code, "print('hello')")
        self.assertEqual(parsed.primary_code_block, parsed.code_blocks[0])

    def test_code_only_response_gets_spoken_placeholder(self) -> None:
        parsed = parse_spoken_markdown("```python\nprint('hello')\n```")

        self.assertEqual(parsed.speech_text, "I put the python code on screen.")
        self.assertEqual(parsed.display_text, "[PYTHON shown in the TUI]")

    def test_primary_code_block_prefers_script_language(self) -> None:
        parsed = parse_spoken_markdown(
            "```python\nprint('ignore')\n```\n"
            "```script\nsay hello\n```"
        )

        assert parsed.primary_code_block is not None
        self.assertEqual(parsed.primary_code_block.language, "script")
        self.assertEqual(parsed.primary_code_block.code, "say hello")

    def test_plain_text_passes_through(self) -> None:
        parsed = parse_spoken_markdown("No code here.")

        self.assertEqual(parsed.speech_text, "No code here.")
        self.assertEqual(parsed.display_text, "No code here.")
        self.assertEqual(parsed.code_blocks, ())

    def test_unfenced_go_code_is_not_read_as_tts_text(self) -> None:
        parsed = parse_spoken_markdown(
            'Here is Go FizzBuzz: package main import "fmt" func main() { '
            "for i := 1; i <= 100; i++ { switch { case i%15 == 0: "
            'fmt.Println("FizzBuzz") } } }'
        )

        self.assertEqual(len(parsed.code_blocks), 1)
        self.assertEqual(parsed.code_blocks[0].language, "go")
        self.assertIn("package main", parsed.code_blocks[0].code)
        self.assertNotIn("package main", parsed.speech_text)
        self.assertIn("I put the go code on screen.", parsed.speech_text)

    def test_unterminated_fence_is_not_read_as_tts_text(self) -> None:
        parsed = parse_spoken_markdown(
            "Here is the script:\n```script\nsay hello\nthen pause"
        )

        self.assertEqual(len(parsed.code_blocks), 1)
        self.assertEqual(parsed.code_blocks[0].language, "script")
        self.assertEqual(parsed.code_blocks[0].code, "say hello\nthen pause")
        self.assertNotIn("say hello", parsed.speech_text)
        self.assertIn("I put the script on screen.", parsed.speech_text)


if __name__ == "__main__":
    unittest.main()
