from __future__ import annotations

import unittest

from whispertome.text.streaming import SpokenTextChunker


class StreamingTextTests(unittest.TestCase):
    def test_emits_sentence_chunks_from_deltas(self) -> None:
        chunker = SpokenTextChunker()

        self.assertEqual(chunker.push("Hello "), [])
        self.assertEqual(chunker.push("world. "), ["Hello world."])
        self.assertEqual(chunker.push("Next"), [])
        self.assertEqual(chunker.finish(), ["Next"])

    def test_suppresses_fenced_code_until_final_placeholder(self) -> None:
        chunker = SpokenTextChunker()

        self.assertEqual(chunker.push("Here is Go:\n```go\npackage main"), [])
        self.assertEqual(chunker.push("\nfunc main() {}\n```"), [])

        chunks = chunker.finish()

        self.assertEqual(chunks, ["Here is Go: I put the go code on screen."])

    def test_unfenced_code_is_suppressed_at_finish(self) -> None:
        chunker = SpokenTextChunker()

        self.assertEqual(
            chunker.push('Here is Go FizzBuzz: package main import "fmt" func main() {}'),
            [],
        )

        chunks = chunker.finish()

        self.assertEqual(chunks, ["Here is Go FizzBuzz: I put the go code on screen."])


if __name__ == "__main__":
    unittest.main()
