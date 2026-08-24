from __future__ import annotations

import unittest

from cxreason.parsing.json import extract_json_object


class JsonParsingTest(unittest.TestCase):
    def test_extracts_fenced_json_object(self) -> None:
        text = 'Here is the result:\n```json\n{"criterion": "CTR"}\n```'

        self.assertEqual(extract_json_object(text), {"criterion": "CTR"})


if __name__ == "__main__":
    unittest.main()
