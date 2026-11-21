import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import REFUSAL, generate_answer, retrieve


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class TestRag(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {
                "source": "sample.pdf",
                "page": 2,
                "chunk": 1,
                "text": "Classification predicts discrete categories.",
                "score": 0.9,
            }
        ]

    def test_retrieval_ranking_and_threshold(self):
        index = {
            "chunks": [
                {"text": "relevant", "source": "a.pdf", "page": 1},
                {"text": "unrelated", "source": "a.pdf", "page": 2},
            ],
            "vectors": np.array(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
        }

        hits = retrieve(
            "question",
            index,
            FakeEmbedder(),
            top_k=2,
            min_score=0.5,
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["page"], 1)

    @patch("src.rag.requests.post")
    def test_no_evidence_skips_generation(self, post):
        self.assertEqual(generate_answer("question", []), REFUSAL)
        post.assert_not_called()

    @patch("src.rag.requests.post")
    def test_invalid_source_is_rejected(self, post):
        response = Mock()
        response.json.return_value = {
            "response": "An unsupported answer [S9]."
        }
        post.return_value = response

        answer = generate_answer("question", self.hits)

        self.assertIn("without valid source references", answer)

    @patch("src.rag.requests.post")
    def test_valid_source_identifier_is_accepted(self, post):
        response = Mock()
        response.json.return_value = {
            "response": "Classification predicts categories [S1]."
        }
        post.return_value = response

        answer = generate_answer("question", self.hits)

        self.assertEqual(
            answer,
            "Classification predicts categories [S1].",
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)