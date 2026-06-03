from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from verify_zvuk_vectors import (  # noqa: E402
    build_mapping_frame,
    cosine_similarity,
    mapping_range_summary,
    summarize_difference,
)


class ZvukHelperTests(unittest.TestCase):
    def test_mapping_range_summary_reports_contiguous_ranges(self) -> None:
        summary = mapping_range_summary({9: 1, 13: 3, 11: 2})

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min_item_id"], 1)
        self.assertEqual(summary["max_item_id"], 3)
        self.assertTrue(summary["is_contiguous"])

    def test_cosine_similarity_identical_vectors_are_one(self) -> None:
        values = cosine_similarity(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        self.assertTrue(np.allclose(values, np.ones(2), atol=1e-7))

    def test_summarize_difference_reports_exact_match(self) -> None:
        generated = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        author = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        summary = summarize_difference(generated, author)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["max_abs_diff"], 0.0)
        self.assertTrue(summary["allclose_atol_1e_6"])

    def test_build_mapping_frame_matches_raw_track_id_dtype(self) -> None:
        frame = build_mapping_frame({9: 1}, {95: 2})

        self.assertEqual(frame.schema["track_id"], pl.Int32)
        self.assertEqual(frame.schema["item_id"], pl.Int64)


if __name__ == "__main__":
    unittest.main()
