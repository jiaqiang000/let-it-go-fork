import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_amazon_m2_degraded_view_sanity_check import (
    build_hit_detail,
    compose_variant_text,
    summarize_metrics,
    truncate_words,
)


class DegradedViewSanityCheckTest(unittest.TestCase):
    def test_truncate_words_keeps_first_n_tokens(self):
        self.assertEqual(truncate_words("un deux trois quatre", 2), "un deux")
        self.assertEqual(truncate_words("un   deux\t trois", 2), "un deux")
        self.assertEqual(truncate_words("", 3), "")

    def test_compose_variant_text_handles_full_truncated_and_no_title(self):
        row = {
            "title": "alpha beta gamma delta epsilon",
            "brand": "BrandX",
            "color": "red",
            "size": "",
            "model": "M1",
            "material": None,
            "author": "",
        }

        full = compose_variant_text(row, "full_content_zero_delta", title_token_limit=3)
        truncated = compose_variant_text(row, "title_trunc_3_zero_delta", title_token_limit=3)
        no_title = compose_variant_text(row, "no_title_zero_delta", title_token_limit=3)

        self.assertEqual(
            full,
            "title: alpha beta gamma delta epsilon; brand: BrandX; color: red; model: M1",
        )
        self.assertEqual(
            truncated,
            "title: alpha beta gamma; brand: BrandX; color: red; model: M1",
        )
        self.assertEqual(no_title, "brand: BrandX; color: red; model: M1")

    def test_build_hit_detail_ranks_predictions_and_computes_ndcg(self):
        recommendations = pd.DataFrame(
            [
                {"user_id": 1, "item_id": 10, "rating": 0.8},
                {"user_id": 1, "item_id": 11, "rating": 0.9},
                {"user_id": 2, "item_id": 20, "rating": 0.3},
                {"user_id": 2, "item_id": 21, "rating": 0.4},
            ]
        )
        ground_truth = pd.DataFrame(
            [
                {"user_id": 1, "item_id": 10, "field_group": "weak_0_1"},
                {"user_id": 2, "item_id": 22, "field_group": "strong_3_4"},
            ]
        )

        detail = build_hit_detail(recommendations, ground_truth, topk=10)

        first = detail[detail["user_id"] == 1].iloc[0]
        second = detail[detail["user_id"] == 2].iloc[0]
        self.assertTrue(bool(first["hit"]))
        self.assertEqual(int(first["rank"]), 2)
        self.assertAlmostEqual(first["ndcg_contribution@10"], 1.0 / math.log2(3.0))
        self.assertFalse(bool(second["hit"]))
        self.assertEqual(second["recall_contribution@10"], 0.0)

    def test_summarize_metrics_reports_group_and_overall_means(self):
        detail = pd.DataFrame(
            [
                {
                    "field_group": "weak_0_1",
                    "hit": True,
                    "item_id": 10,
                    "rank": 1,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 1.0,
                },
                {
                    "field_group": "weak_0_1",
                    "hit": False,
                    "item_id": 12,
                    "rank": pd.NA,
                    "recall_contribution@10": 0.0,
                    "ndcg_contribution@10": 0.0,
                },
                {
                    "field_group": "strong_3_4",
                    "hit": True,
                    "item_id": 20,
                    "rank": 2,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 0.5,
                },
            ]
        )

        summary = summarize_metrics("demo", detail, topk=10)
        overall = summary[summary["field_group"] == "all"].iloc[0]
        weak = summary[summary["field_group"] == "weak_0_1"].iloc[0]

        self.assertEqual(overall["variant"], "demo")
        self.assertEqual(int(overall["ground_truth_rows"]), 3)
        self.assertAlmostEqual(overall["NDCG@10"], 0.5)
        self.assertAlmostEqual(overall["Recall@10"], 2 / 3)
        self.assertAlmostEqual(weak["NDCG@10"], 0.5)
        self.assertAlmostEqual(weak["Recall@10"], 0.5)


if __name__ == "__main__":
    unittest.main()
