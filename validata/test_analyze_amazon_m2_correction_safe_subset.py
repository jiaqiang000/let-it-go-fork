import unittest

import polars as pl

from validata.analyze_amazon_m2_correction_safe_subset import (
    build_record_metrics,
    compare_against_baseline,
    dataframe_to_markdown,
    summarize_oracle_records,
)


class CorrectionSafeSubsetTest(unittest.TestCase):
    def test_build_record_metrics_assigns_rank_hit_and_ndcg(self):
        recommendations = pl.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "item_id": [10, 20, 30, 40, 50, 60],
                "rating": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            }
        )
        cold_ground_truth = pl.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [20, 99],
                "field_group": ["weak_0_1", "strong_3_4"],
            }
        )

        records = build_record_metrics(
            group="generated_top1_alpha0.1",
            recommendations=recommendations,
            cold_ground_truth=cold_ground_truth,
            metric_topk=3,
        ).sort("user_id")

        self.assertEqual(records.get_column("rank").to_list(), [2, None])
        self.assertEqual(records.get_column("hit").to_list(), [1, 0])
        self.assertAlmostEqual(records.row(0, named=True)["ndcg"], 1.0 / 1.5849625, places=6)
        self.assertEqual(records.row(1, named=True)["ndcg"], 0.0)

    def test_compare_against_baseline_reports_delta_and_status(self):
        baseline = pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                "item_id": [10, 20, 30],
                "field_group": ["weak_0_1", "weak_0_1", "mid_2"],
                "rank": [1, None, 3],
                "hit": [1, 0, 1],
                "ndcg": [1.0, 0.0, 0.5],
            }
        )
        generated = pl.DataFrame(
            {
                "group": ["generated_top1_alpha0.1"] * 3,
                "user_id": [1, 2, 3],
                "item_id": [10, 20, 30],
                "field_group": ["weak_0_1", "weak_0_1", "mid_2"],
                "rank": [2, 4, None],
                "hit": [1, 1, 0],
                "ndcg": [0.6, 0.43, 0.0],
            }
        )

        compared = compare_against_baseline(baseline, generated).sort("user_id")

        self.assertEqual(compared.get_column("oracle_status").to_list(), ["worse", "better", "worse"])
        self.assertEqual(compared.get_column("delta_hit").to_list(), [0, 1, -1])
        self.assertAlmostEqual(compared.row(1, named=True)["delta_ndcg"], 0.43, places=6)

    def test_summarize_oracle_records_counts_better_worse_same(self):
        records = pl.DataFrame(
            {
                "group": ["g1", "g1", "g1", "g1"],
                "field_group": ["weak_0_1", "weak_0_1", "strong_3_4", "strong_3_4"],
                "delta_ndcg": [0.2, -0.1, 0.0, 0.3],
                "delta_hit": [1, -1, 0, 1],
            }
        )

        summary = summarize_oracle_records(records, by=["group", "field_group"]).sort("field_group")

        weak = summary.filter(pl.col("field_group") == "weak_0_1").row(0, named=True)
        strong = summary.filter(pl.col("field_group") == "strong_3_4").row(0, named=True)
        self.assertEqual(weak["records"], 2)
        self.assertEqual(weak["better_records"], 1)
        self.assertEqual(weak["worse_records"], 1)
        self.assertEqual(strong["same_records"], 1)
        self.assertAlmostEqual(strong["better_rate"], 0.5)

    def test_dataframe_to_markdown_does_not_require_tabulate(self):
        table = pl.DataFrame({"group": ["g1"], "mean_delta_ndcg": [0.1234567]}).to_pandas()
        markdown = dataframe_to_markdown(table)

        self.assertIn("| group | mean_delta_ndcg |", markdown)
        self.assertIn("| g1 | 0.123457 |", markdown)


if __name__ == "__main__":
    unittest.main()
