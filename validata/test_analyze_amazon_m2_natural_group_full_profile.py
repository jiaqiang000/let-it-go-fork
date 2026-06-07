import unittest

import pandas as pd

from validata.analyze_amazon_m2_natural_group_full_profile import (
    add_book_like_features,
    build_proxy_gap_summary,
    summarize_subset_metrics,
)


class NaturalGroupFullProfileTest(unittest.TestCase):
    def test_add_book_like_features_uses_multiple_nonexclusive_proxies(self):
        table = pd.DataFrame(
            [
                {
                    "raw_item_id": "2013949952",
                    "title": "Bibliocollège - Le Horla et six contes fantastiques",
                    "brand": "Hachette Éducation",
                    "author": "Guy de Maupassant",
                },
                {
                    "raw_item_id": "B09ABC",
                    "title": "Coque silicone rouge pour iPhone",
                    "brand": "Generic",
                    "author": "",
                },
            ]
        )

        result = add_book_like_features(table)

        first = result.iloc[0]
        second = result.iloc[1]
        self.assertTrue(bool(first["book_like_proxy"]))
        self.assertTrue(bool(first["book_like_raw_id"]))
        self.assertTrue(bool(first["book_like_author"]))
        self.assertTrue(bool(first["book_like_publisher_brand"]))
        self.assertTrue(bool(first["book_like_title_terms"]))
        self.assertGreaterEqual(int(first["book_like_score"]), 4)
        self.assertFalse(bool(second["book_like_proxy"]))

    def test_summarize_subset_metrics_recomputes_group_gaps_inside_subset(self):
        hit_detail = pd.DataFrame(
            [
                {
                    "field_group": "weak_0_1",
                    "hit": True,
                    "item_id": 1,
                    "book_like_proxy": False,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 0.5,
                },
                {
                    "field_group": "weak_0_1",
                    "hit": False,
                    "item_id": 2,
                    "book_like_proxy": False,
                    "recall_contribution@10": 0.0,
                    "ndcg_contribution@10": 0.0,
                },
                {
                    "field_group": "strong_3_4",
                    "hit": True,
                    "item_id": 3,
                    "book_like_proxy": False,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 1.0,
                },
                {
                    "field_group": "strong_3_4",
                    "hit": True,
                    "item_id": 4,
                    "book_like_proxy": True,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 0.25,
                },
            ]
        )

        summary = summarize_subset_metrics(
            hit_detail=hit_detail,
            subsets={"non_book": hit_detail["book_like_proxy"] == False},
            topk=10,
        )

        row = summary[summary["subset"] == "non_book"].iloc[0]
        self.assertEqual(int(row["gt_rows_weak_0_1"]), 2)
        self.assertEqual(int(row["gt_rows_strong_3_4"]), 1)
        self.assertAlmostEqual(row["cold_NDCG@10_weak_0_1"], 0.25)
        self.assertAlmostEqual(row["cold_NDCG@10_strong_3_4"], 1.0)
        self.assertAlmostEqual(row["ndcg_gap_strong_minus_weak@10"], 0.75)

    def test_build_proxy_gap_summary_compares_weak_and_strong_within_bucket(self):
        hit_detail = pd.DataFrame(
            [
                {
                    "field_group": "weak_0_1",
                    "raw_id_type": "asin_B",
                    "item_id": 1,
                    "hit": False,
                    "recall_contribution@10": 0.0,
                    "ndcg_contribution@10": 0.0,
                },
                {
                    "field_group": "strong_3_4",
                    "raw_id_type": "asin_B",
                    "item_id": 2,
                    "hit": True,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 0.5,
                },
                {
                    "field_group": "weak_0_1",
                    "raw_id_type": "isbn_like",
                    "item_id": 3,
                    "hit": True,
                    "recall_contribution@10": 1.0,
                    "ndcg_contribution@10": 1.0,
                },
            ]
        )

        gaps = build_proxy_gap_summary(hit_detail, proxies=["raw_id_type"], topk=10)
        asin = gaps[gaps["proxy_bucket"] == "asin_B"].iloc[0]
        isbn = gaps[gaps["proxy_bucket"] == "isbn_like"].iloc[0]

        self.assertTrue(bool(asin["has_weak_and_strong_gt"]))
        self.assertAlmostEqual(asin["ndcg_gap_strong_minus_weak@10"], 0.5)
        self.assertFalse(bool(isbn["has_weak_and_strong_gt"]))


if __name__ == "__main__":
    unittest.main()
