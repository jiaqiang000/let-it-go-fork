import unittest

import numpy as np

from validata.evaluate_amazon_m2_generated_cold_delta_probe import (
    generate_neighbor_delta,
    l2_normalize,
    parse_float_list,
    parse_int_list,
    summarize_generated_delta,
)


class GeneratedColdDeltaProbeTest(unittest.TestCase):
    def test_parse_int_and_float_lists(self):
        self.assertEqual(parse_int_list("5,1,5,10"), [1, 5, 10])
        self.assertEqual(parse_float_list("0.2,0.1,0.2"), [0.1, 0.2])

        with self.assertRaises(ValueError):
            parse_int_list("0,5")
        with self.assertRaises(ValueError):
            parse_float_list("0.1,-0.2")

    def test_generate_neighbor_delta_uses_nonnegative_similarity_weights(self):
        warm_content = l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [0.5, 0.5],
                    [-1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        cold_content = l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        warm_delta = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        )

        generated, details = generate_neighbor_delta(
            warm_content=warm_content,
            cold_content=cold_content,
            warm_delta=warm_delta,
            topk=2,
            alpha=0.5,
        )

        expected_candidate = (
            np.array([1.0, 0.0]) * 1.0
            + np.array([0.0, 1.0]) * np.sqrt(0.5)
        ) / (1.0 + np.sqrt(0.5))
        np.testing.assert_allclose(generated[0], 0.5 * expected_candidate, atol=1e-6)
        self.assertEqual(details.loc[0, "nearest_warm_item_id"], 1)
        self.assertEqual(details.loc[0, "topk"], 2)

    def test_summarize_generated_delta_reports_content_drift(self):
        cold_content = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        generated_delta = np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32)
        summary = summarize_generated_delta(
            group="generated_top5_alpha0.2",
            topk=5,
            alpha=0.2,
            cold_content=cold_content,
            generated_delta=generated_delta,
        )

        self.assertEqual(summary["group"], "generated_top5_alpha0.2")
        self.assertAlmostEqual(summary["delta_norm_mean"], 0.15, places=6)
        self.assertGreater(summary["cold_content_final_cosine_mean"], 0.98)


if __name__ == "__main__":
    unittest.main()
