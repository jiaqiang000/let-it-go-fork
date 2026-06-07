import sys
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_amazon_m2_degraded_view_training_pilot import (
    build_pilot_gate_summary,
    summarize_degraded_retention,
)
from build_amazon_m2_degraded_view_training_embeddings import (
    compose_degraded_training_text,
    stable_random_value,
    title_for_variant,
)
from run_amazon_m2_degraded_view_training_pilot import build_shell_script


class DegradedViewTrainingPilotTest(unittest.TestCase):
    def test_compose_degraded_training_text_variants(self):
        row = {
            "raw_item_id": "item-1",
            "title": "alpha beta gamma delta",
            "brand": "BrandX",
            "color": "red",
            "size": "",
            "model": "M1",
            "material": None,
            "author": "",
        }
        self.assertEqual(
            compose_degraded_training_text(row, "control_full"),
            "title: alpha beta gamma delta; brand: BrandX; color: red; model: M1",
        )
        self.assertEqual(
            compose_degraded_training_text(row, "title_trunc_2"),
            "title: alpha beta; brand: BrandX; color: red; model: M1",
        )
        self.assertEqual(
            compose_degraded_training_text(row, "no_title"),
            "brand: BrandX; color: red; model: M1",
        )

    def test_random_title_dropout_is_deterministic(self):
        first = stable_random_value("item-1", seed=42, salt="random_title_dropout_p30")
        second = stable_random_value("item-1", seed=42, salt="random_title_dropout_p30")
        other_seed = stable_random_value("item-1", seed=43, salt="random_title_dropout_p30")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)

        row = {"raw_item_id": "item-1", "title": "alpha beta"}
        title, action = title_for_variant(row, "random_title_dropout_p100", seed=42)
        self.assertEqual(title, "")
        self.assertEqual(action, "drop_random")

    def test_summarize_degraded_retention_and_gate(self):
        metrics = pd.DataFrame(
            [
                {"variant": "full_content_zero_delta", "field_group": "all", "NDCG@10": 0.35, "Recall@10": 0.52},
                {"variant": "full_content_zero_delta", "field_group": "weak_0_1", "NDCG@10": 0.27, "Recall@10": 0.44},
                {"variant": "full_content_zero_delta", "field_group": "strong_3_4", "NDCG@10": 0.37, "Recall@10": 0.56},
                {"variant": "title_trunc_8_zero_delta", "field_group": "all", "NDCG@10": 0.30, "Recall@10": 0.48},
                {"variant": "title_trunc_8_zero_delta", "field_group": "weak_0_1", "NDCG@10": 0.22, "Recall@10": 0.36},
                {"variant": "title_trunc_8_zero_delta", "field_group": "strong_3_4", "NDCG@10": 0.32, "Recall@10": 0.52},
                {"variant": "no_title_zero_delta", "field_group": "all", "NDCG@10": 0.16, "Recall@10": 0.25},
            ]
        )
        retention = summarize_degraded_retention(metrics, topk=10)
        row = retention[(retention["variant"] == "title_trunc_8_zero_delta") & (retention["field_group"] == "all")].iloc[0]
        self.assertAlmostEqual(row["drop_NDCG@10"], 0.05)
        self.assertEqual(row["candidate_role"], "mild_degraded_candidate")

        controlled = pd.DataFrame([{"variant": "no_title", "drop_cold_NDCG@10": 0.12}])
        gates = build_pilot_gate_summary(retention, controlled, topk=10)
        self.assertEqual(gates.iloc[0]["status"], "weak_go")
        self.assertEqual(gates.iloc[1]["status"], "not_tested")

    def test_build_shell_script_contains_expected_matrix(self):
        args = Namespace(
            repo_root=Path("/repo/let-it-go"),
            data_root=Path("/data/amazon_m2_fr"),
            products_path=Path("/repo/let-it-go/row_data/amazon_m2_raw/products_train.csv"),
            log_dir=Path("/tmp/logs"),
            run_output_dir=Path("/tmp/run_outputs"),
            embedding_output_dir=Path("/tmp/emb"),
            checkpoint_dir=Path("/tmp/ckpt"),
            python_bin="/opt/anaconda3/envs/let-it-go-py3.11/bin/python",
            sentence_checkpoint="intfloat/multilingual-e5-base",
            encode_batch_size=128,
            project_name="pilot",
            max_epochs=3,
            devices="[0]",
            learning_rate="1e-3",
            max_delta_norm="0.5",
            check_only=False,
            dry_run=False,
            skip_embedding=False,
        )
        script = build_shell_script(args, ["control_full", "title_trunc_8"], [42, 43])
        self.assertIn("run_amazon_m2_degraded_view_training_pilot_2seeds.sh", script)
        self.assertIn("export VARIANTS=control_full,title_trunc_8", script)
        self.assertIn("export SEEDS=42,43", script)
        self.assertIn("export VARIANT_EMBEDDING_ROOT=/tmp/emb", script)
        self.assertNotIn("scripts/run.py", script)


if __name__ == "__main__":
    unittest.main()
