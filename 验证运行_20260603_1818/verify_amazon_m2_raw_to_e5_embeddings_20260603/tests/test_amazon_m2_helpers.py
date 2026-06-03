from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from verify_amazon_m2_e5_embeddings import (  # noqa: E402
    compose_metadata_text,
    mapping_range_summary,
    read_filtered_products,
    select_lowest_cosine_rows,
)


class AmazonM2HelperTests(unittest.TestCase):
    def test_compose_metadata_text_matches_notebook_order_and_empty_handling(self) -> None:
        row = {
            "title": "A title",
            "brand": "",
            "color": "Blue",
            "size": None,
            "model": "M1",
            "material": "",
            "author": "Ada",
        }

        self.assertEqual(
            compose_metadata_text(row),
            "title: A title; color: Blue; model: M1; author: Ada",
        )

    def test_compose_metadata_text_strips_leading_separator_when_title_missing(self) -> None:
        row = {
            "title": "",
            "brand": "BrandX",
            "color": "",
            "size": "",
            "model": "",
            "material": "Cotton",
            "author": "",
        }

        self.assertEqual(
            compose_metadata_text(row),
            "brand: BrandX; material: Cotton",
        )

    def test_mapping_range_summary_reports_contiguous_ranges(self) -> None:
        summary = mapping_range_summary({"a": 3, "b": 1, "c": 2})

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min_item_id"], 1)
        self.assertEqual(summary["max_item_id"], 3)
        self.assertTrue(summary["is_contiguous"])

    def test_select_lowest_cosine_rows_keeps_lowest_positions(self) -> None:
        rows = [{"source_product_id": str(index), "item_id": index} for index in range(31)]
        rows[10]["source_product_id"] = "a"
        rows[20]["source_product_id"] = "b"
        rows[30]["source_product_id"] = "c"

        selected = select_lowest_cosine_rows(rows, [10, 20, 30], [0.9, 0.7, 0.8], limit=2)

        self.assertEqual([row["source_product_id"] for row in selected], ["b", "c"])
        self.assertEqual([row["position_in_embedding_file"] for row in selected], [20, 30])
        self.assertEqual([row["cosine_to_author_embedding"] for row in selected], [0.7, 0.8])

    def test_read_filtered_products_uses_notebook_default_na_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "products_train.csv"
            path.write_text(
                "\n".join(
                    [
                        "id,locale,title,price,brand,color,size,model,material,author,desc",
                        "B1,FR,Title,1,Brand,Blue,,NA,nan,,",
                    ]
                ),
                encoding="utf-8",
            )

            products, _ = read_filtered_products(path, {"B1"}, "FR", chunksize=10, show_progress=False)

        self.assertEqual(
            compose_metadata_text(products["B1"]),
            "title: Title; brand: Brand; color: Blue",
        )


if __name__ == "__main__":
    unittest.main()
