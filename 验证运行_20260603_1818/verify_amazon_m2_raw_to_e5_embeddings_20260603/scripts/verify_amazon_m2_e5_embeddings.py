from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_COLUMNS = ["title", "brand", "color", "size", "model", "material", "author"]
PRODUCT_COLUMNS = ["id", "locale", *METADATA_COLUMNS]
TASK_FILES = [
    "products_train.csv",
    "sessions_train.csv",
    "task1/sessions_test_task1_phase1.csv",
    "task1/sessions_test_task1_phase2.csv",
    "task1/gt_task1_phase1.csv",
    "task1/gt_task1_phase2.csv",
    "task2/sessions_test_task2_phase1.csv",
    "task2/sessions_test_task2_phase2.csv",
    "task2/gt_task2_phase1.csv",
    "task2/gt_task2_phase2.csv",
    "task3/sessions_test_task3_phase1.csv",
    "task3/sessions_test_task3_phase2.csv",
    "task3/gt_task3_phase1.csv",
    "task3/gt_task3_phase2.csv",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    verification_root = script_path.parents[1]
    project_root = script_path.parents[3]
    topic_root = project_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "Verify whether Amazon-M2 raw FR product content aligns with "
            "Let It Go amazon_m2_fr mappings and content embeddings."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root / "row_data" / "amazon_m2_raw",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=topic_root / "letitgo-data" / "data" / "amazon_m2_fr" / "processed",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=topic_root / "letitgo-data" / "data" / "amazon_m2_fr" / "item_embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=verification_root / "outputs" / "amazon_m2_alignment_verify",
    )
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--checkpoint", default="intfloat/multilingual-e5-base")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-warm", type=int, default=300)
    parser.add_argument("--sample-cold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Run SentenceTransformer encoding and compare cosine similarity.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Encode all warm/cold items instead of samples. Requires --encode.",
    )
    parser.add_argument(
        "--save-generated",
        action="store_true",
        help="When used with --encode --full, save generated warm/cold embedding npy files.",
    )
    return parser.parse_args()


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def compose_metadata_text(row: dict[str, object]) -> str:
    parts = []
    for column in METADATA_COLUMNS:
        value = clean_cell(row.get(column))
        if value:
            parts.append(f"{column}: {value}")
    return "; ".join(parts)


def load_mapping(path: Path) -> dict[str, int]:
    with path.open("rb") as file:
        mapping = pickle.load(file)
    if not isinstance(mapping, dict):
        raise TypeError(f"{path} is not a pickle dict.")
    return {str(key): int(value) for key, value in mapping.items()}


def mapping_range_summary(mapping: dict[str, int]) -> dict[str, object]:
    values = sorted(mapping.values())
    unique_values = set(values)
    min_value = min(values) if values else None
    max_value = max(values) if values else None
    return {
        "count": len(values),
        "unique_item_ids": len(unique_values),
        "min_item_id": min_value,
        "max_item_id": max_value,
        "is_unique": len(unique_values) == len(values),
        "is_contiguous": (
            bool(values)
            and len(unique_values) == len(values)
            and max_value - min_value + 1 == len(values)
        ),
    }


def read_csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def summarize_expected_files(raw_dir: Path) -> list[dict[str, object]]:
    rows = []
    for relative in TASK_FILES:
        path = raw_dir / relative
        rows.append(
            {
                "relative_path": relative,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "columns": read_csv_header(path) if path.exists() and path.suffix == ".csv" else [],
            }
        )
    return rows


def maybe_progress(iterable: Iterable, desc: str, enabled: bool) -> Iterable:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, unit="chunk")


def read_filtered_products(
    products_path: Path,
    keep_ids: set[str],
    locale: str,
    chunksize: int,
    show_progress: bool = True,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    header = read_csv_header(products_path)
    missing_columns = [column for column in PRODUCT_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(f"{products_path} is missing required columns: {missing_columns}")

    products_by_id: dict[str, dict[str, str]] = {}
    duplicate_matched_ids: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    total_rows = 0
    matched_rows = 0

    chunks = pd.read_csv(
        products_path,
        usecols=PRODUCT_COLUMNS,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in maybe_progress(chunks, "Scanning products_train.csv", show_progress):
        total_rows += len(chunk)
        locale_counts.update(chunk["locale"].tolist())

        matched = chunk[(chunk["locale"] == locale) & (chunk["id"].isin(keep_ids))]
        matched_rows += len(matched)
        for record in matched.to_dict("records"):
            product_id = str(record["id"])
            if product_id in products_by_id:
                duplicate_matched_ids[product_id] += 1
                continue
            products_by_id[product_id] = {key: clean_cell(value) for key, value in record.items()}

    summary = {
        "total_product_rows": total_rows,
        "pandas_default_na_rules": True,
        "locale_counts": dict(sorted(locale_counts.items())),
        "matched_rows_for_locale": matched_rows,
        "unique_matched_ids_for_locale": len(products_by_id),
        "duplicate_matched_ids": dict(duplicate_matched_ids),
    }
    return products_by_id, summary


def build_ordered_item_rows(
    mapping: dict[str, int],
    products_by_id: dict[str, dict[str, str]],
    split: str,
) -> list[dict[str, object]]:
    rows = []
    for position, (product_id, item_id) in enumerate(sorted(mapping.items(), key=lambda pair: pair[1])):
        product = products_by_id.get(product_id)
        metadata_text = compose_metadata_text(product or {})
        row = {
            "split": split,
            "source_product_id": product_id,
            "item_id": item_id,
            "position_in_embedding_file": position,
            "has_product_metadata": int(product is not None),
            "metadata_text": metadata_text,
            "metadata_text_len": len(metadata_text),
        }
        for column in METADATA_COLUMNS:
            row[column] = clean_cell((product or {}).get(column))
        rows.append(row)
    return rows


def sample_indices(total: int, sample_size: int, seed: int) -> list[int]:
    if sample_size >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), sample_size))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_field_completeness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = len(rows)
    summary_rows = []
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for column in METADATA_COLUMNS:
            nonempty = sum(bool(clean_cell(row.get(column))) for row in split_rows)
            summary_rows.append(
                {
                    "split": split,
                    "field": column,
                    "items": len(split_rows),
                    "nonempty": nonempty,
                    "nonempty_rate": nonempty / len(split_rows) if split_rows else None,
                }
            )
    for column in METADATA_COLUMNS:
        nonempty = sum(bool(clean_cell(row.get(column))) for row in rows)
        summary_rows.append(
            {
                "split": "all",
                "field": column,
                "items": total,
                "nonempty": nonempty,
                "nonempty_rate": nonempty / total if total else None,
            }
        )
    return summary_rows


def summarize_parquet(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    summary: dict[str, object] = {
        "path": str(path),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": frame.columns.tolist(),
    }
    if "item_id" in frame.columns:
        summary["item_id_min"] = int(frame["item_id"].min())
        summary["item_id_max"] = int(frame["item_id"].max())
        summary["unique_item_ids"] = int(frame["item_id"].nunique())
    if "is_cold" in frame.columns:
        counts = frame["is_cold"].value_counts(dropna=False).to_dict()
        summary["is_cold_counts"] = {str(key): int(value) for key, value in counts.items()}
    return summary


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1).clip(min=1e-12)
    b_norm = np.linalg.norm(b, axis=1).clip(min=1e-12)
    return (a * b).sum(axis=1) / (a_norm * b_norm)


def summarize_cosines(values: np.ndarray) -> dict[str, object]:
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "below_0_999": int((values < 0.999).sum()),
        "below_0_99": int((values < 0.99).sum()),
        "below_0_95": int((values < 0.95).sum()),
    }


def select_lowest_cosine_rows(
    rows: list[dict[str, object]],
    indices: list[int],
    cosines: list[float] | np.ndarray,
    limit: int,
) -> list[dict[str, object]]:
    order = np.argsort(np.asarray(cosines))[:limit]
    selected = []
    for local_idx in order:
        position = indices[int(local_idx)]
        row = dict(rows[position])
        row["position_in_embedding_file"] = position
        row["cosine_to_author_embedding"] = float(cosines[int(local_idx)])
        selected.append(row)
    return selected


def compare_embeddings(
    rows: list[dict[str, object]],
    indices: list[int],
    author_embeddings: np.ndarray,
    checkpoint: str,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(checkpoint)
    texts = [str(rows[index]["metadata_text"]) for index in indices]
    generated = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    generated = np.asarray(generated)
    expected = author_embeddings[indices]
    cosines = cosine_similarity(generated, expected)

    comparison_rows = []
    for local_idx, position in enumerate(indices[:500]):
        row = dict(rows[position])
        row["cosine_to_author_embedding"] = float(cosines[local_idx])
        comparison_rows.append(row)

    worst_rows = select_lowest_cosine_rows(rows, indices, cosines, limit=100)
    return summarize_cosines(cosines), comparison_rows, worst_rows, generated


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    products_path = raw_dir / "products_train.csv"
    warm_map = load_mapping(processed_dir / "item2index_warm.pkl")
    cold_map = load_mapping(processed_dir / "item2index_cold.pkl")
    keep_ids = set(warm_map) | set(cold_map)

    expected_files = summarize_expected_files(raw_dir)
    products_by_id, product_scan_summary = read_filtered_products(
        products_path=products_path,
        keep_ids=keep_ids,
        locale=args.locale,
        chunksize=args.chunksize,
    )

    warm_rows = build_ordered_item_rows(warm_map, products_by_id, "warm")
    cold_rows = build_ordered_item_rows(cold_map, products_by_id, "cold")
    all_rows = warm_rows + cold_rows

    missing_product_rows = [
        {
            "split": row["split"],
            "source_product_id": row["source_product_id"],
            "item_id": row["item_id"],
            "position_in_embedding_file": row["position_in_embedding_file"],
        }
        for row in all_rows
        if not row["has_product_metadata"]
    ]

    author_warm = np.load(embedding_dir / "embeddings_warm.npy", mmap_mode="r")
    author_cold = np.load(embedding_dir / "embeddings_cold.npy", mmap_mode="r")

    parquet_summary = {
        path.name: summarize_parquet(path)
        for path in [
            processed_dir / "train_interactions.parquet",
            processed_dir / "val_interactions.parquet",
            processed_dir / "test_interactions.parquet",
            processed_dir / "ground_truth.parquet",
        ]
    }

    sample_rows = []
    for rows in (warm_rows, cold_rows):
        sample_rows.extend(rows[:50])
    write_csv(output_dir / "amazon_m2_ordered_item_text_sample.csv", sample_rows)
    write_csv(
        output_dir / "amazon_m2_missing_product_ids.csv",
        missing_product_rows,
        fieldnames=["split", "source_product_id", "item_id", "position_in_embedding_file"],
    )
    write_csv(
        output_dir / "amazon_m2_field_completeness.csv",
        summarize_field_completeness(all_rows),
    )
    write_csv(
        output_dir / "amazon_m2_expected_files.csv",
        expected_files,
        fieldnames=["relative_path", "exists", "size_bytes", "columns"],
    )

    warm_range = mapping_range_summary(warm_map)
    cold_range = mapping_range_summary(cold_map)
    summary: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(output_dir),
        "locale": args.locale,
        "checkpoint": args.checkpoint,
        "expected_file_count": len(expected_files),
        "expected_files_present": all(row["exists"] for row in expected_files),
        "product_scan_summary": product_scan_summary,
        "warm_mapping": warm_range,
        "cold_mapping": cold_range,
        "mapping_total_items": len(keep_ids),
        "matched_product_metadata": len(products_by_id),
        "missing_product_metadata": len(missing_product_rows),
        "warm_embedding_shape": list(author_warm.shape),
        "cold_embedding_shape": list(author_cold.shape),
        "warm_embedding_rows_match_mapping": int(author_warm.shape[0]) == len(warm_map),
        "cold_embedding_rows_match_mapping": int(author_cold.shape[0]) == len(cold_map),
        "embedding_dim": int(author_warm.shape[1]) if len(author_warm.shape) == 2 else None,
        "cold_embedding_dim": int(author_cold.shape[1]) if len(author_cold.shape) == 2 else None,
        "parquet_summary": parquet_summary,
        "encode_requested": bool(args.encode),
        "full_encode": bool(args.full),
        "save_generated": bool(args.save_generated),
    }

    if args.encode:
        warm_indices = (
            list(range(len(warm_rows)))
            if args.full
            else sample_indices(len(warm_rows), args.sample_warm, args.seed)
        )
        cold_indices = (
            list(range(len(cold_rows)))
            if args.full
            else sample_indices(len(cold_rows), args.sample_cold, args.seed + 1)
        )
        warm_cosine, warm_comparison_rows, warm_worst_rows, generated_warm = compare_embeddings(
            warm_rows,
            warm_indices,
            np.asarray(author_warm),
            args.checkpoint,
            args.batch_size,
        )
        cold_cosine, cold_comparison_rows, cold_worst_rows, generated_cold = compare_embeddings(
            cold_rows,
            cold_indices,
            np.asarray(author_cold),
            args.checkpoint,
            args.batch_size,
        )
        summary["cosine_summary"] = {"warm": warm_cosine, "cold": cold_cosine}
        write_csv(
            output_dir / "amazon_m2_e5_cosine_comparison.csv",
            [*warm_comparison_rows, *cold_comparison_rows],
        )
        write_csv(
            output_dir / "amazon_m2_e5_worst_cosines.csv",
            [*warm_worst_rows, *cold_worst_rows],
        )
        if args.full and args.save_generated:
            np.save(output_dir / "generated_embeddings_warm.npy", generated_warm)
            np.save(output_dir / "generated_embeddings_cold.npy", generated_cold)

    summary_path = output_dir / "amazon_m2_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
