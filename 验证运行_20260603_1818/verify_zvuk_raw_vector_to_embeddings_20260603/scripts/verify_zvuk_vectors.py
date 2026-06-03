from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl


RAW_FILES = {
    "zvuk-interactions.parquet": ["user_id", "session_id", "datetime", "track_id", "play_duration"],
    "zvuk-track_artist_embedding.parquet": ["track_id", "artist_id", "cluster_id", "vector"],
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    verification_root = script_path.parents[1]
    project_root = script_path.parents[3]
    topic_root = project_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "Verify whether Zvuk raw track vectors align with Let It Go "
            "zvuk mappings and item_embeddings npy files."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root / "row_data" / "zvuk_raw",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=topic_root / "letitgo-data" / "data" / "zvuk" / "processed",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=topic_root / "letitgo-data" / "data" / "zvuk" / "item_embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=verification_root / "outputs" / "zvuk_vector_alignment_verify",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100,
        help="Number of ordered item rows to write for manual inspection.",
    )
    return parser.parse_args()


def maybe_progress(iterable: Iterable, desc: str) -> Iterable:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc)


def load_mapping(path: Path) -> dict[int, int]:
    with path.open("rb") as file:
        mapping = pickle.load(file)
    if not isinstance(mapping, dict):
        raise TypeError(f"{path} is not a pickle dict.")
    return {int(key): int(value) for key, value in mapping.items()}


def mapping_range_summary(mapping: dict[int, int]) -> dict[str, object]:
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


def summarize_raw_file(path: Path, expected_columns: list[str]) -> dict[str, object]:
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    columns = list(schema.keys())
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "columns": columns,
        "expected_columns_present": all(column in columns for column in expected_columns),
        "shape": [
            int(lazy.select(pl.len()).collect().item()),
            len(columns),
        ],
        "schema": {name: str(dtype) for name, dtype in schema.items()},
    }


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


def build_mapping_frame(warm_map: dict[int, int], cold_map: dict[int, int]) -> pl.DataFrame:
    rows = []
    for split, mapping in (("warm", warm_map), ("cold", cold_map)):
        rows.extend(
            {
                "track_id": track_id,
                "item_id": item_id,
                "split": split,
            }
            for track_id, item_id in mapping.items()
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("track_id").cast(pl.Int32),
        pl.col("item_id").cast(pl.Int64),
    )


def reconstruct_ordered_vectors(raw_embedding_path: Path, mapping_frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, object]]:
    keep_track_ids = mapping_frame.get_column("track_id").to_list()
    unique_vectors = (
        pl.scan_parquet(raw_embedding_path)
        .select(["track_id", "vector"])
        .filter(pl.col("track_id").is_in(keep_track_ids))
        .unique(["track_id", "vector"])
        .collect()
    )
    per_track_counts = unique_vectors.group_by("track_id").len(name="unique_vector_count")
    ambiguous = per_track_counts.filter(pl.col("unique_vector_count") > 1)
    ordered = (
        unique_vectors.join(mapping_frame, on="track_id", how="inner")
        .sort("item_id")
    )
    summary = {
        "raw_unique_track_vector_rows": int(unique_vectors.height),
        "tracks_with_multiple_unique_vectors": int(ambiguous.height),
        "joined_rows": int(ordered.height),
        "joined_unique_track_ids": int(ordered.get_column("track_id").n_unique()),
        "joined_unique_item_ids": int(ordered.get_column("item_id").n_unique()),
    }
    return ordered, summary


def vectors_to_matrix(ordered: pl.DataFrame) -> np.ndarray:
    vectors = [
        vector
        for vector in maybe_progress(
            ordered.get_column("vector"),
            "Stacking ordered track vectors",
        )
    ]
    return np.asarray(vectors, dtype=np.float32)


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


def summarize_difference(generated: np.ndarray, author: np.ndarray) -> dict[str, object]:
    diff = generated - author
    abs_diff = np.abs(diff)
    l2 = np.linalg.norm(diff, axis=1)
    cosines = cosine_similarity(generated, author)
    return {
        "count": int(generated.shape[0]),
        "shape": list(generated.shape),
        "dtype_generated": str(generated.dtype),
        "dtype_author": str(author.dtype),
        "allclose_atol_1e_8": bool(np.allclose(generated, author, rtol=0.0, atol=1e-8)),
        "allclose_atol_1e_7": bool(np.allclose(generated, author, rtol=0.0, atol=1e-7)),
        "allclose_atol_1e_6": bool(np.allclose(generated, author, rtol=0.0, atol=1e-6)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_l2_diff": float(l2.max()),
        "mean_l2_diff": float(l2.mean()),
        "cosine": summarize_cosines(cosines),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ordered_samples(path: Path, ordered: pl.DataFrame, sample_rows: int) -> None:
    rows = []
    for record in ordered.head(sample_rows).iter_rows(named=True):
        vector = record["vector"]
        rows.append(
            {
                "split": record["split"],
                "track_id": int(record["track_id"]),
                "item_id": int(record["item_id"]),
                "position_in_embedding_file": int(record["item_id"]) - 1,
                "vector_dim": len(vector),
                "vector_head": json.dumps([float(value) for value in vector[:8]]),
            }
        )
    write_csv(path, rows)


def write_worst_rows(
    path: Path,
    ordered: pl.DataFrame,
    generated: np.ndarray,
    author: np.ndarray,
    limit: int = 100,
) -> None:
    l2 = np.linalg.norm(generated - author, axis=1)
    order = np.argsort(l2)[-limit:][::-1]
    rows = []
    records = ordered.select(["split", "track_id", "item_id"]).to_dicts()
    for position in order:
        record = records[int(position)]
        rows.append(
            {
                "split": record["split"],
                "track_id": int(record["track_id"]),
                "item_id": int(record["item_id"]),
                "position_in_embedding_file": int(position),
                "l2_diff": float(l2[int(position)]),
                "max_abs_diff": float(np.abs(generated[int(position)] - author[int(position)]).max()),
                "cosine_to_author_embedding": float(cosine_similarity(generated[int(position): int(position) + 1], author[int(position): int(position) + 1])[0]),
            }
        )
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = {
        name: summarize_raw_file(raw_dir / name, expected_columns)
        for name, expected_columns in RAW_FILES.items()
    }

    warm_map = load_mapping(processed_dir / "item2index_warm.pkl")
    cold_map = load_mapping(processed_dir / "item2index_cold.pkl")
    mapping_frame = build_mapping_frame(warm_map, cold_map)
    ordered, vector_summary = reconstruct_ordered_vectors(
        raw_dir / "zvuk-track_artist_embedding.parquet",
        mapping_frame,
    )
    generated = vectors_to_matrix(ordered)

    author_warm = np.load(embedding_dir / "embeddings_warm.npy")
    author_cold = np.load(embedding_dir / "embeddings_cold.npy")
    author_full = np.vstack([author_warm, author_cold]).astype(np.float32)

    warm_count = len(warm_map)
    generated_warm = generated[:warm_count]
    generated_cold = generated[warm_count:]

    parquet_summary = {
        path.name: summarize_parquet(path)
        for path in [
            processed_dir / "train_interactions.parquet",
            processed_dir / "val_interactions.parquet",
            processed_dir / "test_interactions.parquet",
            processed_dir / "ground_truth.parquet",
        ]
    }

    write_ordered_samples(output_dir / "zvuk_ordered_vector_sample.csv", ordered, args.sample_rows)
    write_worst_rows(output_dir / "zvuk_worst_vector_differences.csv", ordered, generated, author_full)

    summary = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(output_dir),
        "raw_summary": raw_summary,
        "warm_mapping": mapping_range_summary(warm_map),
        "cold_mapping": mapping_range_summary(cold_map),
        "mapping_total_items": int(mapping_frame.height),
        "vector_join_summary": vector_summary,
        "author_warm_shape": list(author_warm.shape),
        "author_cold_shape": list(author_cold.shape),
        "generated_shape": list(generated.shape),
        "warm_embedding_rows_match_mapping": int(author_warm.shape[0]) == len(warm_map),
        "cold_embedding_rows_match_mapping": int(author_cold.shape[0]) == len(cold_map),
        "generated_rows_match_mapping": int(generated.shape[0]) == int(mapping_frame.height),
        "full_difference": summarize_difference(generated, author_full),
        "warm_difference": summarize_difference(generated_warm, author_warm.astype(np.float32)),
        "cold_difference": summarize_difference(generated_cold, author_cold.astype(np.float32)),
        "parquet_summary": parquet_summary,
    }

    summary_path = output_dir / "zvuk_vector_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
