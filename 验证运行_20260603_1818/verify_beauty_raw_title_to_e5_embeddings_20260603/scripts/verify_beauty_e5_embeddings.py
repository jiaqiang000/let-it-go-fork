from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def discover_vault_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "let-it-go论文思路").is_dir() and (path / "推荐系统论文").is_dir():
            return path
    raise RuntimeError(
        "Could not discover the Obsidian research vault root. "
        "Pass --vault-root explicitly."
    )


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    vault_root = discover_vault_root(script_path.parent)
    project_root = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Verify whether Beauty raw item titles can be aligned and encoded "
            "to reproduce Let It Go content embeddings."
        )
    )
    parser.add_argument("--vault-root", type=Path, default=vault_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "beauty_title_e5_verify",
    )
    parser.add_argument("--checkpoint", default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-warm", type=int, default=500)
    parser.add_argument("--sample-cold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Actually run SentenceTransformer encoding and compare cosine similarity.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Encode all warm/cold items instead of samples. Requires --encode.",
    )
    return parser.parse_args()


def parse_one_record(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def iter_meta_rows(meta_path: Path, keep_asins: set[str]) -> Iterable[dict]:
    with meta_path.open("r", encoding="utf-8") as file:
        for line in file:
            obj = parse_one_record(line)
            if obj is None:
                continue

            asin = str(obj.get("asin", "")).strip()
            if asin in keep_asins:
                yield obj


def normalize_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value).strip()
    if value is None:
        return ""
    return str(value).strip()


def normalize_title_for_encoding(value: object) -> str:
    if value is None:
        return "None"
    return normalize_text(value)


def load_mapping(path: Path) -> dict[str, int]:
    with path.open("rb") as file:
        mapping = pickle.load(file)
    if not isinstance(mapping, dict):
        raise TypeError(f"{path} is not a pickle dict.")
    return {str(key): int(value) for key, value in mapping.items()}


def build_ordered_items(mapping: dict[str, int], meta_by_asin: dict[str, dict], split: str) -> list[dict]:
    rows = []
    for asin, item_id in sorted(mapping.items(), key=lambda pair: pair[1]):
        meta = meta_by_asin.get(asin)
        if meta is None:
            raise KeyError(f"Missing metadata for {split} item asin={asin}")

        rows.append(
            {
                "split": split,
                "asin": asin,
                "item_id": item_id,
                "title": normalize_text(meta.get("title")),
                "title_for_encoding": normalize_title_for_encoding(meta.get("title")),
                "description": normalize_text(meta.get("description")),
                "categories": json.dumps(meta.get("categories", []), ensure_ascii=False),
            }
        )
    return rows


def sample_indices(total: int, sample_size: int, seed: int) -> list[int]:
    if sample_size >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), sample_size))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1).clip(min=1e-12)
    b_norm = np.linalg.norm(b, axis=1).clip(min=1e-12)
    return (a * b).sum(axis=1) / (a_norm * b_norm)


def summarize_cosines(values: np.ndarray) -> dict:
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


def main() -> None:
    args = parse_args()
    vault_root = args.vault_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = vault_root / "let-it-go论文思路" / "let-it-go" / "row_data" / "meta_Beauty.json"
    processed_dir = vault_root / "let-it-go论文思路" / "letitgo-data" / "data" / "beauty" / "processed"
    embedding_dir = vault_root / "let-it-go论文思路" / "letitgo-data" / "data" / "beauty" / "item_embeddings"

    warm_map = load_mapping(processed_dir / "item2index_warm.pkl")
    cold_map = load_mapping(processed_dir / "item2index_cold.pkl")
    keep_asins = set(warm_map) | set(cold_map)
    meta_by_asin = {
        str(row["asin"]): row for row in iter_meta_rows(meta_path, keep_asins)
    }

    warm_rows = build_ordered_items(warm_map, meta_by_asin, "warm")
    cold_rows = build_ordered_items(cold_map, meta_by_asin, "cold")

    author_warm = np.load(embedding_dir / "embeddings_warm.npy")
    author_cold = np.load(embedding_dir / "embeddings_cold.npy")

    warm_indices = list(range(len(warm_rows))) if args.full else sample_indices(
        len(warm_rows), args.sample_warm, args.seed
    )
    cold_indices = list(range(len(cold_rows))) if args.full else sample_indices(
        len(cold_rows), args.sample_cold, args.seed + 1
    )

    sampled_rows = []
    for split, rows, indices in (("warm", warm_rows, warm_indices), ("cold", cold_rows, cold_indices)):
        for position in indices[:50]:
            row = dict(rows[position])
            row["position_in_embedding_file"] = position
            sampled_rows.append(row)

    write_csv(output_dir / "ordered_item_text_sample.csv", sampled_rows)

    summary = {
        "meta_path": str(meta_path),
        "processed_dir": str(processed_dir),
        "embedding_dir": str(embedding_dir),
        "checkpoint": args.checkpoint,
        "warm_items": len(warm_rows),
        "cold_items": len(cold_rows),
        "metadata_rows_matched": len(meta_by_asin),
        "author_warm_shape": list(author_warm.shape),
        "author_cold_shape": list(author_cold.shape),
        "warm_shape_matches_mapping": author_warm.shape[0] == len(warm_rows),
        "cold_shape_matches_mapping": author_cold.shape[0] == len(cold_rows),
        "encode_requested": bool(args.encode),
        "full_encode": bool(args.full),
        "sample_warm": len(warm_indices),
        "sample_cold": len(cold_indices),
    }

    if args.encode:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            summary["encode_error"] = (
                "sentence_transformers is not available in this Python environment: "
                f"{type(exc).__name__}: {exc}"
            )
            (output_dir / "linkage_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise

        model = SentenceTransformer(args.checkpoint)

        comparison_rows = []
        worst_rows = []
        all_cosines = {}
        for split, rows, indices, author_embeddings in (
            ("warm", warm_rows, warm_indices, author_warm),
            ("cold", cold_rows, cold_indices, author_cold),
        ):
            texts = [rows[index]["title_for_encoding"] for index in indices]
            generated = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=True,
                normalize_embeddings=False,
            )
            generated = np.asarray(generated)
            expected = author_embeddings[indices]
            cosines = cosine_similarity(generated, expected)
            all_cosines[split] = summarize_cosines(cosines)

            for local_idx, position in enumerate(indices[:200]):
                row = dict(rows[position])
                row["position_in_embedding_file"] = position
                row["cosine_to_author_embedding"] = float(cosines[local_idx])
                comparison_rows.append(row)

            worst_local_indices = np.argsort(cosines)[:50]
            for local_idx in worst_local_indices:
                position = indices[int(local_idx)]
                row = dict(rows[position])
                row["position_in_embedding_file"] = position
                row["cosine_to_author_embedding"] = float(cosines[int(local_idx)])
                worst_rows.append(row)

        summary["cosine_summary"] = all_cosines
        write_csv(output_dir / "cosine_sample_comparison.csv", comparison_rows)
        write_csv(output_dir / "worst_cosine_comparison.csv", worst_rows)

    (output_dir / "linkage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
