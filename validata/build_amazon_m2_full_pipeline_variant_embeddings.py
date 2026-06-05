"""生成 Amazon-M2 full-pipeline retrain 用的 warm/cold 字段 variant embeddings。

本脚本只生成输入 embedding 文件，不训练模型，不改 run.py。
用途：让后续训练从同一种字段版本的 warm/cold content embedding 开始，
而不是只在评测阶段替换 cold embedding。
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "letitgo-data" / "data" / "amazon_m2_fr"
SERVER_DATA_ROOT = Path("/root/letitgo-data/data/amazon_m2_fr")
DEFAULT_DATA_ROOT = SERVER_DATA_ROOT if SERVER_DATA_ROOT.exists() else LOCAL_DATA_ROOT
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "amazon_m2_full_pipeline_variant_embeddings"

METADATA_COLUMNS = ("title", "brand", "color", "size", "model", "material", "author")
FIELD_NAMES = ("color", "size", "model", "material")
MISSING_STRINGS = {"", "null", "none", "nan", "[]"}

VARIANT_FIELDS: dict[str, tuple[str, ...] | None] = {
    "original_author": None,
    "control_full": ("title", "brand", "color", "size", "model", "material", "author"),
    "drop_four": ("title", "brand", "author"),
    "title_brand_only": ("title", "brand"),
    "title_only": ("title",),
    "brand_only": ("brand",),
    "no_title": ("brand", "color", "size", "model", "material", "author"),
    "no_brand": ("title", "color", "size", "model", "material", "author"),
    "no_author": ("title", "brand", "color", "size", "model", "material"),
    "no_title_brand": ("color", "size", "model", "material", "author"),
    "structured_four_only": ("color", "size", "model", "material"),
    "empty_text": (),
}
DEFAULT_VARIANTS = "control_full,drop_four"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build warm/cold Amazon-M2 field-variant embeddings for full-pipeline retrain."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--sentence-checkpoint", default="intfloat/multilingual-e5-base")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查输入和样本对齐，不加载 SentenceTransformer，不生成 embedding。",
    )
    return parser.parse_args()


def parse_csv_list(value: str, allowed: set[str], name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(items) - allowed)
    if unsupported:
        raise ValueError(f"{name} 不支持：{unsupported}；允许值：{sorted(allowed)}")
    if not items:
        raise ValueError(f"{name} 不能为空。")
    return items


def resolve_paths(data_root: Path, products_path: Path, output_dir: Path) -> dict[str, Path]:
    data_root = data_root.expanduser().resolve()
    return {
        "data_root": data_root,
        "products": products_path.expanduser().resolve(),
        "output_dir": output_dir.expanduser().resolve(),
        "item2index_warm": data_root / "processed" / "item2index_warm.pkl",
        "item2index_cold": data_root / "processed" / "item2index_cold.pkl",
        "author_warm_embeddings": data_root / "item_embeddings" / "embeddings_warm.npy",
        "author_cold_embeddings": data_root / "item_embeddings" / "embeddings_cold.npy",
    }


def require_files(paths: dict[str, Path], names: list[str]) -> None:
    missing = [name for name in names if not paths[name].is_file()]
    if missing:
        details = "\n".join(f"{name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(f"缺少必要输入文件：\n{details}")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_item2index(mapping: dict[Any, Any], name: str) -> dict[str, int]:
    if not mapping:
        raise ValueError(f"{name} 为空。")

    first_key, first_value = next(iter(mapping.items()))
    if isinstance(first_key, str) and isinstance(first_value, int):
        return {str(key): int(value) for key, value in mapping.items()}
    if isinstance(first_key, int) and isinstance(first_value, str):
        return {str(value): int(key) for key, value in mapping.items()}

    raise TypeError(
        f"{name} 的方向无法识别：key={type(first_key).__name__}, "
        f"value={type(first_value).__name__}"
    )


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in MISSING_STRINGS else text


def is_present_value(value: object) -> bool:
    return bool(clean_cell(value))


def compose_metadata_text(row: dict[str, object], fields: tuple[str, ...]) -> str:
    parts = []
    for column in fields:
        value = clean_cell(row.get(column))
        if value:
            parts.append(f"{column}: {value}")
    return "; ".join(parts)


def assign_field_group(present_count: int, metadata_found: bool = True) -> str:
    if not metadata_found:
        return "missing_metadata"
    if present_count <= 1:
        return "weak_0_1"
    if present_count == 2:
        return "mid_2"
    return "strong_3_4"


def read_products_by_id(
    products_path: Path,
    required_ids: set[str],
    locale: str,
    chunksize: int = 200_000,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    required_columns = ["id", "locale", *METADATA_COLUMNS]
    header = pd.read_csv(products_path, nrows=0).columns.tolist()
    missing = sorted(set(required_columns) - set(header))
    if missing:
        raise ValueError(f"products_train.csv 缺少字段：{missing}")

    products_by_id: dict[str, dict[str, str]] = {}
    total_rows = 0
    locale_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        products_path,
        usecols=required_columns,
        dtype=str,
        chunksize=chunksize,
    ):
        total_rows += len(chunk)
        locale_chunk = chunk[chunk["locale"] == locale]
        locale_rows += len(locale_chunk)

        # 中文注释：只读取 Amazon-M2 FR 链路中的 warm/cold item，避免其他 locale 污染文本。
        matched = locale_chunk[locale_chunk["id"].isin(required_ids)]
        matched_rows += len(matched)
        for record in matched.to_dict("records"):
            product_id = str(record["id"])
            products_by_id.setdefault(
                product_id,
                {key: clean_cell(value) for key, value in record.items()},
            )

    summary = {
        "products_total_rows": total_rows,
        "products_locale_rows": locale_rows,
        "matched_rows": matched_rows,
        "matched_unique_ids": len(products_by_id),
        "required_unique_ids": len(required_ids),
        "missing_required_ids": len(required_ids - set(products_by_id)),
        "locale": locale,
    }
    return products_by_id, summary


def build_item_rows(
    item2index: dict[str, int],
    products_by_id: dict[str, dict[str, str]],
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, (raw_item_id, model_item_id) in enumerate(
        sorted(item2index.items(), key=lambda pair: pair[1])
    ):
        product = products_by_id.get(raw_item_id)
        metadata_found = product is not None
        product = product or {}
        row: dict[str, object] = {
            "split": split,
            "raw_item_id": raw_item_id,
            "item_id": int(model_item_id),
            "position_in_embedding_file": int(position),
            "metadata_found": metadata_found,
        }
        for column in METADATA_COLUMNS:
            row[column] = clean_cell(product.get(column))
            row[f"{column}_present"] = is_present_value(product.get(column))

        present_count = sum(bool(row[f"{field}_present"]) for field in FIELD_NAMES)
        row["present_field_count"] = int(present_count)
        row["field_group"] = assign_field_group(present_count, metadata_found)
        rows.append(row)

    return rows


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1).clip(min=1e-12)
    b_norm = np.linalg.norm(b, axis=1).clip(min=1e-12)
    return (a * b).sum(axis=1) / (a_norm * b_norm)


def summarize_values(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float32)
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


def encode_variant(model: Any, rows: list[dict[str, object]], variant: str, batch_size: int) -> np.ndarray:
    fields = VARIANT_FIELDS[variant]
    if fields is None:
        raise ValueError("original_author 不需要 E5 encode。")

    texts = [compose_metadata_text(row, fields) for row in rows]
    print(f"Encoding split={rows[0]['split'] if rows else 'empty'} variant={variant}, items={len(texts)}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_variant(
    output_dir: Path,
    variant: str,
    warm_embeddings: np.ndarray,
    cold_embeddings: np.ndarray,
) -> dict[str, str]:
    item_embedding_dir = output_dir / variant / "item_embeddings"
    item_embedding_dir.mkdir(parents=True, exist_ok=True)
    warm_path = item_embedding_dir / "embeddings_warm.npy"
    cold_path = item_embedding_dir / "embeddings_cold.npy"
    np.save(warm_path, warm_embeddings.astype(np.float32))
    np.save(cold_path, cold_embeddings.astype(np.float32))
    return {"warm": str(warm_path), "cold": str(cold_path)}


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants, set(VARIANT_FIELDS), "variants")
    paths = resolve_paths(args.data_root, args.products_path, args.output_dir)
    require_files(
        paths,
        [
            "products",
            "item2index_warm",
            "item2index_cold",
            "author_warm_embeddings",
            "author_cold_embeddings",
        ],
    )
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    warm_item2index = normalize_item2index(load_pickle(paths["item2index_warm"]), "item2index_warm")
    cold_item2index = normalize_item2index(load_pickle(paths["item2index_cold"]), "item2index_cold")
    required_ids = set(warm_item2index) | set(cold_item2index)
    products_by_id, product_summary = read_products_by_id(
        paths["products"],
        required_ids=required_ids,
        locale=args.locale,
    )
    warm_rows = build_item_rows(warm_item2index, products_by_id, split="warm")
    cold_rows = build_item_rows(cold_item2index, products_by_id, split="cold")

    author_warm = np.load(paths["author_warm_embeddings"])
    author_cold = np.load(paths["author_cold_embeddings"])
    if author_warm.shape[0] != len(warm_rows):
        raise ValueError(f"warm embedding 行数不匹配：{author_warm.shape[0]} vs {len(warm_rows)}")
    if author_cold.shape[0] != len(cold_rows):
        raise ValueError(f"cold embedding 行数不匹配：{author_cold.shape[0]} vs {len(cold_rows)}")

    field_profile = pd.DataFrame([*warm_rows, *cold_rows])
    field_profile.to_csv(output_dir / "item_field_profile.csv", index=False)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths["data_root"]),
        "products_path": str(paths["products"]),
        "output_dir": str(output_dir),
        "sentence_checkpoint": args.sentence_checkpoint,
        "locale": args.locale,
        "variants": variants,
        "check_only": args.check_only,
        "warm_items": len(warm_rows),
        "cold_items": len(cold_rows),
        "author_warm_shape": list(author_warm.shape),
        "author_cold_shape": list(author_cold.shape),
        "product_summary": product_summary,
    }

    print("Amazon-M2 full-pipeline variant embedding builder")
    print("data_root:", paths["data_root"])
    print("products:", paths["products"])
    print("output_dir:", output_dir)
    print("variants:", ",".join(variants))
    print("warm_items:", len(warm_rows))
    print("cold_items:", len(cold_rows))
    print("author_warm_shape:", author_warm.shape)
    print("author_cold_shape:", author_cold.shape)
    print("product_summary:", product_summary)

    if args.check_only:
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("CHECK_ONLY: 输入、字段和 warm/cold 对齐检查完成；未加载 E5，未生成 embedding。")
        return

    from sentence_transformers import SentenceTransformer

    print(f"Loading SentenceTransformer: {args.sentence_checkpoint}")
    sentence_model = SentenceTransformer(args.sentence_checkpoint)

    embedding_paths: dict[str, dict[str, str]] = {}
    summary_rows: list[dict[str, object]] = []
    generated_embeddings: dict[tuple[str, str], np.ndarray] = {}

    for variant_index, variant in enumerate(variants, start=1):
        print()
        print(f"===== [{variant_index}/{len(variants)}] variant={variant} START =====")

        if variant == "original_author":
            warm_embeddings = author_warm.astype(np.float32)
            cold_embeddings = author_cold.astype(np.float32)
        else:
            warm_embeddings = encode_variant(
                sentence_model,
                warm_rows,
                variant=variant,
                batch_size=args.encode_batch_size,
            )
            cold_embeddings = encode_variant(
                sentence_model,
                cold_rows,
                variant=variant,
                batch_size=args.encode_batch_size,
            )

        embedding_paths[variant] = save_variant(output_dir, variant, warm_embeddings, cold_embeddings)
        generated_embeddings[("warm", variant)] = warm_embeddings
        generated_embeddings[("cold", variant)] = cold_embeddings

        warm_to_author = summarize_values(cosine_similarity(warm_embeddings, author_warm))
        warm_to_author.update({"split": "warm", "variant": variant, "compared_to": "author"})
        cold_to_author = summarize_values(cosine_similarity(cold_embeddings, author_cold))
        cold_to_author.update({"split": "cold", "variant": variant, "compared_to": "author"})
        summary_rows.extend([warm_to_author, cold_to_author])
        print(f"saved warm: {embedding_paths[variant]['warm']}")
        print(f"saved cold: {embedding_paths[variant]['cold']}")
        print(f"===== [{variant_index}/{len(variants)}] variant={variant} DONE =====")

    if "control_full" in variants:
        control_warm = generated_embeddings[("warm", "control_full")]
        control_cold = generated_embeddings[("cold", "control_full")]
        for variant in variants:
            if variant == "control_full":
                continue
            warm = generated_embeddings[("warm", variant)]
            cold = generated_embeddings[("cold", variant)]
            warm_to_control = summarize_values(cosine_similarity(warm, control_warm))
            warm_to_control.update({"split": "warm", "variant": variant, "compared_to": "control_full"})
            cold_to_control = summarize_values(cosine_similarity(cold, control_cold))
            cold_to_control.update({"split": "cold", "variant": variant, "compared_to": "control_full"})
            summary_rows.extend([warm_to_control, cold_to_control])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "variant_embedding_summary.csv", index=False)
    manifest["embedding_paths"] = embedding_paths
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("DONE: full-pipeline variant embeddings 已生成。")


if __name__ == "__main__":
    main()
