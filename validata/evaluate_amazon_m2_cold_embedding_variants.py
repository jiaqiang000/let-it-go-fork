"""Amazon-M2 cold embedding 字段消融推荐评测。

本脚本只做诊断评测，不重新训练模型，也不修改 run.py / source/ 主逻辑。
核心目的：固定已经训练好的 A1/A2 推荐空间，只替换 cold item content embedding，
检查字段消融是否会影响 cold-start 推荐指标。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "letitgo-data" / "data" / "amazon_m2_fr"
SERVER_DATA_ROOT = Path("/root/letitgo-data/data/amazon_m2_fr")
DEFAULT_DATA_ROOT = SERVER_DATA_ROOT if SERVER_DATA_ROOT.exists() else LOCAL_DATA_ROOT
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "amazon_m2_cold_embedding_variants"

LOCAL_CKPT_ROOT = (
    PROJECT_ROOT.parent
    / "temp_202606_实验文件记录"
    / "temp_20260605"
    / "自然字段完整度分组评测"
    / "amazon_m2_A1A2_20260605"
)
LOCAL_A1_CHECKPOINT = (
    LOCAL_CKPT_ROOT
    / "offline-90381974e88d4868a302f761cb96a70d"
    / "recommender"
    / "epoch=8-step=6615.ckpt"
)
LOCAL_A2_CHECKPOINT = (
    LOCAL_CKPT_ROOT
    / "offline-09a0f19af9b04b908ef51015948abb8b"
    / "recommender"
    / "epoch=9-step=7350.ckpt"
)
SERVER_A1_CHECKPOINT = Path(
    "/hy-tmp/letitgo_ckpt/amazon_m2_A0A1_20260605/"
    "offline-d2c7d67301fa451b894b5862cf5cacdd/recommender/epoch=8-step=6615.ckpt"
)
SERVER_A2_CHECKPOINT = Path(
    "/hy-tmp/letitgo_ckpt/amazon_m2_baseline_20260605/"
    "offline-3ec73ef18570400a8f690c7de74b8ccd/recommender/epoch=6-step=5145.ckpt"
)
DEFAULT_A1_CHECKPOINT = LOCAL_A1_CHECKPOINT if LOCAL_A1_CHECKPOINT.exists() else SERVER_A1_CHECKPOINT
DEFAULT_A2_CHECKPOINT = LOCAL_A2_CHECKPOINT if LOCAL_A2_CHECKPOINT.exists() else SERVER_A2_CHECKPOINT

METADATA_COLUMNS = ("title", "brand", "color", "size", "model", "material", "author")
FIELD_NAMES = ("color", "size", "model", "material")
FIELD_GROUP_ORDER = ("weak_0_1", "mid_2", "strong_3_4", "missing_metadata")
MISSING_STRINGS = {"", "null", "none", "nan", "[]"}

VARIANT_FIELDS = {
    "original_author": None,
    # 中文注释：control_full 用于校验 raw products -> E5 -> PCA 链路是否能复现作者 embedding。
    "control_full": ("title", "brand", "color", "size", "model", "material", "author"),
    # 中文注释：drop_four 是当前主要的结构化字段缺失版本，保留 title/brand/author。
    "drop_four": ("title", "brand", "author"),
    "title_brand_only": ("title", "brand"),
    # 中文注释：下面这些扩展组用于判断 title/brand 是否才是主要信息来源。
    "title_only": ("title",),
    "brand_only": ("brand",),
    "no_title": ("brand", "color", "size", "model", "material", "author"),
    "no_brand": ("title", "color", "size", "model", "material", "author"),
    "no_author": ("title", "brand", "color", "size", "model", "material"),
    "no_title_brand": ("color", "size", "model", "material", "author"),
    # 中文注释：单字段删除用于判断 color/size/model/material 哪个字段更可能有贡献。
    "no_color": ("title", "brand", "size", "model", "material", "author"),
    "no_size": ("title", "brand", "color", "model", "material", "author"),
    "no_model": ("title", "brand", "color", "size", "material", "author"),
    "no_material": ("title", "brand", "color", "size", "model", "author"),
    "structured_four_only": ("color", "size", "model", "material"),
    "empty_text": (),
}
DEFAULT_VARIANTS = ",".join(VARIANT_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Amazon-M2 recommendation metrics under cold embedding field variants."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a1-checkpoint", type=Path, default=DEFAULT_A1_CHECKPOINT)
    parser.add_argument("--a2-checkpoint", type=Path, default=DEFAULT_A2_CHECKPOINT)
    parser.add_argument("--models", default="A1,A2", help="逗号分隔；支持 A1,A2。")
    parser.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        help=f"逗号分隔；支持 {DEFAULT_VARIANTS}。",
    )
    parser.add_argument("--sentence-checkpoint", default="intfloat/multilingual-e5-base")
    parser.add_argument("--locale", default="FR", help="Amazon-M2 locale；默认和原论文实验一致使用 FR。")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=0,
        help="本地 smoke test 用；0 表示评测全部 ground-truth users。",
    )
    parser.add_argument(
        "--sample-cold-gt-users",
        action="store_true",
        help="配合 --max-eval-users 使用；优先抽真实答案为 cold item 的用户，方便本地 smoke test。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查路径、字段、样本数，不加载 E5，不加载 checkpoint，不跑预测。",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="只生成/校验 variant cold embeddings，不加载 A1/A2，不跑推荐预测。",
    )
    return parser.parse_args()


def parse_csv_list(value: str, allowed: set[str], name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(items) - allowed)
    if unsupported:
        raise ValueError(f"{name} 不支持：{unsupported}；允许值：{sorted(allowed)}")
    return items


def resolve_paths(data_root: Path, products_path: Path, output_dir: Path) -> dict[str, Path]:
    data_root = data_root.expanduser().resolve()
    return {
        "data_root": data_root,
        "products": products_path.expanduser().resolve(),
        "output_dir": output_dir.expanduser().resolve(),
        "test": data_root / "processed" / "test_interactions.parquet",
        "ground_truth": data_root / "processed" / "ground_truth.parquet",
        "item2index_warm": data_root / "processed" / "item2index_warm.pkl",
        "item2index_cold": data_root / "processed" / "item2index_cold.pkl",
        "warm_embeddings": data_root / "item_embeddings" / "embeddings_warm.npy",
        "cold_embeddings": data_root / "item_embeddings" / "embeddings_cold.npy",
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


def is_present_value(value: Any) -> bool:
    return bool(clean_cell(value))


def compose_metadata_text(row: dict[str, object], fields: tuple[str, ...]) -> str:
    parts = []
    for column in fields:
        value = clean_cell(row.get(column))
        if value:
            parts.append(f"{column}: {value}")
    return "; ".join(parts)


def read_cold_products(
    products_path: Path,
    cold_item2index: dict[str, int],
    locale: str,
    chunksize: int = 200_000,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    required_columns = ["id", "locale", *METADATA_COLUMNS]
    header = pd.read_csv(products_path, nrows=0).columns.tolist()
    missing = sorted(set(required_columns) - set(header))
    if missing:
        raise ValueError(f"products_train.csv 缺少字段：{missing}")

    keep_ids = set(cold_item2index)
    products_by_id: dict[str, dict[str, str]] = {}
    total_rows = 0
    matched_rows = 0

    chunks = pd.read_csv(
        products_path,
        usecols=required_columns,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in chunks:
        total_rows += len(chunk)
        # 中文注释：Amazon-M2 products_train.csv 包含多个 locale。
        # 原论文这里使用 France locale，所以生成 variant embedding 时必须先过滤 FR，
        # 否则同一商品 id 在其他 locale 的文本可能污染字段消融结果。
        matched = chunk[(chunk["locale"] == locale) & (chunk["id"].isin(keep_ids))]
        matched_rows += len(matched)
        for record in matched.to_dict("records"):
            product_id = str(record["id"])
            products_by_id.setdefault(
                product_id,
                {key: clean_cell(value) for key, value in record.items()},
            )

    rows = []
    for position, (raw_item_id, model_item_id) in enumerate(
        sorted(cold_item2index.items(), key=lambda pair: pair[1])
    ):
        product = products_by_id.get(raw_item_id)
        metadata_found = product is not None
        product = product or {}
        row = {
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

    summary = {
        "products_total_rows": total_rows,
        "matched_rows": matched_rows,
        "matched_unique_ids": len(products_by_id),
        "cold_items": len(rows),
        "missing_metadata": sum(1 for row in rows if not row["metadata_found"]),
        "locale": locale,
    }
    return rows, summary


def assign_field_group(present_count: int, metadata_found: bool = True) -> str:
    if not metadata_found:
        return "missing_metadata"
    if present_count <= 1:
        return "weak_0_1"
    if present_count == 2:
        return "mid_2"
    return "strong_3_4"


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


def encode_variant(model, rows: list[dict[str, object]], variant: str, batch_size: int) -> np.ndarray:
    fields = VARIANT_FIELDS[variant]
    if fields is None:
        raise ValueError("original_author 不需要 E5 encode。")

    texts = [compose_metadata_text(row, fields) for row in rows]
    print(f"Encoding variant={variant}, cold_items={len(texts)}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_cold_embeddings(
    variants: list[str],
    cold_rows: list[dict[str, object]],
    author_cold: np.ndarray,
    checkpoint: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pl.DataFrame]:
    variant_embeddings: dict[str, np.ndarray] = {}
    summary_rows = []

    if "original_author" in variants:
        variant_embeddings["original_author"] = np.asarray(author_cold, dtype=np.float32)
        row = summarize_values(np.ones(author_cold.shape[0], dtype=np.float32))
        row.update({"variant": "original_author", "compared_to": "author_cold"})
        summary_rows.append(row)

    encode_variants = [variant for variant in variants if variant != "original_author"]
    sentence_model = None
    if encode_variants:
        from sentence_transformers import SentenceTransformer

        print(f"Loading SentenceTransformer: {checkpoint}")
        sentence_model = SentenceTransformer(checkpoint)

    for variant in encode_variants:
        assert sentence_model is not None
        embeddings = encode_variant(sentence_model, cold_rows, variant, batch_size)
        variant_embeddings[variant] = embeddings
        cosines = cosine_similarity(embeddings, author_cold)
        row = summarize_values(cosines)
        row.update({"variant": variant, "compared_to": "author_cold"})
        summary_rows.append(row)

    return variant_embeddings, pl.from_dicts(summary_rows)


def import_letitgo_runtime() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def load_embedding_manager(checkpoint_path: Path):
    manager_path = checkpoint_path.parent.parent / "embedding_manager.pkl"
    if not manager_path.is_file():
        raise FileNotFoundError(
            "找不到和 checkpoint 对应的 embedding_manager.pkl："
            f"\ncheckpoint={checkpoint_path}\nembedding_manager={manager_path}"
        )

    from source.embedding_manager import EmbeddingManager

    return EmbeddingManager.load(str(manager_path))


def transform_with_checkpoint_manager(
    checkpoint_path: Path,
    warm_embeddings_path: Path,
    cold_variant_embeddings: np.ndarray,
):
    import torch

    manager = load_embedding_manager(checkpoint_path)
    warm_raw = np.load(warm_embeddings_path)

    # 中文注释：这里必须复用 checkpoint 旁边的 embedding_manager.pkl。
    # 不能对 variant embedding 重新 fit PCA，否则 cold item 会进入另一个坐标系，
    # 和已训练好的 warm item 空间不再对齐。
    warm = manager.transform(warm_raw)
    cold = manager.transform(cold_variant_embeddings)
    return torch.tensor(warm).float(), torch.tensor(cold).float()


def build_model(model_name: str):
    from source.recommender import SASRecModel
    from source.winter.recommender import SASRecModelWithTrainableDelta

    params = dict(
        num_items=42647,
        embedding_dim=64,
        num_blocks=2,
        num_heads=1,
        intermediate_dim=64,
        p=0.3,
        max_length=64,
    )
    if model_name == "A1":
        return SASRecModel(**params)
    if model_name == "A2":
        return SASRecModelWithTrainableDelta(max_delta_norm=0.5, **params)
    raise ValueError(f"不支持的模型：{model_name}")


def load_recommender(
    model_name: str,
    checkpoint_path: Path,
    warm_embeddings,
    cold_embeddings,
    topk: int,
):
    from run import add_cold_item_embeddings
    from source.winter.recommender import ColdStartSequentialRecommender

    model = build_model(model_name)
    model.set_pretrained_item_embeddings(
        warm_embeddings.clone(),
        add_padding_embedding=True,
        freeze=model_name == "A2",
    )
    recommender = ColdStartSequentialRecommender.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        remove_seen=True,
        metrics=["NDCG", "Recall"],
        topk=topk,
        map_location="cpu",
    )
    add_cold_item_embeddings(recommender.model, cold_embeddings)
    recommender.recommend_cold_items = True
    recommender.eval()
    return recommender


def limit_eval_users(
    test_interactions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    max_eval_users: int,
    sample_cold_gt_users: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if max_eval_users <= 0:
        return test_interactions, ground_truth

    # 中文注释：本地 smoke test 如果只取 ground_truth 前几个用户，可能全是 warm GT，
    # 这样无法检查 cold-start 指标链路；因此提供 cold GT 用户抽样开关。
    user_source = ground_truth.filter(pl.col("is_cold")) if sample_cold_gt_users else ground_truth
    users = user_source.get_column("user_id").unique(maintain_order=True).head(max_eval_users)
    user_values = users.to_list()
    return (
        test_interactions.filter(pl.col("user_id").is_in(user_values)),
        ground_truth.filter(pl.col("user_id").is_in(user_values)),
    )


def predict_recommendations(
    recommender,
    test_interactions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: str,
) -> pl.DataFrame:
    from lightning import Trainer
    from source.dataset import TestCausalDataset
    from torch.utils.data import DataLoader

    dataset = TestCausalDataset(
        test_interactions,
        add_labels=False,
        max_length=recommender.model.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=devices,
    )
    recommendations = pl.concat(trainer.predict(recommender, dataloaders=dataloader))
    return recommendations.with_columns(pl.col("user_id").cast(ground_truth.schema["user_id"]))


def build_ground_truth_groups(ground_truth: pl.DataFrame, cold_rows: list[dict[str, object]]) -> pl.DataFrame:
    field_profile = pl.from_dicts(
        [
            {
                "item_id": row["item_id"],
                "field_group": row["field_group"],
                "present_field_count": row["present_field_count"],
            }
            for row in cold_rows
        ]
    )
    cold_ground_truth = ground_truth.filter(pl.col("is_cold"))
    grouped = cold_ground_truth.join(field_profile, on="item_id", how="left")
    return grouped.with_columns(
        [
            pl.col("field_group").fill_null("missing_metadata"),
            pl.col("present_field_count").fill_null(0).cast(pl.Int64),
        ]
    )


def build_count_table(cold_rows: list[dict[str, object]], ground_truth_groups: pl.DataFrame) -> pl.DataFrame:
    field_profile = pl.from_dicts(
        [{"field_group": row["field_group"], "item_id": row["item_id"]} for row in cold_rows]
    )
    item_counts = field_profile.group_by("field_group").agg(pl.len().alias("cold_items"))
    gt_counts = ground_truth_groups.group_by("field_group").agg(
        [
            pl.len().alias("cold_ground_truth_rows"),
            pl.col("user_id").n_unique().alias("gt_user_id_count"),
        ]
    )
    groups = pl.DataFrame({"field_group": list(FIELD_GROUP_ORDER)})
    return (
        groups.join(item_counts, on="field_group", how="left")
        .join(gt_counts, on="field_group", how="left")
        .fill_null(0)
        .with_columns(
            [
                pl.col("cold_items").cast(pl.Int64),
                pl.col("cold_ground_truth_rows").cast(pl.Int64),
                pl.col("gt_user_id_count").cast(pl.Int64),
            ]
        )
    )


def compute_overall_metrics(
    model_name: str,
    variant: str,
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    topk: int,
) -> pl.DataFrame:
    from source.winter.evaluation.metrics import ColdStartOfflineMetrics

    evaluator = ColdStartOfflineMetrics(metrics=["NDCG", "Recall"], topk=topk)
    metrics = evaluator(recommendations, ground_truth)
    row = {"model": model_name, "variant": variant}
    for key, value in metrics.items():
        row[key] = float(value)
    return pl.from_dicts([row])


def compute_group_metrics(
    model_name: str,
    variant: str,
    recommendations: pl.DataFrame,
    ground_truth_groups: pl.DataFrame,
    topk: int,
) -> pl.DataFrame:
    from source.winter.evaluation.metrics import ColdStartOfflineMetrics

    evaluator = ColdStartOfflineMetrics(metrics=["NDCG", "Recall"], topk=topk)
    rows = []
    for group_name in FIELD_GROUP_ORDER:
        group_gt = ground_truth_groups.filter(pl.col("field_group") == group_name)
        if len(group_gt) == 0:
            metrics = {f"cold_NDCG@{topk}": 0.0, f"cold_Recall@{topk}": 0.0}
        else:
            group_predictions = recommendations.filter(
                pl.col("user_id").is_in(group_gt.get_column("user_id"))
            )
            metrics = evaluator(group_predictions, group_gt)

        rows.append(
            {
                "model": model_name,
                "variant": variant,
                "field_group": group_name,
                "cold_ground_truth_rows": len(group_gt),
                "gt_user_id_count": group_gt.get_column("user_id").n_unique() if len(group_gt) else 0,
                f"cold_NDCG@{topk}": float(metrics[f"cold_NDCG@{topk}"]),
                f"cold_Recall@{topk}": float(metrics[f"cold_Recall@{topk}"]),
            }
        )
    return pl.from_dicts(rows)


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.data_root, args.products_path, args.output_dir)
    require_files(
        paths,
        [
            "products",
            "test",
            "ground_truth",
            "item2index_warm",
            "item2index_cold",
            "warm_embeddings",
            "cold_embeddings",
        ],
    )

    models = parse_csv_list(args.models, {"A1", "A2"}, "models")
    variants = parse_csv_list(args.variants, set(VARIANT_FIELDS), "variants")
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cold_item2index = normalize_item2index(load_pickle(paths["item2index_cold"]), "item2index_cold")
    warm_item2index = normalize_item2index(load_pickle(paths["item2index_warm"]), "item2index_warm")
    cold_rows, product_summary = read_cold_products(
        paths["products"],
        cold_item2index,
        locale=args.locale,
    )
    ground_truth = pl.read_parquet(paths["ground_truth"])
    test_interactions = pl.read_parquet(paths["test"])
    test_interactions, ground_truth = limit_eval_users(
        test_interactions,
        ground_truth,
        args.max_eval_users,
        args.sample_cold_gt_users,
    )
    ground_truth_groups = build_ground_truth_groups(ground_truth, cold_rows)
    count_table = build_count_table(cold_rows, ground_truth_groups)

    count_table.write_csv(output_dir / "field_group_counts.csv")
    pl.from_dicts(cold_rows).write_csv(output_dir / "field_profile_cold_items.csv")

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths["data_root"]),
        "products_path": str(paths["products"]),
        "output_dir": str(output_dir),
        "sentence_checkpoint": args.sentence_checkpoint,
        "locale": args.locale,
        "models": models,
        "variants": variants,
        "max_eval_users": args.max_eval_users,
        "sample_cold_gt_users": args.sample_cold_gt_users,
        "check_only": args.check_only,
        "skip_prediction": args.skip_prediction,
        "product_summary": product_summary,
        "recommend_cold_items": True,
        "filter_cold_items": False,
    }

    print("Amazon-M2 cold embedding variant evaluation")
    print("data_root:", paths["data_root"])
    print("products:", paths["products"])
    print("output_dir:", output_dir)
    print("warm_items:", len(warm_item2index))
    print("cold_items:", len(cold_rows))
    print("ground_truth_rows:", len(ground_truth))
    print("test_interaction_rows:", len(test_interactions))
    print(count_table)

    if args.check_only:
        write_manifest(output_dir, manifest)
        print("CHECK_ONLY: 路径、字段和样本检查完成；未加载 E5，未加载 checkpoint，未跑预测。")
        return

    author_cold = np.load(paths["cold_embeddings"])
    variant_embeddings, embedding_summary = build_cold_embeddings(
        variants=variants,
        cold_rows=cold_rows,
        author_cold=author_cold,
        checkpoint=args.sentence_checkpoint,
        batch_size=args.encode_batch_size,
    )
    embedding_summary.write_csv(output_dir / "variant_embedding_summary.csv")
    print(embedding_summary)

    if args.skip_prediction:
        write_manifest(output_dir, manifest)
        print("SKIP_PREDICTION: embedding variant 已生成和校验；未加载 A1/A2，未跑推荐预测。")
        return

    import_letitgo_runtime()
    checkpoints = {
        "A1": args.a1_checkpoint.expanduser().resolve(),
        "A2": args.a2_checkpoint.expanduser().resolve(),
    }
    for model_name in models:
        if not checkpoints[model_name].is_file():
            raise FileNotFoundError(f"{model_name} checkpoint 不存在：{checkpoints[model_name]}")

    overall_tables = []
    group_tables = []
    total_runs = len(models) * len(variants)
    run_index = 0

    for model_name in models:
        checkpoint_path = checkpoints[model_name]
        for variant in variants:
            run_index += 1
            print()
            print(f"===== [{run_index}/{total_runs}] model={model_name} variant={variant} START =====")
            print("checkpoint:", checkpoint_path)

            warm_embeddings, cold_embeddings = transform_with_checkpoint_manager(
                checkpoint_path=checkpoint_path,
                warm_embeddings_path=paths["warm_embeddings"],
                cold_variant_embeddings=variant_embeddings[variant],
            )
            recommender = load_recommender(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                warm_embeddings=warm_embeddings,
                cold_embeddings=cold_embeddings,
                topk=args.topk,
            )
            recommendations = predict_recommendations(
                recommender=recommender,
                test_interactions=test_interactions,
                ground_truth=ground_truth,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                accelerator=args.accelerator,
                devices=args.devices,
            )
            overall = compute_overall_metrics(model_name, variant, recommendations, ground_truth, args.topk)
            group = compute_group_metrics(model_name, variant, recommendations, ground_truth_groups, args.topk)
            overall_tables.append(overall)
            group_tables.append(group)
            print(overall)
            print(group)
            print(f"===== [{run_index}/{total_runs}] model={model_name} variant={variant} DONE =====")

    overall_metrics = pl.concat(overall_tables)
    group_metrics = pl.concat(group_tables)
    overall_metrics.write_csv(output_dir / "variant_overall_metrics.csv")
    group_metrics.write_csv(output_dir / "variant_field_group_metrics.csv")

    manifest["checkpoints"] = {name: str(path) for name, path in checkpoints.items()}
    manifest["topk"] = args.topk
    manifest["eval_batch_size"] = args.eval_batch_size
    manifest["encode_batch_size"] = args.encode_batch_size
    manifest["num_workers"] = args.num_workers
    manifest["accelerator"] = args.accelerator
    manifest["devices"] = args.devices
    write_manifest(output_dir, manifest)
    print("DONE: variant_overall_metrics.csv 和 variant_field_group_metrics.csv 已输出。")


if __name__ == "__main__":
    main()
