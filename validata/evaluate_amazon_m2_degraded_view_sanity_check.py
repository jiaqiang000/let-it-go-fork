"""Amazon-M2 degraded-view sanity check。

本脚本是 Let It Go 方法线的最后一次 Go/Stop 诊断，不训练模型，也不修改
scripts/run.py / source/ 主逻辑。它固定 A2 checkpoint，对 warm ground-truth
target item 做 content-only 反事实替换：

1. full_content_zero_delta：目标 item 使用完整 metadata content embedding，delta 置零；
2. title_trunc_N_zero_delta：目标 item 的 title 截短到 N 个 token，delta 置零；
3. no_title_zero_delta：目标 item 删除 title，delta 置零。

目的不是解释 natural weak/mid/strong 的因果根因，而是检查人工 degraded view
是否能经验上复现 weak/mid 的 ranking failure。若不能复现，就停止
training-time weak-evidence robustness / degraded-view consistency 方法候选。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent
LOCAL_DATA_ROOT = RESEARCH_ROOT / "letitgo-data" / "data" / "amazon_m2_fr"
SERVER_DATA_ROOT = Path("/root/letitgo-data/data/amazon_m2_fr")
DEFAULT_DATA_ROOT = SERVER_DATA_ROOT if SERVER_DATA_ROOT.exists() else LOCAL_DATA_ROOT
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
TEMP_ROOT = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260606"
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "degraded-view-sanity-check"

LOCAL_A2_RUN_DIR = (
    RESEARCH_ROOT
    / "temp_202606_实验文件记录"
    / "temp_20260605"
    / "自然字段完整度分组评测"
    / "amazon_m2_A1A2_20260605"
    / "offline-09a0f19af9b04b908ef51015948abb8b"
)
LOCAL_A2_CHECKPOINT = LOCAL_A2_RUN_DIR / "recommender" / "epoch=9-step=7350.ckpt"
SERVER_A2_CHECKPOINT = Path(
    "/hy-tmp/letitgo_ckpt/amazon_m2_baseline_20260605/"
    "offline-3ec73ef18570400a8f690c7de74b8ccd/recommender/epoch=6-step=5145.ckpt"
)
DEFAULT_A2_CHECKPOINT = LOCAL_A2_CHECKPOINT if LOCAL_A2_CHECKPOINT.exists() else SERVER_A2_CHECKPOINT

NATURAL_METRICS_PATH = (
    RESEARCH_ROOT
    / "temp_202606_实验文件记录"
    / "temp_20260605"
    / "自然字段完整度分组评测_FR口径修正版"
    / "field_group_eval_outputs"
    / "field_group_metrics.csv"
)

METADATA_COLUMNS = ("title", "brand", "color", "size", "model", "material", "author")
FIELD_NAMES = ("color", "size", "model", "material")
FIELD_GROUP_ORDER = ("weak_0_1", "mid_2", "strong_3_4", "missing_metadata")
MISSING_STRINGS = {"", "null", "none", "nan", "[]"}
DEFAULT_VARIANTS = "A2_original_warm_delta,full_content_zero_delta,title_trunc_8_zero_delta,no_title_zero_delta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Amazon-M2 degraded-view sanity check.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a2-checkpoint", type=Path, default=DEFAULT_A2_CHECKPOINT)
    parser.add_argument("--natural-metrics", type=Path, default=NATURAL_METRICS_PATH)
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--sentence-checkpoint", default="intfloat/multilingual-e5-base")
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
        help="smoke test 用；0 表示使用全部 warm ground-truth users。",
    )
    parser.add_argument(
        "--max-target-items",
        type=int,
        default=0,
        help="smoke test 用；0 表示对所有 warm ground-truth target items 构造 view。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查路径、warm GT 和目标 item，不加载 E5，不跑预测。",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("variants 不能为空。")
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
        "warm_embeddings": data_root / "item_embeddings" / "embeddings_warm.npy",
        "cold_embeddings": data_root / "item_embeddings" / "embeddings_cold.npy",
    }


def require_files(paths: dict[str, Path], names: list[str]) -> None:
    missing = [name for name in names if not paths[name].is_file()]
    if missing:
        details = "\n".join(f"{name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(f"缺少必要输入文件：\n{details}")


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{label}: {path}")
    return path


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


def truncate_words(text: str, limit: int) -> str:
    if limit <= 0:
        raise ValueError(f"title_token_limit 必须为正数：{limit}")
    words = clean_cell(text).split()
    return " ".join(words[:limit])


def parse_title_token_limit(variant: str, default: int = 8) -> int:
    match = re.fullmatch(r"title_trunc_(\d+)_zero_delta", variant)
    if not match:
        return default
    return int(match.group(1))


def compose_variant_text(
    row: dict[str, object],
    variant: str,
    title_token_limit: int = 8,
) -> str:
    if variant == "A2_original_warm_delta":
        raise ValueError("A2_original_warm_delta 不需要构造 content text。")

    fields: tuple[str, ...]
    title_override: str | None = None
    if variant == "full_content_zero_delta":
        fields = METADATA_COLUMNS
    elif variant.startswith("title_trunc_") and variant.endswith("_zero_delta"):
        fields = METADATA_COLUMNS
        title_override = truncate_words(clean_cell(row.get("title")), title_token_limit)
    elif variant == "no_title_zero_delta":
        fields = tuple(column for column in METADATA_COLUMNS if column != "title")
    else:
        raise ValueError(f"不支持的 degraded variant：{variant}")

    parts = []
    for column in fields:
        value = title_override if column == "title" and title_override is not None else clean_cell(row.get(column))
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


def read_warm_products(
    products_path: Path,
    warm_item2index: dict[str, int],
    target_item_ids: set[int],
    locale: str,
    chunksize: int = 200_000,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    required_columns = ["id", "locale", *METADATA_COLUMNS]
    header = pd.read_csv(products_path, nrows=0).columns.tolist()
    missing = sorted(set(required_columns) - set(header))
    if missing:
        raise ValueError(f"products_train.csv 缺少字段：{missing}")

    item_id_to_raw = {int(item_id): raw_id for raw_id, item_id in warm_item2index.items()}
    keep_raw_ids = {item_id_to_raw[item_id] for item_id in target_item_ids if item_id in item_id_to_raw}
    if len(keep_raw_ids) != len(target_item_ids):
        missing_ids = sorted(target_item_ids - set(item_id_to_raw))
        raise ValueError(f"部分 warm target item_id 找不到 raw id：{missing_ids[:10]}")

    products_by_id: dict[str, dict[str, str]] = {}
    total_rows = 0
    locale_rows = 0
    matched_rows = 0

    chunks = pd.read_csv(
        products_path,
        usecols=required_columns,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in chunks:
        total_rows += len(chunk)
        locale_chunk = chunk[chunk["locale"] == locale]
        locale_rows += len(locale_chunk)
        matched = locale_chunk[locale_chunk["id"].isin(keep_raw_ids)]
        matched_rows += len(matched)
        for record in matched.to_dict("records"):
            product_id = str(record["id"])
            # 中文注释：如果同一 FR 商品 id 出现重复行，保留文件中更靠后的那一行；
            # 这与现有 Amazon-M2 字段分组脚本的处理口径一致。
            products_by_id[product_id] = {key: clean_cell(value) for key, value in record.items()}

    rows = []
    for item_id in sorted(target_item_ids):
        raw_item_id = item_id_to_raw[item_id]
        product = products_by_id.get(raw_item_id)
        metadata_found = product is not None
        product = product or {}
        row: dict[str, object] = {
            "raw_item_id": raw_item_id,
            "item_id": int(item_id),
            "position_in_embedding_file": int(item_id - 1),
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
        "products_locale_rows": locale_rows,
        "matched_rows": matched_rows,
        "matched_unique_ids": len(products_by_id),
        "target_items": len(target_item_ids),
        "missing_metadata": sum(1 for row in rows if not row["metadata_found"]),
        "locale": locale,
    }
    return rows, summary


def build_warm_ground_truth(
    ground_truth: pl.DataFrame,
    max_eval_users: int = 0,
    max_target_items: int = 0,
) -> pl.DataFrame:
    warm_gt = ground_truth.filter(~pl.col("is_cold")).select(["user_id", "item_id", "is_cold"])
    if max_target_items > 0:
        keep_items = warm_gt.get_column("item_id").unique(maintain_order=True).head(max_target_items)
        warm_gt = warm_gt.filter(pl.col("item_id").is_in(keep_items.to_list()))
    if max_eval_users > 0:
        keep_users = warm_gt.get_column("user_id").unique(maintain_order=True).head(max_eval_users)
        warm_gt = warm_gt.filter(pl.col("user_id").is_in(keep_users.to_list()))
    return warm_gt


def build_target_ground_truth(warm_gt: pl.DataFrame, target_rows: list[dict[str, object]]) -> pd.DataFrame:
    field_profile = pd.DataFrame(
        [
            {
                "item_id": row["item_id"],
                "raw_item_id": row["raw_item_id"],
                "field_group": row["field_group"],
                "present_field_count": row["present_field_count"],
                "metadata_found": row["metadata_found"],
            }
            for row in target_rows
        ]
    )
    warm_gt_pd = warm_gt.to_pandas()
    return warm_gt_pd.merge(field_profile, on="item_id", how="left")


def encode_target_variant(
    sentence_model,
    target_rows: list[dict[str, object]],
    variant: str,
    batch_size: int,
) -> np.ndarray:
    title_token_limit = parse_title_token_limit(variant)
    texts = [
        compose_variant_text(row, variant, title_token_limit=title_token_limit)
        for row in target_rows
    ]
    print(f"Encoding variant={variant}, target_items={len(texts)}")
    embeddings = sentence_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_projected_base_embeddings(checkpoint_path: Path, warm_embeddings_path: Path, cold_embeddings_path: Path):
    manager = load_embedding_manager(checkpoint_path)
    warm_raw = np.load(warm_embeddings_path)
    cold_raw = np.load(cold_embeddings_path)
    # 中文注释：必须复用 A2 checkpoint 保存的 embedding_manager.pkl。
    # 这里不能重新 fit PCA，否则 degraded target 会进入不同坐标系。
    warm = manager.transform(warm_raw).astype(np.float32, copy=False)
    cold = manager.transform(cold_raw).astype(np.float32, copy=False)
    return manager, warm, cold


def build_a2_model(num_items: int, embedding_dim: int):
    from source.winter.recommender import SASRecModelWithTrainableDelta

    return SASRecModelWithTrainableDelta(
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_blocks=2,
        num_heads=1,
        intermediate_dim=embedding_dim,
        p=0.3,
        max_length=64,
        max_delta_norm=0.5,
    )


def append_cold_embeddings_zero_delta(model, cold_embeddings: torch.Tensor) -> None:
    item_embeddings = model.item_embedding.weight[: model.num_items + 1]
    delta_embeddings = model.delta_embedding.weight[: model.num_items + 1]
    model.set_pretrained_item_embeddings(
        item_embeddings=torch.vstack([item_embeddings, cold_embeddings.to(item_embeddings.device)]),
        delta_embeddings=torch.vstack(
            [delta_embeddings, torch.zeros_like(cold_embeddings).to(delta_embeddings.device)]
        ),
        add_padding_embedding=False,
        freeze=True,
    )


def replace_target_content_only(
    model,
    target_item_ids: list[int],
    target_embeddings: np.ndarray,
) -> None:
    if len(target_item_ids) != target_embeddings.shape[0]:
        raise ValueError(
            f"target_item_ids 数量与 target_embeddings 不一致："
            f"{len(target_item_ids)} vs {target_embeddings.shape[0]}"
        )

    with torch.no_grad():
        item_weight = model.item_embedding.weight
        delta_weight = model.delta_embedding.weight
        for row_idx, item_id in enumerate(target_item_ids):
            # 中文注释：warm item_id 在模型里就是 1..num_warm，0 是 padding。
            # 这里只替换目标 item，自身 delta 置零，模拟 content-only pseudo-cold target。
            item_weight[item_id] = torch.tensor(target_embeddings[row_idx], dtype=item_weight.dtype)
            delta_weight[item_id].zero_()


def load_a2_recommender(
    checkpoint_path: Path,
    warm_embeddings: np.ndarray,
    cold_embeddings: np.ndarray,
    topk: int,
):
    from source.winter.recommender import ColdStartSequentialRecommender

    warm_tensor = torch.tensor(warm_embeddings).float()
    cold_tensor = torch.tensor(cold_embeddings).float()

    model = build_a2_model(num_items=warm_embeddings.shape[0], embedding_dim=warm_embeddings.shape[1])
    model.set_pretrained_item_embeddings(
        warm_tensor.clone(),
        add_padding_embedding=True,
        freeze=True,
    )
    recommender = ColdStartSequentialRecommender.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        remove_seen=True,
        metrics=["NDCG", "Recall"],
        topk=topk,
        map_location="cpu",
    )
    append_cold_embeddings_zero_delta(recommender.model, cold_tensor)
    recommender.recommend_cold_items = True
    recommender.eval()
    return recommender


def predict_recommendations(
    recommender,
    test_interactions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: str,
) -> pd.DataFrame:
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
    return recommendations.to_pandas()


def filter_test_to_users(test_interactions: pl.DataFrame, user_ids: list[Any]) -> pl.DataFrame:
    return test_interactions.filter(pl.col("user_id").is_in(user_ids))


def build_hit_detail(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, topk: int) -> pd.DataFrame:
    required_rec = {"user_id", "item_id", "rating"}
    required_gt = {"user_id", "item_id", "field_group"}
    missing_rec = sorted(required_rec - set(recommendations.columns))
    missing_gt = sorted(required_gt - set(ground_truth.columns))
    if missing_rec:
        raise ValueError(f"recommendations 缺少列：{missing_rec}")
    if missing_gt:
        raise ValueError(f"ground_truth 缺少列：{missing_gt}")

    recommendations = recommendations.sort_values(
        ["user_id", "rating", "item_id"],
        ascending=[True, False, False],
    ).copy()
    recommendations["rank"] = recommendations.groupby("user_id").cumcount() + 1
    recommendations = recommendations[recommendations["rank"] <= topk]

    detail = ground_truth.merge(
        recommendations[["user_id", "item_id", "rank", "rating"]],
        on=["user_id", "item_id"],
        how="left",
    )
    detail["hit"] = detail["rank"].notna()
    detail["rank"] = detail["rank"].astype("Int64")
    detail[f"recall_contribution@{topk}"] = detail["hit"].astype(float)
    detail[f"ndcg_contribution@{topk}"] = detail["rank"].map(
        lambda rank: 0.0 if pd.isna(rank) else 1.0 / math.log2(float(rank) + 1.0)
    )
    return detail


def summarize_metrics(variant: str, hit_detail: pd.DataFrame, topk: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name in ["all", *FIELD_GROUP_ORDER]:
        group = hit_detail if group_name == "all" else hit_detail[hit_detail["field_group"] == group_name]
        rows.append(
            {
                "variant": variant,
                "field_group": group_name,
                "ground_truth_rows": int(len(group)),
                "target_items": int(group["item_id"].nunique()) if len(group) else 0,
                "hit_rows": int(group["hit"].sum()) if len(group) else 0,
                f"NDCG@{topk}": float(group[f"ndcg_contribution@{topk}"].mean()) if len(group) else 0.0,
                f"Recall@{topk}": float(group[f"recall_contribution@{topk}"].mean()) if len(group) else 0.0,
                "mean_hit_rank": float(group.loc[group["hit"], "rank"].mean())
                if len(group) and group["hit"].any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_item_concentration(variant: str, hit_detail: pd.DataFrame, topk: int) -> pd.DataFrame:
    item_summary = (
        hit_detail.groupby(["field_group", "item_id"], dropna=False)
        .agg(
            raw_item_id=("raw_item_id", "first"),
            gt_rows=("user_id", "size"),
            hit_rows=("hit", "sum"),
            ndcg_sum=(f"ndcg_contribution@{topk}", "sum"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )
    item_summary["variant"] = variant
    item_summary["hit_rate"] = item_summary["hit_rows"] / item_summary["gt_rows"].replace(0, np.nan)
    return item_summary.sort_values(
        ["variant", "field_group", "hit_rows", "ndcg_sum"],
        ascending=[True, True, False, False],
    )


def summarize_variant_deltas(summary: pd.DataFrame, topk: int) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "full_content_zero_delta"][
        ["field_group", f"NDCG@{topk}", f"Recall@{topk}"]
    ].rename(
        columns={
            f"NDCG@{topk}": f"baseline_NDCG@{topk}",
            f"Recall@{topk}": f"baseline_Recall@{topk}",
        }
    )
    merged = summary.merge(baseline, on="field_group", how="left")
    merged[f"delta_NDCG@{topk}"] = merged[f"NDCG@{topk}"] - merged[f"baseline_NDCG@{topk}"]
    merged[f"delta_Recall@{topk}"] = merged[f"Recall@{topk}"] - merged[f"baseline_Recall@{topk}"]
    return merged


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants)
    paths = resolve_paths(args.data_root, args.products_path, args.output_dir)
    require_files(
        paths,
        [
            "products",
            "test",
            "ground_truth",
            "item2index_warm",
            "warm_embeddings",
            "cold_embeddings",
        ],
    )
    checkpoint_path = require_file(args.a2_checkpoint, "A2 checkpoint")

    import_letitgo_runtime()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    ground_truth = pl.read_parquet(paths["ground_truth"])
    test_interactions = pl.read_parquet(paths["test"])
    warm_item2index = normalize_item2index(load_pickle(paths["item2index_warm"]), "item2index_warm")
    warm_gt = build_warm_ground_truth(
        ground_truth,
        max_eval_users=args.max_eval_users,
        max_target_items=args.max_target_items,
    )
    target_item_ids = set(int(value) for value in warm_gt.get_column("item_id").unique().to_list())
    target_rows, product_summary = read_warm_products(
        paths["products"],
        warm_item2index,
        target_item_ids,
        locale=args.locale,
    )
    target_gt = build_target_ground_truth(warm_gt, target_rows)
    target_user_ids = target_gt["user_id"].unique().tolist()
    test_subset = filter_test_to_users(test_interactions, target_user_ids)

    print("Amazon-M2 degraded-view sanity check")
    print(f"data_root: {paths['data_root']}")
    print(f"products: {paths['products']}")
    print(f"output_dir: {paths['output_dir']}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"variants: {variants}")
    print(f"warm_gt_rows: {len(warm_gt)}")
    print(f"target_items: {len(target_rows)}")
    print(f"test_users: {len(target_user_ids)}")
    print(f"product_summary: {product_summary}")

    target_gt.to_csv(paths["output_dir"] / "target_ground_truth_profile.csv", index=False)
    pd.DataFrame(target_rows).to_csv(paths["output_dir"] / "target_item_profile.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths["data_root"]),
        "products": str(paths["products"]),
        "output_dir": str(paths["output_dir"]),
        "a2_checkpoint": str(checkpoint_path),
        "natural_metrics": str(args.natural_metrics),
        "variants": variants,
        "locale": args.locale,
        "topk": args.topk,
        "max_eval_users": args.max_eval_users,
        "max_target_items": args.max_target_items,
        "warm_gt_rows": len(warm_gt),
        "target_items": len(target_rows),
        "test_users": len(target_user_ids),
        "product_summary": product_summary,
    }
    write_manifest(paths["output_dir"], manifest)

    if args.check_only:
        print("CHECK_ONLY: 输入、target items 和 warm ground-truth 检查完成；未加载 E5，未跑预测。")
        return

    _, warm_projected, cold_projected = load_projected_base_embeddings(
        checkpoint_path,
        paths["warm_embeddings"],
        paths["cold_embeddings"],
    )
    target_item_ids_ordered = [int(row["item_id"]) for row in target_rows]

    variant_embeddings: dict[str, np.ndarray | None] = {"A2_original_warm_delta": None}
    encode_variants = [variant for variant in variants if variant != "A2_original_warm_delta"]
    if encode_variants:
        from sentence_transformers import SentenceTransformer

        print(f"Loading SentenceTransformer: {args.sentence_checkpoint}")
        sentence_model = SentenceTransformer(args.sentence_checkpoint)
        manager = load_embedding_manager(checkpoint_path)
        for variant in encode_variants:
            raw_embeddings = encode_target_variant(
                sentence_model,
                target_rows,
                variant,
                batch_size=args.encode_batch_size,
            )
            variant_embeddings[variant] = manager.transform(raw_embeddings).astype(np.float32, copy=False)

    overall_summaries: list[pd.DataFrame] = []
    hit_details: list[pd.DataFrame] = []
    item_summaries: list[pd.DataFrame] = []

    for run_idx, variant in enumerate(variants, start=1):
        print(f"\n===== [{run_idx}/{len(variants)}] variant={variant} START =====")
        recommender = load_a2_recommender(
            checkpoint_path=checkpoint_path,
            warm_embeddings=warm_projected,
            cold_embeddings=cold_projected,
            topk=args.topk,
        )
        target_embeddings = variant_embeddings[variant]
        if target_embeddings is not None:
            replace_target_content_only(
                recommender.model,
                target_item_ids_ordered,
                target_embeddings,
            )
        recommendations = predict_recommendations(
            recommender=recommender,
            test_interactions=test_subset,
            ground_truth=warm_gt,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        detail = build_hit_detail(recommendations, target_gt, topk=args.topk)
        detail["variant"] = variant
        summary = summarize_metrics(variant, detail, topk=args.topk)
        item_summary = summarize_item_concentration(variant, detail, topk=args.topk)

        detail.to_csv(paths["output_dir"] / f"hit_detail_{variant}.csv", index=False)
        hit_details.append(detail)
        overall_summaries.append(summary)
        item_summaries.append(item_summary)
        print(f"===== [{run_idx}/{len(variants)}] variant={variant} DONE =====")

    summary = pd.concat(overall_summaries, ignore_index=True)
    deltas = summarize_variant_deltas(summary, topk=args.topk)
    item_summary_all = pd.concat(item_summaries, ignore_index=True)
    hit_detail_all = pd.concat(hit_details, ignore_index=True)

    summary.to_csv(paths["output_dir"] / "degraded_view_metrics.csv", index=False)
    deltas.to_csv(paths["output_dir"] / "degraded_view_metric_deltas.csv", index=False)
    item_summary_all.to_csv(paths["output_dir"] / "degraded_view_item_concentration.csv", index=False)
    hit_detail_all.to_csv(paths["output_dir"] / "degraded_view_hit_detail_all.csv", index=False)

    if args.natural_metrics.expanduser().is_file():
        natural = pd.read_csv(args.natural_metrics)
        natural[natural["model"] == "A2"].to_csv(
            paths["output_dir"] / "natural_group_metrics_A2_reference.csv",
            index=False,
        )

    print("DONE: degraded-view sanity check outputs 已生成。")


if __name__ == "__main__":
    main()
