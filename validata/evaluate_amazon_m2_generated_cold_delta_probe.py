"""Amazon-M2 generated cold delta 最小评测探针。

这个脚本不训练模型、不修改 scripts/run.py。它固定 Let It Go A2 checkpoint，
只在评测阶段把 cold item 的 zero delta 替换成由 warm content 邻居生成的非零 delta。

核心问题：
1. warm delta 的邻域平滑性迁移到 cold item 后，整体 cold 推荐指标是否改善？
2. 这种改善是否能接回 weak/mid/strong 分组，而不是只改善 strong item？
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
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

TEMP_ROOT = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260605"
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "cold-delta-generation-probe"

LOCAL_A2_RUN_DIR = (
    TEMP_ROOT
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

FIELD_NAMES = ("color", "size", "model", "material")
METADATA_COLUMNS = ("title", "brand", "color", "size", "model", "material", "author")
FIELD_GROUP_ORDER = ("weak_0_1", "mid_2", "strong_3_4", "missing_metadata")
MISSING_STRINGS = {"", "null", "none", "nan", "[]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated cold delta probe on Amazon-M2 A2 checkpoint."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a2-checkpoint", type=Path, default=DEFAULT_A2_CHECKPOINT)
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--neighbor-topk", default="1,5,10")
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--metric-topk", type=int, default=10)
    parser.add_argument("--neighbor-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=0,
        help="本地 smoke test 用；0 表示评测全部 ground-truth users。",
    )
    parser.add_argument(
        "--sample-cold-gt-users",
        action="store_true",
        help="配合 --max-eval-users 使用；优先抽真实答案为 cold item 的用户。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查路径、字段分组、embedding 形状和 checkpoint，不跑预测。",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise ValueError(f"必须提供正整数列表：{value}")
    return values


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise ValueError(f"必须提供正数列表：{value}")
    return values


def resolve_paths(data_root: Path, products_path: Path, output_dir: Path) -> dict[str, Path]:
    data_root = data_root.expanduser().resolve()
    return {
        "data_root": data_root,
        "products": products_path.expanduser().resolve(),
        "output_dir": output_dir.expanduser().resolve(),
        "test": data_root / "processed" / "test_interactions.parquet",
        "ground_truth": data_root / "processed" / "ground_truth.parquet",
        "item2index_cold": data_root / "processed" / "item2index_cold.pkl",
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


def assign_field_group(present_count: int, metadata_found: bool = True) -> str:
    if not metadata_found:
        return "missing_metadata"
    if present_count <= 1:
        return "weak_0_1"
    if present_count == 2:
        return "mid_2"
    return "strong_3_4"


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
        # 这里必须过滤 FR，才能和作者 amazon_m2_fr 的 embedding/交互预处理口径一致。
        matched = chunk[(chunk["locale"] == locale) & (chunk["id"].isin(keep_ids))]
        matched_rows += len(matched)
        for record in matched.to_dict("records"):
            product_id = str(record["id"])
            # 中文注释：和 evaluate_amazon_m2_field_groups.py 的 dict comprehension 口径对齐；
            # 如果同一 FR 商品 id 出现重复行，保留文件中更靠后的那一行。
            products_by_id[product_id] = {key: clean_cell(value) for key, value in record.items()}

    rows = []
    for position, (raw_item_id, model_item_id) in enumerate(
        sorted(cold_item2index.items(), key=lambda pair: pair[1])
    ):
        product = products_by_id.get(raw_item_id)
        metadata_found = product is not None
        product = product or {}
        row: dict[str, object] = {
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


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def safe_row_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, eps)


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


def load_projected_embeddings(
    checkpoint_path: Path,
    warm_embeddings_path: Path,
    cold_embeddings_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    manager = load_embedding_manager(checkpoint_path)
    warm_raw = np.load(warm_embeddings_path)
    cold_raw = np.load(cold_embeddings_path)

    # 中文注释：必须复用 A2 checkpoint 旁边的 PCA/Normalizer。
    # 不能重新 fit PCA，否则 generated cold delta 会被接到另一个坐标系。
    warm = manager.transform(warm_raw).astype(np.float32, copy=False)
    cold = manager.transform(cold_raw).astype(np.float32, copy=False)
    return warm, cold


def load_warm_delta(checkpoint_path: Path, expected_warm_items: int) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    key = "model.delta_embedding.weight"
    if key not in state_dict:
        available = [name for name in state_dict if "delta" in name.lower()]
        raise KeyError(f"checkpoint 中找不到 {key}，可用 delta keys={available}")

    delta = state_dict[key].detach().cpu().float().numpy()
    if delta.shape[0] != expected_warm_items + 1:
        raise ValueError(
            "delta_embedding 行数和 warm item 数不一致："
            f"delta_shape={delta.shape}, expected_warm_items={expected_warm_items}"
        )

    # 中文注释：第 0 行是 padding，真实 warm item_id 是 1..num_warm。
    return delta[1 : expected_warm_items + 1].astype(np.float32, copy=False)


def generate_neighbor_delta(
    warm_content: np.ndarray,
    cold_content: np.ndarray,
    warm_delta: np.ndarray,
    topk: int,
    alpha: float,
    batch_size: int = 256,
) -> tuple[np.ndarray, pd.DataFrame]:
    warm_content = l2_normalize(warm_content)
    cold_content = l2_normalize(cold_content)
    warm_delta = np.asarray(warm_delta, dtype=np.float32)

    if topk <= 0 or topk > warm_content.shape[0]:
        raise ValueError(f"topk={topk} 不合法，warm_items={warm_content.shape[0]}")
    if alpha <= 0:
        raise ValueError(f"alpha 必须为正数：{alpha}")
    if warm_delta.shape != warm_content.shape:
        raise ValueError(
            f"warm_delta shape 必须等于 warm_content shape："
            f"{warm_delta.shape} vs {warm_content.shape}"
        )

    warm_item_ids = np.arange(1, warm_content.shape[0] + 1, dtype=np.int64)
    cold_item_ids = np.arange(
        warm_content.shape[0] + 1,
        warm_content.shape[0] + cold_content.shape[0] + 1,
        dtype=np.int64,
    )

    generated = np.zeros((cold_content.shape[0], warm_delta.shape[1]), dtype=np.float32)
    rows: list[dict[str, float | int]] = []

    for start in range(0, cold_content.shape[0], batch_size):
        end = min(start + batch_size, cold_content.shape[0])
        similarities = cold_content[start:end] @ warm_content.T
        candidate_idx = np.argpartition(-similarities, kth=topk - 1, axis=1)[:, :topk]
        candidate_values = np.take_along_axis(similarities, candidate_idx, axis=1)
        order = np.argsort(-candidate_values, axis=1)
        neighbor_idx = np.take_along_axis(candidate_idx, order, axis=1)
        neighbor_sim = np.take_along_axis(candidate_values, order, axis=1)

        for local_row in range(end - start):
            item_position = start + local_row
            idx = neighbor_idx[local_row]
            sims = neighbor_sim[local_row]
            weights = np.maximum(sims, 0.0)

            # 中文注释：只让非负 content cosine 参与加权。
            # 如果权重退化为 0，就回退到简单均值，避免 NaN。
            if float(weights.sum()) <= 1e-12:
                candidate_delta = warm_delta[idx].mean(axis=0)
            else:
                candidate_delta = np.average(warm_delta[idx], axis=0, weights=weights)

            generated_delta = alpha * candidate_delta
            generated[item_position] = generated_delta.astype(np.float32, copy=False)
            neighbor_ids = warm_item_ids[idx]
            rows.append(
                {
                    "item_id": int(cold_item_ids[item_position]),
                    "topk": int(topk),
                    "alpha": float(alpha),
                    "nearest_warm_item_id": int(neighbor_ids[0]),
                    "nearest_warm_cosine": float(sims[0]),
                    "neighbor_cosine_mean": float(sims.mean()),
                    "neighbor_cosine_min": float(sims.min()),
                    "candidate_delta_norm": float(np.linalg.norm(candidate_delta)),
                    "generated_delta_norm": float(np.linalg.norm(generated_delta)),
                }
            )

    return generated, pd.DataFrame(rows)


def summarize_generated_delta(
    group: str,
    topk: int,
    alpha: float,
    cold_content: np.ndarray,
    generated_delta: np.ndarray,
) -> dict[str, float | int | str]:
    cold_content = np.asarray(cold_content, dtype=np.float32)
    generated_delta = np.asarray(generated_delta, dtype=np.float32)
    final_content = cold_content + generated_delta
    delta_norm = np.linalg.norm(generated_delta, axis=1)
    content_final_cosine = safe_row_cosine(cold_content, final_content)

    return {
        "group": group,
        "topk": int(topk),
        "alpha": float(alpha),
        "cold_items": int(generated_delta.shape[0]),
        "delta_norm_mean": float(delta_norm.mean()),
        "delta_norm_median": float(np.median(delta_norm)),
        "delta_norm_p95": float(np.quantile(delta_norm, 0.95)),
        "delta_norm_max": float(delta_norm.max()),
        "cold_content_final_cosine_mean": float(content_final_cosine.mean()),
        "cold_content_final_cosine_min": float(content_final_cosine.min()),
    }


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


def append_cold_item_embeddings_with_delta(
    model,
    cold_embeddings: torch.Tensor,
    cold_delta: torch.Tensor,
) -> None:
    if cold_embeddings.shape != cold_delta.shape:
        raise ValueError(
            f"cold_embeddings 和 cold_delta shape 必须一致："
            f"{tuple(cold_embeddings.shape)} vs {tuple(cold_delta.shape)}"
        )

    item_embeddings = model.item_embedding.weight[: model.num_items + 1]
    delta_embeddings = model.delta_embedding.weight[: model.num_items + 1]
    model.set_pretrained_item_embeddings(
        item_embeddings=torch.vstack(
            [item_embeddings, cold_embeddings.to(item_embeddings.device)]
        ),
        delta_embeddings=torch.vstack(
            [delta_embeddings, cold_delta.to(delta_embeddings.device)]
        ),
        add_padding_embedding=False,
        freeze=True,
    )


def load_a2_recommender(
    checkpoint_path: Path,
    warm_embeddings: np.ndarray,
    cold_embeddings: np.ndarray,
    cold_delta: np.ndarray,
    metric_topk: int,
):
    from source.winter.recommender import ColdStartSequentialRecommender

    warm_tensor = torch.tensor(warm_embeddings).float()
    cold_tensor = torch.tensor(cold_embeddings).float()
    cold_delta_tensor = torch.tensor(cold_delta).float()

    model = build_a2_model(
        num_items=warm_embeddings.shape[0],
        embedding_dim=warm_embeddings.shape[1],
    )
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
        topk=metric_topk,
        map_location="cpu",
    )
    append_cold_item_embeddings_with_delta(recommender.model, cold_tensor, cold_delta_tensor)
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

    # 中文注释：本地 smoke test 优先抽 cold GT 用户，避免抽样后 cold 指标全为 0。
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


def compute_overall_metrics(
    group: str,
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    metric_topk: int,
) -> pl.DataFrame:
    from source.winter.evaluation.metrics import ColdStartOfflineMetrics

    evaluator = ColdStartOfflineMetrics(metrics=["NDCG", "Recall"], topk=metric_topk)
    metrics = evaluator(recommendations, ground_truth)
    row = {
        "group": group,
        "recommend-cold-items": True,
        "filter-cold-items": False,
    }
    for key, value in metrics.items():
        row[key] = float(value)
    return pl.from_dicts([row])


def compute_group_metrics(
    group: str,
    recommendations: pl.DataFrame,
    ground_truth_groups: pl.DataFrame,
    metric_topk: int,
) -> pl.DataFrame:
    from source.winter.evaluation.metrics import ColdStartOfflineMetrics

    evaluator = ColdStartOfflineMetrics(metrics=["NDCG", "Recall"], topk=metric_topk)
    rows = []
    for field_group in FIELD_GROUP_ORDER:
        group_gt = ground_truth_groups.filter(pl.col("field_group") == field_group)
        if len(group_gt) == 0:
            metrics = {
                f"cold_NDCG@{metric_topk}": 0.0,
                f"cold_Recall@{metric_topk}": 0.0,
            }
        else:
            group_predictions = recommendations.filter(
                pl.col("user_id").is_in(group_gt.get_column("user_id"))
            )
            metrics = evaluator(group_predictions, group_gt)

        rows.append(
            {
                "group": group,
                "field_group": field_group,
                "cold_ground_truth_rows": len(group_gt),
                "gt_user_id_count": group_gt.get_column("user_id").n_unique() if len(group_gt) else 0,
                f"cold_NDCG@{metric_topk}": float(metrics[f"cold_NDCG@{metric_topk}"]),
                f"cold_Recall@{metric_topk}": float(metrics[f"cold_Recall@{metric_topk}"]),
            }
        )
    return pl.from_dicts(rows)


def build_key_rows(
    overall_metrics: pl.DataFrame,
    group_metrics: pl.DataFrame,
    metric_topk: int,
) -> pl.DataFrame:
    weak = group_metrics.filter(pl.col("field_group") == "weak_0_1").select(
        [
            "group",
            pl.col(f"cold_NDCG@{metric_topk}").alias(f"weak_cold_NDCG@{metric_topk}"),
            pl.col(f"cold_Recall@{metric_topk}").alias(f"weak_cold_Recall@{metric_topk}"),
        ]
    )
    strong = group_metrics.filter(pl.col("field_group") == "strong_3_4").select(
        [
            "group",
            pl.col(f"cold_NDCG@{metric_topk}").alias(f"strong_cold_NDCG@{metric_topk}"),
            pl.col(f"cold_Recall@{metric_topk}").alias(f"strong_cold_Recall@{metric_topk}"),
        ]
    )
    return (
        overall_metrics.join(weak, on="group", how="left")
        .join(strong, on="group", how="left")
        .sort(f"cold_NDCG@{metric_topk}", descending=True)
    )


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
            "item2index_cold",
            "warm_embeddings",
            "cold_embeddings",
        ],
    )
    checkpoint_path = require_file(args.a2_checkpoint, "A2 checkpoint")
    topk_values = parse_int_list(args.neighbor_topk)
    alpha_values = parse_float_list(args.alphas)

    import_letitgo_runtime()
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cold_item2index = normalize_item2index(load_pickle(paths["item2index_cold"]), "item2index_cold")
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

    warm_embeddings, cold_embeddings = load_projected_embeddings(
        checkpoint_path=checkpoint_path,
        warm_embeddings_path=paths["warm_embeddings"],
        cold_embeddings_path=paths["cold_embeddings"],
    )
    warm_delta = load_warm_delta(checkpoint_path, expected_warm_items=warm_embeddings.shape[0])

    count_table.write_csv(output_dir / "field_group_counts.csv")
    pl.from_dicts(cold_rows).write_csv(output_dir / "field_profile_cold_items.csv")

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths["data_root"]),
        "products_path": str(paths["products"]),
        "output_dir": str(output_dir),
        "a2_checkpoint": str(checkpoint_path),
        "embedding_manager": str(checkpoint_path.parent.parent / "embedding_manager.pkl"),
        "locale": args.locale,
        "neighbor_topk": topk_values,
        "alphas": alpha_values,
        "metric_topk": args.metric_topk,
        "max_eval_users": args.max_eval_users,
        "sample_cold_gt_users": args.sample_cold_gt_users,
        "recommend_cold_items": True,
        "filter_cold_items": False,
        "check_only": args.check_only,
        "product_summary": product_summary,
        "warm_embedding_shape": list(warm_embeddings.shape),
        "cold_embedding_shape": list(cold_embeddings.shape),
        "warm_delta_shape": list(warm_delta.shape),
    }

    print("Amazon-M2 generated cold delta probe")
    print("data_root:", paths["data_root"])
    print("products:", paths["products"])
    print("output_dir:", output_dir)
    print("a2_checkpoint:", checkpoint_path)
    print("warm_embedding_shape:", warm_embeddings.shape)
    print("cold_embedding_shape:", cold_embeddings.shape)
    print("warm_delta_shape:", warm_delta.shape)
    print(count_table)

    if args.check_only:
        write_manifest(output_dir, manifest)
        print("CHECK_ONLY: 输入、字段分组、embedding 和 delta 形状检查完成；未跑预测。")
        return

    runs: list[tuple[str, int, float, np.ndarray, pd.DataFrame | None]] = [
        (
            "A2_original_zero_delta",
            0,
            0.0,
            np.zeros_like(cold_embeddings, dtype=np.float32),
            None,
        )
    ]
    summary_rows = [
        summarize_generated_delta(
            group="A2_original_zero_delta",
            topk=0,
            alpha=0.0,
            cold_content=cold_embeddings,
            generated_delta=np.zeros_like(cold_embeddings, dtype=np.float32),
        )
    ]
    neighbor_detail_tables = []
    total_generated_runs = len(topk_values) * len(alpha_values)
    generated_run_index = 0

    for neighbor_topk in topk_values:
        for alpha in alpha_values:
            generated_run_index += 1
            group = f"generated_top{neighbor_topk}_alpha{alpha:g}"
            start_time = time.perf_counter()
            print(
                f"===== [generate {generated_run_index}/{total_generated_runs}] "
                f"group={group} START =====",
                flush=True,
            )
            generated_delta, details = generate_neighbor_delta(
                warm_content=warm_embeddings,
                cold_content=cold_embeddings,
                warm_delta=warm_delta,
                topk=neighbor_topk,
                alpha=alpha,
                batch_size=args.neighbor_batch_size,
            )
            runs.append((group, neighbor_topk, alpha, generated_delta, details))
            summary_rows.append(
                summarize_generated_delta(
                    group=group,
                    topk=neighbor_topk,
                    alpha=alpha,
                    cold_content=cold_embeddings,
                    generated_delta=generated_delta,
                )
            )
            neighbor_detail_tables.append(pl.from_pandas(details).with_columns(pl.lit(group).alias("group")))
            elapsed = time.perf_counter() - start_time
            print(
                f"===== [generate {generated_run_index}/{total_generated_runs}] "
                f"group={group} DONE, elapsed={elapsed:.2f}s =====",
                flush=True,
            )

    generated_delta_summary = pl.from_dicts(summary_rows)
    generated_delta_summary.write_csv(output_dir / "generated_delta_summary.csv")
    if neighbor_detail_tables:
        pl.concat(neighbor_detail_tables).write_csv(output_dir / "generated_delta_neighbor_details.csv")

    overall_tables = []
    group_tables = []
    total_runs = len(runs)
    for run_index, (group, neighbor_topk, alpha, cold_delta, _details) in enumerate(runs, start=1):
        run_start_time = time.perf_counter()
        print()
        print(f"===== [{run_index}/{total_runs}] group={group} START =====")
        print("neighbor_topk:", neighbor_topk, "alpha:", alpha)
        recommender = load_a2_recommender(
            checkpoint_path=checkpoint_path,
            warm_embeddings=warm_embeddings,
            cold_embeddings=cold_embeddings,
            cold_delta=cold_delta,
            metric_topk=args.metric_topk,
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
        overall = compute_overall_metrics(group, recommendations, ground_truth, args.metric_topk)
        field_group = compute_group_metrics(
            group,
            recommendations,
            ground_truth_groups,
            args.metric_topk,
        )
        overall_tables.append(overall)
        group_tables.append(field_group)
        print(overall)
        print(field_group)
        run_elapsed = time.perf_counter() - run_start_time
        print(f"===== [{run_index}/{total_runs}] group={group} DONE, elapsed={run_elapsed:.2f}s =====")

    overall_metrics = pl.concat(overall_tables)
    field_group_metrics = pl.concat(group_tables)
    key_rows = build_key_rows(overall_metrics, field_group_metrics, args.metric_topk)

    overall_metrics.write_csv(output_dir / "probe_overall_metrics.csv")
    field_group_metrics.write_csv(output_dir / "probe_field_group_metrics.csv")
    key_rows.write_csv(output_dir / "probe_key_rows.csv")
    write_manifest(output_dir, manifest)
    print("DONE: probe_overall_metrics.csv / probe_field_group_metrics.csv / probe_key_rows.csv 已输出。")


if __name__ == "__main__":
    main()
