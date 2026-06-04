"""Amazon-M2 字段完整度分组评测。

本脚本只做诊断评测，不训练模型，也不修改 run.py / source/ 主逻辑。
核心目的：检查 color/size/model/material 完整度不同的 cold item，在 A1/A2
checkpoint 下的 cold-start 推荐效果是否不同。
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "letitgo-data" / "data" / "amazon_m2_fr"
SERVER_DATA_ROOT = Path("/root/letitgo-data/data/amazon_m2_fr")
DEFAULT_DATA_ROOT = SERVER_DATA_ROOT if SERVER_DATA_ROOT.exists() else LOCAL_DATA_ROOT
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "amazon_m2_field_group_eval"

DEFAULT_A1_CHECKPOINT = Path(
    "/hy-tmp/letitgo_ckpt/amazon_m2_A0A1_20260605/"
    "offline-d2c7d67301fa451b894b5862cf5cacdd/recommender/epoch=8-step=6615.ckpt"
)
DEFAULT_A2_CHECKPOINT = Path(
    "/hy-tmp/letitgo_ckpt/amazon_m2_baseline_20260605/"
    "offline-3ec73ef18570400a8f690c7de74b8ccd/recommender/epoch=6-step=5145.ckpt"
)

FIELD_NAMES = ("color", "size", "model", "material")
FIELD_GROUP_ORDER = ("weak_0_1", "mid_2", "strong_3_4", "missing_metadata")
MISSING_STRINGS = {"", "null", "none", "nan", "[]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Amazon-M2 cold items by metadata field-completeness groups."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a1-checkpoint", type=Path, default=DEFAULT_A1_CHECKPOINT)
    parser.add_argument("--a2-checkpoint", type=Path, default=DEFAULT_A2_CHECKPOINT)
    parser.add_argument(
        "--models",
        default="A1,A2",
        help="要评测的模型，逗号分隔；当前支持 A1,A2。",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查数据路径、字段分组和样本数，不加载 checkpoint，不跑预测。",
    )
    parser.add_argument(
        "--no-save-recommendations",
        action="store_true",
        help="不保存 A1/A2 的 top-k 推荐明细，只保存分组指标。",
    )
    return parser.parse_args()


def is_present_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, float) and np.isnan(value):
        return False

    text = str(value).strip()
    return text.lower() not in MISSING_STRINGS


def count_present_fields(row: dict[str, Any], fields: list[str] | tuple[str, ...]) -> int:
    return sum(1 for field in fields if is_present_value(row.get(field)))


def assign_field_group(present_count: int, metadata_found: bool = True) -> str:
    if not metadata_found:
        return "missing_metadata"

    if present_count <= 1:
        return "weak_0_1"
    if present_count == 2:
        return "mid_2"
    return "strong_3_4"


def resolve_paths(data_root: Path, products_path: Path, output_dir: Path) -> dict[str, Path]:
    data_root = data_root.expanduser().resolve()
    products_path = products_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    return {
        "data_root": data_root,
        "products": products_path,
        "output_dir": output_dir,
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


def build_field_profile(
    products: pl.DataFrame,
    cold_item2index: dict[str, int],
    fields: tuple[str, ...] = FIELD_NAMES,
) -> pl.DataFrame:
    required = {"id", *fields}
    missing_columns = sorted(required - set(products.columns))
    if missing_columns:
        raise ValueError(f"products_train.csv 缺少字段：{missing_columns}")

    product_rows = {
        row["id"]: row for row in products.select(["id", *fields]).iter_rows(named=True)
    }

    rows = []
    for raw_item_id, model_item_id in cold_item2index.items():
        raw_row = product_rows.get(raw_item_id)
        metadata_found = raw_row is not None
        raw_row = raw_row or {}
        present_count = count_present_fields(raw_row, fields)

        output_row = {
            "raw_item_id": raw_item_id,
            "item_id": model_item_id,
            "metadata_found": metadata_found,
            "present_field_count": present_count,
            "field_group": assign_field_group(present_count, metadata_found=metadata_found),
        }

        # 中文注释：保留每个字段是否存在，后续结果异常时可以回查到底是哪类字段缺失。
        for field in fields:
            output_row[f"{field}_present"] = is_present_value(raw_row.get(field))

        rows.append(output_row)

    return pl.from_dicts(rows)


def build_ground_truth_groups(
    ground_truth: pl.DataFrame,
    field_profile: pl.DataFrame,
) -> pl.DataFrame:
    cold_ground_truth = ground_truth.filter(pl.col("is_cold"))
    grouped = cold_ground_truth.join(
        field_profile.select(["item_id", "field_group", "present_field_count"]),
        on="item_id",
        how="left",
    )
    return grouped.with_columns(
        [
            pl.col("field_group").fill_null("missing_metadata"),
            pl.col("present_field_count").fill_null(0).cast(pl.Int64),
        ]
    )


def build_count_table(field_profile: pl.DataFrame, ground_truth_groups: pl.DataFrame) -> pl.DataFrame:
    item_counts = (
        field_profile.group_by("field_group")
        .agg(pl.len().alias("cold_items"))
        .select(["field_group", "cold_items"])
    )
    gt_counts = (
        ground_truth_groups.group_by("field_group")
        .agg(
            [
                pl.len().alias("cold_ground_truth_rows"),
                pl.col("user_id").n_unique().alias("cold_users"),
            ]
        )
        .select(["field_group", "cold_ground_truth_rows", "cold_users"])
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
                pl.col("cold_users").cast(pl.Int64),
            ]
        )
    )


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


def transform_embeddings(
    checkpoint_path: Path,
    warm_embeddings_path: Path,
    cold_embeddings_path: Path,
):
    import torch

    manager = load_embedding_manager(checkpoint_path)
    warm_raw = np.load(warm_embeddings_path)
    cold_raw = np.load(cold_embeddings_path)

    # 中文注释：正式评测必须复用训练时保存的 PCA/Normalizer，
    # 这样追加 cold embedding 时才和 checkpoint 所在空间一致。
    warm = manager.transform(warm_raw)
    cold = manager.transform(cold_raw)
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
    warm_embeddings: torch.Tensor,
    cold_embeddings: torch.Tensor,
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
    return recommendations.with_columns(
        pl.col("user_id").cast(ground_truth.schema["user_id"])
    )


def compute_group_metrics(
    model_name: str,
    recommendations: pl.DataFrame,
    ground_truth_groups: pl.DataFrame,
    topk: int,
) -> pl.DataFrame:
    from source.winter.evaluation.metrics import ColdStartOfflineMetrics

    evaluator = ColdStartOfflineMetrics(metrics=["NDCG", "Recall"], topk=topk)
    rows = []

    for group_name in FIELD_GROUP_ORDER:
        group_gt = ground_truth_groups.filter(pl.col("field_group") == group_name)
        group_predictions = recommendations.filter(
            pl.col("user_id").is_in(group_gt.get_column("user_id"))
        )
        metrics = evaluator(group_predictions, group_gt)

        rows.append(
            {
                "model": model_name,
                "field_group": group_name,
                "cold_ground_truth_rows": len(group_gt),
                "cold_users": group_gt.get_column("user_id").n_unique() if len(group_gt) else 0,
                f"cold_NDCG@{topk}": float(metrics[f"cold_NDCG@{topk}"]),
                f"cold_Recall@{topk}": float(metrics[f"cold_Recall@{topk}"]),
            }
        )

    return pl.from_dicts(rows)


def write_gzip_csv(frame: pl.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        frame.write_csv(f)


def write_readme(output_dir: Path, count_table: pl.DataFrame, metrics: pl.DataFrame | None) -> None:
    lines = [
        "# Amazon-M2 字段完整度分组评测输出",
        "",
        "这组结果只用于诊断：看 color/size/model/material 字段完整度不同的 cold item，",
        "在 Let It Go 的 A1/A2 checkpoint 下是否有不同推荐表现。",
        "",
        "## 字段分组",
        "",
        "- weak_0_1：四个字段中存在 0 或 1 个",
        "- mid_2：四个字段中存在 2 个",
        "- strong_3_4：四个字段中存在 3 或 4 个",
        "- missing_metadata：raw metadata 中找不到该 cold item",
        "",
        "## 样本计数",
        "",
        count_table.write_csv(),
    ]

    if metrics is not None:
        lines.extend(
            [
                "",
                "## 分组指标",
                "",
                metrics.write_csv(),
            ]
        )

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_models(value: str) -> list[str]:
    models = [model.strip() for model in value.split(",") if model.strip()]
    unsupported = sorted(set(models) - {"A1", "A2"})
    if unsupported:
        raise ValueError(f"当前只支持 A1,A2，不支持：{unsupported}")
    return models


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

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    products = pl.read_csv(paths["products"])
    ground_truth = pl.read_parquet(paths["ground_truth"])
    cold_item2index = normalize_item2index(
        load_pickle(paths["item2index_cold"]),
        "item2index_cold",
    )
    warm_item2index = normalize_item2index(
        load_pickle(paths["item2index_warm"]),
        "item2index_warm",
    )

    field_profile = build_field_profile(products, cold_item2index)
    ground_truth_groups = build_ground_truth_groups(ground_truth, field_profile)
    count_table = build_count_table(field_profile, ground_truth_groups)

    field_profile.write_csv(output_dir / "field_profile_cold_items.csv")
    ground_truth_groups.write_csv(output_dir / "ground_truth_cold_with_field_group.csv")
    count_table.write_csv(output_dir / "field_group_counts.csv")

    print("Amazon-M2 field group evaluation")
    print("data_root:", paths["data_root"])
    print("products:", paths["products"])
    print("output_dir:", output_dir)
    print("warm_items:", len(warm_item2index))
    print("cold_items:", len(field_profile))
    print("cold_ground_truth_rows:", len(ground_truth_groups))
    print(count_table)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths["data_root"]),
        "products_path": str(paths["products"]),
        "output_dir": str(output_dir),
        "fields": list(FIELD_NAMES),
        "models": parse_models(args.models),
        "check_only": args.check_only,
        "recommend_cold_items": True,
        "filter_cold_items": False,
    }

    if args.check_only:
        write_readme(output_dir, count_table, metrics=None)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("CHECK_ONLY: 字段分组检查完成；未加载 checkpoint，未跑预测。")
        return

    checkpoints = {
        "A1": args.a1_checkpoint.expanduser().resolve(),
        "A2": args.a2_checkpoint.expanduser().resolve(),
    }
    for model_name in parse_models(args.models):
        if not checkpoints[model_name].is_file():
            raise FileNotFoundError(f"{model_name} checkpoint 不存在：{checkpoints[model_name]}")

    import_letitgo_runtime()
    test_interactions = pl.read_parquet(paths["test"])
    metric_tables = []

    for index, model_name in enumerate(parse_models(args.models), start=1):
        checkpoint_path = checkpoints[model_name]
        print()
        print(f"===== [{index}/{len(parse_models(args.models))}] {model_name} START =====")
        print("checkpoint:", checkpoint_path)

        warm_embeddings, cold_embeddings = transform_embeddings(
            checkpoint_path,
            paths["warm_embeddings"],
            paths["cold_embeddings"],
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
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
        )

        if not args.no_save_recommendations:
            write_gzip_csv(recommendations, output_dir / f"recommendations_{model_name}.csv.gz")

        model_metrics = compute_group_metrics(
            model_name=model_name,
            recommendations=recommendations,
            ground_truth_groups=ground_truth_groups,
            topk=args.topk,
        )
        metric_tables.append(model_metrics)
        print(model_metrics)
        print(f"===== [{index}/{len(parse_models(args.models))}] {model_name} DONE =====")

    metrics = pl.concat(metric_tables)
    metrics.write_csv(output_dir / "field_group_metrics.csv")

    manifest["checkpoints"] = {name: str(path) for name, path in checkpoints.items()}
    manifest["batch_size"] = args.batch_size
    manifest["num_workers"] = args.num_workers
    manifest["accelerator"] = args.accelerator
    manifest["devices"] = args.devices
    manifest["topk"] = args.topk
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(output_dir, count_table, metrics=metrics)
    print("DONE: field_group_metrics.csv 已输出。")


if __name__ == "__main__":
    main()
