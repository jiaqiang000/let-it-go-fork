"""Amazon-M2 cold item embedding 邻域难度诊断。

这个脚本不训练模型、不重新预测，只回答一个前置问题：
表现差的 cold item，是否在 content/PCA 空间里本来就更难找到相近 warm item。

如果这个现象成立，后续才有理由考虑基于 warm-neighborhood 的 cold delta 生成、
邻域可靠性校准或 weak-evidence robustness 方法。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent
TEMP_ROOT = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260605"
DATA_ROOT = RESEARCH_ROOT / "letitgo-data" / "data" / "amazon_m2_fr"

DEFAULT_A2_RUN_DIR = (
    TEMP_ROOT
    / "自然字段完整度分组评测"
    / "amazon_m2_A1A2_20260605"
    / "offline-09a0f19af9b04b908ef51015948abb8b"
)
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "embedding邻域难度诊断"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether cold item hit difficulty is related to warm-neighbor similarity."
    )
    parser.add_argument(
        "--warm-embeddings",
        type=Path,
        default=DATA_ROOT / "item_embeddings" / "embeddings_warm.npy",
        help="作者原始 warm item content embeddings。",
    )
    parser.add_argument(
        "--cold-embeddings",
        type=Path,
        default=DATA_ROOT / "item_embeddings" / "embeddings_cold.npy",
        help="作者原始 cold item content embeddings。",
    )
    parser.add_argument(
        "--embedding-manager",
        type=Path,
        default=DEFAULT_A2_RUN_DIR / "embedding_manager.pkl",
        help="A2 checkpoint 旁边保存的 PCA/Normalizer。",
    )
    parser.add_argument(
        "--hit-detail",
        type=Path,
        default=TEMP_ROOT / "自然分组命中明细诊断" / "hit_detail_A2.csv",
        help="A2 cold ground-truth 命中明细。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="诊断结果输出目录。",
    )
    parser.add_argument(
        "--topk",
        default="1,5,10,20,50",
        help="用于统计 nearest warm neighbors 的 k 值，逗号分隔。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="计算 cold-warm 相似度时的 cold batch size。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查输入文件，不计算邻域。",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{label}: {path}")
    return path


def parse_topk(value: str) -> list[int]:
    topk = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not topk or topk[0] <= 0:
        raise ValueError(f"topk 必须是正整数列表：{value}")
    return topk


def import_letitgo_runtime() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_projected_embeddings(
    warm_embeddings_path: Path,
    cold_embeddings_path: Path,
    embedding_manager_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import_letitgo_runtime()
    from source.embedding_manager import EmbeddingManager

    manager = EmbeddingManager.load(str(embedding_manager_path))
    warm_raw = np.load(warm_embeddings_path)
    cold_raw = np.load(cold_embeddings_path)

    # 中文注释：这里必须复用 A2 训练时保存的 PCA/Normalizer，
    # 诊断空间才和 A2 推荐结果所在的 item embedding 空间一致。
    warm = manager.transform(warm_raw).astype(np.float32, copy=False)
    cold = manager.transform(cold_raw).astype(np.float32, copy=False)
    return l2_normalize(warm), l2_normalize(cold)


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def compute_neighborhood_features(
    warm: np.ndarray,
    cold: np.ndarray,
    topk: list[int],
    batch_size: int,
) -> pd.DataFrame:
    max_k = max(topk)
    if max_k > warm.shape[0]:
        raise ValueError(f"max(topk)={max_k} 大于 warm item 数量 {warm.shape[0]}")

    warm_item_ids = np.arange(1, warm.shape[0] + 1, dtype=np.int64)
    cold_item_ids = np.arange(warm.shape[0] + 1, warm.shape[0] + cold.shape[0] + 1, dtype=np.int64)
    rows: list[dict[str, float | int]] = []

    for start in range(0, cold.shape[0], batch_size):
        end = min(start + batch_size, cold.shape[0])
        similarities = cold[start:end] @ warm.T

        # 中文注释：只取 top-k warm 邻居，避免保存完整 cold-warm 相似度矩阵。
        candidate_idx = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
        candidate_values = np.take_along_axis(similarities, candidate_idx, axis=1)
        order = np.argsort(-candidate_values, axis=1)
        top_indices = np.take_along_axis(candidate_idx, order, axis=1)
        top_values = np.take_along_axis(candidate_values, order, axis=1)

        for local_row in range(end - start):
            item_id = int(cold_item_ids[start + local_row])
            sims = top_values[local_row]
            neighbor_ids = warm_item_ids[top_indices[local_row]]
            row: dict[str, float | int] = {
                "item_id": item_id,
                "nearest_warm_item_id": int(neighbor_ids[0]),
                "nearest_warm_cosine": float(sims[0]),
            }
            for k in topk:
                values = sims[:k]
                row[f"top{k}_mean_cosine"] = float(values.mean())
                row[f"top{k}_min_cosine"] = float(values.min())
            rows.append(row)

    return pd.DataFrame(rows)


def build_item_hit_summary(hit_detail: pd.DataFrame) -> pd.DataFrame:
    required = {"item_id", "field_group", "hit", "recall_contribution@10", "ndcg_contribution@10"}
    missing = sorted(required - set(hit_detail.columns))
    if missing:
        raise ValueError(f"hit_detail 缺少列：{missing}")

    grouped = (
        hit_detail.groupby(["item_id", "field_group"], dropna=False)
        .agg(
            gt_rows=("user_id", "size"),
            hit_rows=("hit", "sum"),
            item_recall_at10=("recall_contribution@10", "mean"),
            item_ndcg_at10=("ndcg_contribution@10", "mean"),
            raw_item_id=("raw_item_id", "first") if "raw_item_id" in hit_detail.columns else ("item_id", "first"),
            title_len_bucket=("title_len_bucket", "first") if "title_len_bucket" in hit_detail.columns else ("item_id", "first"),
            desc_len_bucket=("desc_len_bucket", "first") if "desc_len_bucket" in hit_detail.columns else ("item_id", "first"),
        )
        .reset_index()
    )
    return grouped


def summarize_by_group(item_table: pd.DataFrame, topk: list[int]) -> pd.DataFrame:
    rows = []
    for group_name, group in item_table.groupby("field_group", dropna=False):
        row: dict[str, float | int | str] = {
            "field_group": group_name,
            "cold_gt_items": int(group["item_id"].nunique()),
            "cold_ground_truth_rows": int(group["gt_rows"].sum()),
            "hit_rows": int(group["hit_rows"].sum()),
            "item_recall_at10": float(group["item_recall_at10"].mean()),
            "item_ndcg_at10": float(group["item_ndcg_at10"].mean()),
        }
        for k in topk:
            row[f"top{k}_mean_cosine_mean"] = float(group[f"top{k}_mean_cosine"].mean())
            row[f"top{k}_mean_cosine_std"] = float(group[f"top{k}_mean_cosine"].std(ddof=0))
            row[f"top{k}_min_cosine_mean"] = float(group[f"top{k}_min_cosine"].mean())
        rows.append(row)
    return round_numeric(pd.DataFrame(rows).sort_values("field_group"))


def summarize_hit_vs_miss(row_table: pd.DataFrame, topk: list[int]) -> pd.DataFrame:
    rows = []
    for keys, group in row_table.groupby(["field_group", "hit"], dropna=False):
        field_group, hit = keys
        row: dict[str, float | int | str | bool] = {
            "field_group": field_group,
            "hit": bool(hit),
            "cold_ground_truth_rows": int(len(group)),
            "cold_gt_items": int(group["item_id"].nunique()),
            "cold_NDCG_at10": float(group["ndcg_contribution@10"].mean()),
            "cold_Recall_at10": float(group["recall_contribution@10"].mean()),
        }
        for k in topk:
            row[f"top{k}_mean_cosine_mean"] = float(group[f"top{k}_mean_cosine"].mean())
            row[f"top{k}_min_cosine_mean"] = float(group[f"top{k}_min_cosine"].mean())
        rows.append(row)
    return round_numeric(pd.DataFrame(rows).sort_values(["field_group", "hit"]))


def summarize_correlations(item_table: pd.DataFrame, topk: list[int]) -> pd.DataFrame:
    metric_columns = ["nearest_warm_cosine"] + [f"top{k}_mean_cosine" for k in topk]
    target_columns = ["item_recall_at10", "item_ndcg_at10"]
    rows = []
    groups = [("all", item_table)] + list(item_table.groupby("field_group", dropna=False))

    for group_name, group in groups:
        for metric in metric_columns:
            for target in target_columns:
                if len(group) < 3 or group[metric].nunique() < 2 or group[target].nunique() < 2:
                    pearson = np.nan
                    spearman = np.nan
                else:
                    pearson = group[[metric, target]].corr(method="pearson").iloc[0, 1]
                    spearman = group[[metric, target]].corr(method="spearman").iloc[0, 1]
                rows.append(
                    {
                        "field_group": group_name,
                        "neighbor_metric": metric,
                        "target": target,
                        "items": int(len(group)),
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )
    return round_numeric(pd.DataFrame(rows))


def summarize_neighbor_buckets(item_table: pd.DataFrame) -> pd.DataFrame:
    table = item_table.copy()
    metric = "top10_mean_cosine" if "top10_mean_cosine" in table.columns else "nearest_warm_cosine"
    table["neighbor_bucket"] = pd.qcut(
        table[metric],
        q=3,
        labels=["low_neighbor", "mid_neighbor", "high_neighbor"],
        duplicates="drop",
    )
    rows = []
    for bucket, group in table.groupby("neighbor_bucket", observed=True):
        row = {
            "neighbor_bucket": str(bucket),
            "cold_gt_items": int(group["item_id"].nunique()),
            "cold_ground_truth_rows": int(group["gt_rows"].sum()),
            "hit_rows": int(group["hit_rows"].sum()),
            "item_recall_at10": float(group["item_recall_at10"].mean()),
            "item_ndcg_at10": float(group["item_ndcg_at10"].mean()),
            metric: float(group[metric].mean()),
        }
        for group_name in ["weak_0_1", "mid_2", "strong_3_4"]:
            row[f"share_{group_name}"] = float((group["field_group"] == group_name).mean())
        rows.append(row)
    return round_numeric(pd.DataFrame(rows))


def round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=["float", "float32", "float64"]).columns
    result[numeric_columns] = result[numeric_columns].round(6)
    return result


def main() -> None:
    args = parse_args()
    topk = parse_topk(args.topk)

    warm_embeddings = require_file(args.warm_embeddings, "warm_embeddings")
    cold_embeddings = require_file(args.cold_embeddings, "cold_embeddings")
    embedding_manager = require_file(args.embedding_manager, "embedding_manager")
    hit_detail_path = require_file(args.hit_detail, "hit_detail")
    output_dir = args.output_dir.expanduser().resolve()

    print("Amazon-M2 cold-neighborhood difficulty diagnostic")
    print("warm_embeddings:", warm_embeddings)
    print("cold_embeddings:", cold_embeddings)
    print("embedding_manager:", embedding_manager)
    print("hit_detail:", hit_detail_path)
    print("output_dir:", output_dir)
    print("topk:", topk)

    if args.check_only:
        print("CHECK_ONLY: 输入文件存在；未计算邻域。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    warm, cold = load_projected_embeddings(warm_embeddings, cold_embeddings, embedding_manager)
    print(f"projected warm shape={warm.shape}, cold shape={cold.shape}")

    neighborhood = compute_neighborhood_features(
        warm=warm,
        cold=cold,
        topk=topk,
        batch_size=args.batch_size,
    )
    hit_detail = pd.read_csv(hit_detail_path)
    item_hit = build_item_hit_summary(hit_detail)
    item_table = item_hit.merge(neighborhood, on="item_id", how="left")
    row_table = hit_detail.merge(neighborhood, on="item_id", how="left")

    if item_table["nearest_warm_cosine"].isna().any():
        missing = item_table.loc[item_table["nearest_warm_cosine"].isna(), "item_id"].head(10).tolist()
        raise ValueError(f"部分 hit item 没有匹配到 cold embedding 邻域特征，示例 item_id={missing}")

    field_group_summary = summarize_by_group(item_table, topk)
    hit_vs_miss_summary = summarize_hit_vs_miss(row_table, topk)
    correlations = summarize_correlations(item_table, topk)
    bucket_summary = summarize_neighbor_buckets(item_table)

    neighborhood.to_csv(output_dir / "cold_neighborhood_features.csv", index=False)
    item_table.to_csv(output_dir / "item_hit_with_neighborhood.csv", index=False)
    row_table.to_csv(output_dir / "hit_detail_with_neighborhood.csv", index=False)
    field_group_summary.to_csv(output_dir / "field_group_neighborhood_summary.csv", index=False)
    hit_vs_miss_summary.to_csv(output_dir / "hit_vs_miss_neighborhood_summary.csv", index=False)
    correlations.to_csv(output_dir / "neighborhood_performance_correlation.csv", index=False)
    bucket_summary.to_csv(output_dir / "neighbor_bucket_summary.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warm_embeddings": str(warm_embeddings),
        "cold_embeddings": str(cold_embeddings),
        "embedding_manager": str(embedding_manager),
        "hit_detail": str(hit_detail_path),
        "output_dir": str(output_dir),
        "topk": topk,
        "batch_size": args.batch_size,
        "warm_shape": list(warm.shape),
        "cold_shape": list(cold.shape),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("field_group_neighborhood_summary:")
    print(field_group_summary.to_string(index=False))
    print()
    print("neighbor_bucket_summary:")
    print(bucket_summary.to_string(index=False))
    print("DONE: cold-neighborhood difficulty diagnostic 已输出。")


if __name__ == "__main__":
    main()
