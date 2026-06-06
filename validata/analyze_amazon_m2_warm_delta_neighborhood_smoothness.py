"""Amazon-M2 warm item learned delta 邻域平滑性诊断。

这个脚本不训练模型、不重新预测，只检查 Let It Go A2 学到的 warm delta
是否能被 content embedding 邻域解释。

核心问题：
1. 内容相近的 warm item，它们的 delta 方向是否比随机 warm item 更一致？
2. 用近邻 warm item 的 delta 均值，能否预测目标 warm item 自己的 delta？

如果这两点不成立，后续直接给 cold item 做 nearest-warm delta transfer 就缺少基础。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "warm-delta邻域平滑性诊断"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze warm delta local smoothness in Amazon-M2 A2 content space."
    )
    parser.add_argument(
        "--warm-embeddings",
        type=Path,
        default=DATA_ROOT / "item_embeddings" / "embeddings_warm.npy",
        help="作者原始 warm item content embeddings。",
    )
    parser.add_argument(
        "--embedding-manager",
        type=Path,
        default=DEFAULT_A2_RUN_DIR / "embedding_manager.pkl",
        help="A2 checkpoint 旁边保存的 PCA/Normalizer。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_A2_RUN_DIR / "recommender" / "epoch=9-step=7350.ckpt",
        help="A2 recommender checkpoint。",
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
        help="用于统计 warm-neighbor delta 的 k 值，逗号分隔。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="计算 warm-warm 相似度时的 batch size。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机邻居 baseline 的随机种子。",
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


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_projected_warm_embeddings(
    warm_embeddings_path: Path,
    embedding_manager_path: Path,
) -> np.ndarray:
    import_letitgo_runtime()
    from source.embedding_manager import EmbeddingManager

    manager = EmbeddingManager.load(str(embedding_manager_path))
    warm_raw = np.load(warm_embeddings_path)

    # 中文注释：必须复用 A2 训练时保存的 PCA/Normalizer。
    # 这样 warm 内容邻域才和 A2 learned delta 所在 item embedding 空间一致。
    warm = manager.transform(warm_raw).astype(np.float32, copy=False)
    return l2_normalize(warm)


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


def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, eps)


def sample_random_neighbors(
    rng: np.random.Generator,
    num_items: int,
    max_k: int,
) -> np.ndarray:
    # 中文注释：随机 baseline 只需要排除自身，不要求无放回；
    # 这样可以快速构造和真实邻居同样形状的随机对照。
    random_idx = rng.integers(0, num_items - 1, size=(num_items, max_k), dtype=np.int64)
    item_idx = np.arange(num_items, dtype=np.int64)[:, None]
    random_idx = random_idx + (random_idx >= item_idx)
    return random_idx


def analyze_delta_smoothness(
    warm_content: np.ndarray,
    warm_delta: np.ndarray,
    topk: list[int],
    batch_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_items = warm_content.shape[0]
    max_k = max(topk)
    if max_k >= num_items:
        raise ValueError(f"max(topk)={max_k} 必须小于 warm item 数量 {num_items}")

    delta_norm = np.linalg.norm(warm_delta, axis=1)
    delta_direction = l2_normalize(warm_delta)
    rng = np.random.default_rng(random_seed)
    random_neighbors = sample_random_neighbors(rng, num_items, max_k)

    rows: list[dict[str, float | int]] = []
    item_ids = np.arange(1, num_items + 1, dtype=np.int64)

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        similarities = warm_content[start:end] @ warm_content.T
        similarities[np.arange(end - start), np.arange(start, end)] = -np.inf

        candidate_idx = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
        candidate_values = np.take_along_axis(similarities, candidate_idx, axis=1)
        order = np.argsort(-candidate_values, axis=1)
        neighbor_idx = np.take_along_axis(candidate_idx, order, axis=1)
        neighbor_sim = np.take_along_axis(candidate_values, order, axis=1)

        for local_row in range(end - start):
            global_idx = start + local_row
            target_delta = warm_delta[global_idx]
            target_delta_direction = delta_direction[global_idx]
            target_delta_norm = delta_norm[global_idx]
            actual_idx = neighbor_idx[local_row]
            actual_sim = neighbor_sim[local_row]
            random_idx = random_neighbors[global_idx]

            row: dict[str, float | int] = {
                "item_id": int(item_ids[global_idx]),
                "delta_norm": float(target_delta_norm),
                "nearest_warm_item_id": int(item_ids[actual_idx[0]]),
                "nearest_content_cosine": float(actual_sim[0]),
            }

            for k in topk:
                actual_neighbors = actual_idx[:k]
                random_neighbors_k = random_idx[:k]
                actual_neighbor_delta = warm_delta[actual_neighbors]
                random_neighbor_delta = warm_delta[random_neighbors_k]
                actual_neighbor_direction = delta_direction[actual_neighbors]
                random_neighbor_direction = delta_direction[random_neighbors_k]

                actual_delta_cos = actual_neighbor_direction @ target_delta_direction
                random_delta_cos = random_neighbor_direction @ target_delta_direction

                actual_pred = actual_neighbor_delta.mean(axis=0)
                random_pred = random_neighbor_delta.mean(axis=0)

                # 中文注释：加权预测只用非负 content 相似度做权重；
                # 如果权重退化，就回退到均值，避免单个异常值破坏诊断。
                weights = np.maximum(actual_sim[:k], 0.0)
                if float(weights.sum()) <= 1e-12:
                    weighted_pred = actual_pred
                else:
                    weighted_pred = np.average(actual_neighbor_delta, axis=0, weights=weights)

                row[f"top{k}_content_cosine_mean"] = float(actual_sim[:k].mean())
                row[f"top{k}_actual_delta_cosine_mean"] = float(actual_delta_cos.mean())
                row[f"top{k}_random_delta_cosine_mean"] = float(random_delta_cos.mean())
                row[f"top{k}_actual_pred_delta_cosine"] = float(
                    safe_cosine(target_delta[None, :], actual_pred[None, :])[0]
                )
                row[f"top{k}_weighted_pred_delta_cosine"] = float(
                    safe_cosine(target_delta[None, :], weighted_pred[None, :])[0]
                )
                row[f"top{k}_random_pred_delta_cosine"] = float(
                    safe_cosine(target_delta[None, :], random_pred[None, :])[0]
                )
                row[f"top{k}_actual_pred_relative_l2"] = float(
                    np.linalg.norm(target_delta - actual_pred) / max(target_delta_norm, 1e-12)
                )
                row[f"top{k}_random_pred_relative_l2"] = float(
                    np.linalg.norm(target_delta - random_pred) / max(target_delta_norm, 1e-12)
                )
                row[f"top{k}_actual_neighbor_delta_norm_mean"] = float(
                    delta_norm[actual_neighbors].mean()
                )
                row[f"top{k}_random_neighbor_delta_norm_mean"] = float(
                    delta_norm[random_neighbors_k].mean()
                )

            rows.append(row)

    per_item = pd.DataFrame(rows)
    summary = summarize_smoothness(per_item, topk)
    norm_summary = summarize_delta_norms(delta_norm)
    correlation_summary = summarize_correlations(per_item, topk)
    return per_item, summary, norm_summary, correlation_summary


def summarize_smoothness(per_item: pd.DataFrame, topk: list[int]) -> pd.DataFrame:
    rows = []
    for k in topk:
        actual_delta_cos = per_item[f"top{k}_actual_delta_cosine_mean"]
        random_delta_cos = per_item[f"top{k}_random_delta_cosine_mean"]
        actual_pred_cos = per_item[f"top{k}_actual_pred_delta_cosine"]
        weighted_pred_cos = per_item[f"top{k}_weighted_pred_delta_cosine"]
        random_pred_cos = per_item[f"top{k}_random_pred_delta_cosine"]
        actual_l2 = per_item[f"top{k}_actual_pred_relative_l2"]
        random_l2 = per_item[f"top{k}_random_pred_relative_l2"]

        rows.append(
            {
                "topk": k,
                "items": int(len(per_item)),
                "content_cosine_mean": float(per_item[f"top{k}_content_cosine_mean"].mean()),
                "actual_neighbor_delta_cosine_mean": float(actual_delta_cos.mean()),
                "random_neighbor_delta_cosine_mean": float(random_delta_cos.mean()),
                "neighbor_delta_cosine_lift": float(actual_delta_cos.mean() - random_delta_cos.mean()),
                "actual_pred_delta_cosine_mean": float(actual_pred_cos.mean()),
                "weighted_pred_delta_cosine_mean": float(weighted_pred_cos.mean()),
                "random_pred_delta_cosine_mean": float(random_pred_cos.mean()),
                "pred_delta_cosine_lift": float(actual_pred_cos.mean() - random_pred_cos.mean()),
                "weighted_pred_delta_cosine_lift": float(weighted_pred_cos.mean() - random_pred_cos.mean()),
                "actual_pred_relative_l2_mean": float(actual_l2.mean()),
                "random_pred_relative_l2_mean": float(random_l2.mean()),
                "pred_relative_l2_reduction": float(random_l2.mean() - actual_l2.mean()),
                "actual_pred_better_than_random_share": float((actual_pred_cos > random_pred_cos).mean()),
                "weighted_pred_better_than_random_share": float((weighted_pred_cos > random_pred_cos).mean()),
            }
        )
    return round_numeric(pd.DataFrame(rows))


def summarize_delta_norms(delta_norm: np.ndarray) -> pd.DataFrame:
    return round_numeric(
        pd.DataFrame(
            [
                {
                    "items": int(len(delta_norm)),
                    "delta_norm_mean": float(np.mean(delta_norm)),
                    "delta_norm_std": float(np.std(delta_norm)),
                    "delta_norm_min": float(np.min(delta_norm)),
                    "delta_norm_p25": float(np.quantile(delta_norm, 0.25)),
                    "delta_norm_median": float(np.median(delta_norm)),
                    "delta_norm_p75": float(np.quantile(delta_norm, 0.75)),
                    "delta_norm_p95": float(np.quantile(delta_norm, 0.95)),
                    "delta_norm_max": float(np.max(delta_norm)),
                    "share_norm_gt_0_5": float(np.mean(delta_norm > 0.5 + 1e-6)),
                }
            ]
        )
    )


def summarize_correlations(per_item: pd.DataFrame, topk: list[int]) -> pd.DataFrame:
    rows = []
    for k in topk:
        pairs = [
            (f"top{k}_content_cosine_mean", f"top{k}_actual_delta_cosine_mean"),
            (f"top{k}_content_cosine_mean", f"top{k}_actual_pred_delta_cosine"),
            ("delta_norm", f"top{k}_actual_pred_delta_cosine"),
            ("delta_norm", f"top{k}_actual_pred_relative_l2"),
        ]
        for left, right in pairs:
            rows.append(
                {
                    "topk": k,
                    "left_metric": left,
                    "right_metric": right,
                    "pearson": float(per_item[[left, right]].corr(method="pearson").iloc[0, 1]),
                    "spearman": float(per_item[[left, right]].corr(method="spearman").iloc[0, 1]),
                }
            )
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
    embedding_manager = require_file(args.embedding_manager, "embedding_manager")
    checkpoint = require_file(args.checkpoint, "checkpoint")
    output_dir = args.output_dir.expanduser().resolve()

    print("Amazon-M2 warm-delta neighborhood smoothness diagnostic")
    print("warm_embeddings:", warm_embeddings)
    print("embedding_manager:", embedding_manager)
    print("checkpoint:", checkpoint)
    print("output_dir:", output_dir)
    print("topk:", topk)
    print("random_seed:", args.random_seed)

    if args.check_only:
        print("CHECK_ONLY: 输入文件存在；未计算 warm delta 邻域。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    warm_content = load_projected_warm_embeddings(warm_embeddings, embedding_manager)
    warm_delta = load_warm_delta(checkpoint, expected_warm_items=warm_content.shape[0])
    print(f"warm_content_shape={warm_content.shape}, warm_delta_shape={warm_delta.shape}")

    per_item, smoothness_summary, norm_summary, correlation_summary = analyze_delta_smoothness(
        warm_content=warm_content,
        warm_delta=warm_delta,
        topk=topk,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
    )

    per_item.to_csv(output_dir / "warm_delta_neighborhood_per_item.csv", index=False)
    smoothness_summary.to_csv(output_dir / "warm_delta_smoothness_summary.csv", index=False)
    norm_summary.to_csv(output_dir / "warm_delta_norm_summary.csv", index=False)
    correlation_summary.to_csv(output_dir / "warm_delta_neighborhood_correlation.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warm_embeddings": str(warm_embeddings),
        "embedding_manager": str(embedding_manager),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "topk": topk,
        "batch_size": args.batch_size,
        "random_seed": args.random_seed,
        "warm_content_shape": list(warm_content.shape),
        "warm_delta_shape": list(warm_delta.shape),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("warm_delta_norm_summary:")
    print(norm_summary.to_string(index=False))
    print()
    print("warm_delta_smoothness_summary:")
    print(smoothness_summary.to_string(index=False))
    print("DONE: warm-delta neighborhood smoothness diagnostic 已输出。")


if __name__ == "__main__":
    main()
