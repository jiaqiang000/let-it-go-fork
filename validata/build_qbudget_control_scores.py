"""为 Beauty q-budget diagnostic 实验生成隔离的 q 控制文件。

本脚本只读取原始 Beauty quality score，并把 current / shuffle / reverse-rank
等控制版本写到 outputs/qbudget_controls/ 下。它不修改训练代码，也不覆盖
quality_score_output/beauty/ 下的原始 q 文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "quality_score_output" / "beauty"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "qbudget_controls" / "beauty"
DEFAULT_MIN_BUDGET = 0.3
DEFAULT_MAX_BUDGET = 0.6
DEFAULT_SHUFFLE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build isolated q-budget control files for Beauty diagnostics."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="包含 warm_quality.npy 和 cold_quality.npy 的原始 q 目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="隔离输出目录；默认写到 outputs/qbudget_controls/beauty。",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help="shuffle 控制组使用的随机种子。",
    )
    parser.add_argument(
        "--min-budget",
        type=float,
        default=DEFAULT_MIN_BUDGET,
        help="和 run.py 中 quality_score.min_budget 对齐。",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=DEFAULT_MAX_BUDGET,
        help="和 run.py 中 quality_score.max_budget 对齐。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖 output-dir 下本脚本生成的隔离控制文件。",
    )
    return parser.parse_args()


def load_quality_array(path: Path, split_name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} quality file not found: {path}")

    values = np.load(path)
    if values.ndim != 1:
        raise ValueError(f"{split_name} quality must be 1D, found shape {values.shape}")

    if np.any((values < -1e-6) | (values > 1.0 + 1e-6)):
        raise ValueError(f"{split_name} quality values must be within [0, 1]")

    return np.clip(values.astype(np.float32), 0.0, 1.0)


def build_budget(q_values: np.ndarray, min_budget: float, max_budget: float) -> np.ndarray:
    if min_budget < 0 or max_budget <= 0 or min_budget > max_budget:
        raise ValueError("budget must satisfy 0 <= min_budget <= max_budget")

    return min_budget + (1.0 - q_values) * (max_budget - min_budget)


def build_shuffle_q(q_values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = q_values.copy()

    # 保持 q 值分布不变，只随机打乱 q 和 warm item 的对应关系。
    # 目的：检验 q-item 对应关系本身是否携带有效信息。
    rng.shuffle(shuffled)
    return shuffled.astype(np.float32)


def build_reverse_rank_q(q_values: np.ndarray) -> np.ndarray:
    # 保持 q 值分布不变，只按原始 q 排名反向分配给 warm item。
    # 这是公平的 reverse-rank 对照；不要使用 quality_score.reverse_budget=True，
    # 因为那个开关会改变平均 budget，导致对照不公平。
    order = np.argsort(q_values, kind="mergesort")
    reversed_q = np.empty_like(q_values)
    reversed_q[order] = q_values[order[::-1]]
    return reversed_q.astype(np.float32)


def stats_for_variant(
    variant_name: str,
    q_values: np.ndarray,
    reference_budget: np.ndarray,
    min_budget: float,
    max_budget: float,
) -> dict[str, float | int | str]:
    budget = build_budget(q_values, min_budget, max_budget)
    sorted_budget_diff = float(
        np.max(np.abs(np.sort(budget) - np.sort(reference_budget)))
    )

    return {
        "variant": variant_name,
        "num_warm_items": int(q_values.shape[0]),
        "q_min": float(q_values.min()),
        "q_mean": float(q_values.mean()),
        "q_max": float(q_values.max()),
        "q_std": float(q_values.std()),
        "budget_min": float(budget.min()),
        "budget_mean": float(budget.mean()),
        "budget_rms": float(np.sqrt(np.mean(np.square(budget)))),
        "budget_max": float(budget.max()),
        "budget_std": float(budget.std()),
        "sorted_budget_max_abs_diff_vs_current": sorted_budget_diff,
    }


def prepare_variant_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"output variant dir already exists: {path}. Use --force to overwrite."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=False)


def write_variant(
    output_dir: Path,
    variant_name: str,
    warm_q: np.ndarray,
    cold_q: np.ndarray,
    stats: dict[str, float | int | str],
    force: bool,
) -> None:
    variant_dir = output_dir / variant_name
    prepare_variant_dir(variant_dir, force=force)

    np.save(variant_dir / "warm_quality.npy", warm_q.astype(np.float32))
    np.save(variant_dir / "cold_quality.npy", cold_q.astype(np.float32))

    with (variant_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def write_summary(output_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    summary_path = output_dir / "qbudget_summary.csv"
    fieldnames = list(rows[0].keys())

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    warm_q = load_quality_array(input_dir / "warm_quality.npy", "warm")
    cold_q = load_quality_array(input_dir / "cold_quality.npy", "cold")
    current_budget = build_budget(warm_q, args.min_budget, args.max_budget)

    variants = {
        "current": warm_q.copy(),
        f"shuffle_seed{args.shuffle_seed}": build_shuffle_q(warm_q, args.shuffle_seed),
        "reverse_rank": build_reverse_rank_q(warm_q),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for variant_name, variant_warm_q in variants.items():
        stats = stats_for_variant(
            variant_name=variant_name,
            q_values=variant_warm_q,
            reference_budget=current_budget,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
        )
        write_variant(
            output_dir=output_dir,
            variant_name=variant_name,
            warm_q=variant_warm_q,
            cold_q=cold_q,
            stats=stats,
            force=args.force,
        )
        summary_rows.append(stats)

    write_summary(output_dir, summary_rows)

    print("input dir:", input_dir)
    print("output dir:", output_dir)
    print("warm items:", warm_q.shape[0])
    print("cold items:", cold_q.shape[0])
    print("budget mean:", round(float(current_budget.mean()), 6))
    print("budget rms:", round(float(np.sqrt(np.mean(np.square(current_budget)))), 6))
    print("variants:", ", ".join(variants))


if __name__ == "__main__":
    main()
