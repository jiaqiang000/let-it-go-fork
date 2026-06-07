"""Amazon-M2 warm target delta-alpha 局部响应诊断。

本脚本用于检查 warm target item 对 learned delta 强度的局部响应。
它固定 Let It Go A2 checkpoint，不重新训练模型，只在评测阶段缩放目标
item 自己的 learned delta：

    final_embedding_i(alpha) = content_i + alpha * learned_delta_i

直观含义：如果某些 warm item 在 alpha 变小后反而排名更好，说明“delta 越满
越好”并不对所有 item 成立，存在 item-specific local response。但这个诊断
只发生在已经有真实 learned delta 的 warm target 上，不能直接推出 strict cold
item 也能从 warm/cold 插值或生成 delta 中受益。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch

import evaluate_amazon_m2_degraded_view_sanity_check as degraded


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent
TEMP_20260607 = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260607"
DEFAULT_OUTPUT_DIR = TEMP_20260607 / "warm-delta-alpha-response"

CAT_FEATURES = ["field_group", "brand_present", "author_present", "metadata_found"]
NUM_FEATURES = ["present_field_count", "title_len"]
SELECTION_FRACTIONS = [0.05, 0.10, 0.15, 0.20]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate warm target delta-alpha response.")
    parser.add_argument("--data-root", type=Path, default=degraded.DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=degraded.DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a2-checkpoint", type=Path, default=degraded.DEFAULT_A2_CHECKPOINT)
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-eval-users", type=int, default=0)
    parser.add_argument("--max-target-items", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("alphas cannot be empty.")
    if values[0] < 0:
        raise ValueError(f"alpha must be non-negative: {values}")
    return values


def scale_target_delta(model: Any, target_item_ids: list[int], alpha: float) -> None:
    with torch.no_grad():
        delta_weight = model.delta_embedding.weight
        original = delta_weight.detach().clone()
        for item_id in target_item_ids:
            delta_weight[item_id] = original[item_id] * float(alpha)


def alpha_name(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def build_feature_table(target_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in target_rows:
        title = degraded.clean_cell(row.get("title"))
        rows.append(
            {
                "item_id": int(row["item_id"]),
                "field_group": row["field_group"],
                "present_field_count": int(row["present_field_count"]),
                "title_len": len(title.split()) if title else 0,
                "brand_present": bool(row.get("brand_present")),
                "author_present": bool(row.get("author_present")),
                "metadata_found": bool(row.get("metadata_found")),
            }
        )
    return pd.DataFrame(rows)


def build_response_tables(hit_detail: pd.DataFrame, topk: int, baseline_variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_col = f"ndcg_contribution@{topk}"
    record_cols = ["user_id", "item_id", "variant", metric_col]
    record = hit_detail[record_cols].copy()
    baseline = record[record["variant"] == baseline_variant][["user_id", "item_id", metric_col]].rename(
        columns={metric_col: "baseline_ndcg"}
    )
    response = record.merge(baseline, on=["user_id", "item_id"], how="left")
    response["delta_ndcg"] = response[metric_col] - response["baseline_ndcg"]

    fixed_summary = (
        response.groupby("variant")
        .agg(
            records=("delta_ndcg", "size"),
            mean_ndcg=(metric_col, "mean"),
            mean_delta_ndcg=("delta_ndcg", "mean"),
            positive_rate=("delta_ndcg", lambda values: float((values > 0).mean())),
            negative_rate=("delta_ndcg", lambda values: float((values < 0).mean())),
        )
        .reset_index()
        .sort_values("mean_ndcg", ascending=False)
    )

    record_for_oracle = response.copy()
    record_for_oracle["is_baseline_variant"] = record_for_oracle["variant"] == baseline_variant
    best_record = (
        record_for_oracle.sort_values(
            ["user_id", "item_id", metric_col, "is_baseline_variant"],
            ascending=[True, True, False, False],
        )
        .groupby(["user_id", "item_id"])
        .head(1)
    )
    oracle_gain = best_record[metric_col] - best_record["baseline_ndcg"]
    oracle_summary = pd.DataFrame(
        [
            {
                "records": int(best_record.shape[0]),
                "baseline_variant": baseline_variant,
                "baseline_mean_ndcg": float(baseline["baseline_ndcg"].mean()),
                "oracle_mean_ndcg": float(best_record[metric_col].mean()),
                "oracle_mean_gain": float(oracle_gain.mean()),
                "positive_gain_rate": float((oracle_gain > EPS).mean()),
                "nonbaseline_chosen_rate": float((best_record["variant"] != baseline_variant).mean()),
                "oracle_gain_p50": float(oracle_gain.quantile(0.50)),
                "oracle_gain_p90": float(oracle_gain.quantile(0.90)),
                "oracle_gain_p95": float(oracle_gain.quantile(0.95)),
            }
        ]
    )

    item_response = (
        response.groupby(["item_id", "variant"])
        .agg(
            mean_ndcg=(metric_col, "mean"),
            baseline_ndcg=("baseline_ndcg", "mean"),
            delta_ndcg=("delta_ndcg", "mean"),
            records=("user_id", "size"),
        )
        .reset_index()
    )
    return response, fixed_summary, oracle_summary, item_response


def bool_to_string(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column].astype(str)
    return frame


def cross_validated_probabilities(
    data: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
    n_splits: int,
) -> dict[str, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    y = labels.astype(int).to_numpy()
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    splits = min(n_splits, positives, negatives)
    if splits < 2:
        return {}

    x = bool_to_string(data[CAT_FEATURES + NUM_FEATURES], CAT_FEATURES)
    transformer = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    models = {
        "logreg": make_pipeline(
            transformer,
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        ),
        "rf_depth4": make_pipeline(
            transformer,
            RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    }
    return {
        name: cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        for name, model in models.items()
    }


def selector_summary(
    item_response: pd.DataFrame,
    feature_table: pd.DataFrame,
    baseline_variant: str,
    random_state: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_for_oracle = item_response.copy()
    item_for_oracle["is_baseline_variant"] = item_for_oracle["variant"] == baseline_variant
    best_item = (
        item_for_oracle.sort_values(
            ["item_id", "mean_ndcg", "is_baseline_variant"],
            ascending=[True, False, False],
        )
        .groupby("item_id")
        .head(1)
        .rename(columns={"variant": "best_variant", "delta_ndcg": "best_delta_ndcg"})
    )
    per_item = feature_table.merge(
        best_item[["item_id", "best_variant", "best_delta_ndcg", "mean_ndcg", "baseline_ndcg", "records"]],
        on="item_id",
        how="inner",
    )
    per_item["oracle_positive"] = per_item["best_delta_ndcg"] > EPS

    rows: list[dict[str, object]] = []
    probabilities = cross_validated_probabilities(
        per_item,
        per_item["oracle_positive"],
        random_state=random_state,
        n_splits=n_splits,
    )
    for model_name, score in probabilities.items():
        order = np.argsort(score)[::-1]
        gains = per_item["best_delta_ndcg"].to_numpy(dtype=float)
        labels = per_item["oracle_positive"].to_numpy(dtype=bool)
        for fraction in SELECTION_FRACTIONS:
            k = max(1, int(round(len(per_item) * fraction)))
            selected = order[:k]
            rows.append(
                {
                    "model": model_name,
                    "fraction": fraction,
                    "selected_items": int(k),
                    "positive_rate": float(labels[selected].mean()),
                    "selected_mean_oracle_gain": float(gains[selected].mean()),
                    "policy_mean_gain_vs_baseline": float(gains[selected].sum() / len(per_item)),
                    "random_same_fraction_expected_gain": float(fraction * np.maximum(gains, 0.0).mean()),
                    "oracle_item_mean_gain": float(np.maximum(gains, 0.0).mean()),
                }
            )
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(
            ["policy_mean_gain_vs_baseline", "selected_mean_oracle_gain"],
            ascending=[False, False],
        )
    return per_item, summary


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
        else:
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in text.values.tolist())
    return "\n".join(lines)


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    variants = [alpha_name(alpha) for alpha in alphas]
    alpha_by_variant = dict(zip(variants, alphas, strict=True))
    baseline_variant = alpha_name(max(alphas))

    paths = degraded.resolve_paths(args.data_root, args.products_path, args.output_dir)
    degraded.require_files(
        paths,
        ["products", "test", "ground_truth", "item2index_warm", "warm_embeddings", "cold_embeddings"],
    )
    checkpoint_path = degraded.require_file(args.a2_checkpoint, "A2 checkpoint")
    degraded.import_letitgo_runtime()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    ground_truth = pl.read_parquet(paths["ground_truth"])
    test_interactions = pl.read_parquet(paths["test"])
    warm_item2index = degraded.normalize_item2index(
        degraded.load_pickle(paths["item2index_warm"]),
        "item2index_warm",
    )
    warm_gt = degraded.build_warm_ground_truth(
        ground_truth,
        max_eval_users=args.max_eval_users,
        max_target_items=args.max_target_items,
    )
    target_item_ids = set(int(value) for value in warm_gt.get_column("item_id").unique().to_list())
    target_rows, product_summary = degraded.read_warm_products(
        paths["products"],
        warm_item2index,
        target_item_ids,
        locale=args.locale,
    )
    target_gt = degraded.build_target_ground_truth(warm_gt, target_rows)
    target_user_ids = target_gt["user_id"].unique().tolist()
    test_subset = degraded.filter_test_to_users(test_interactions, target_user_ids)
    target_item_ids_ordered = [int(row["item_id"]) for row in target_rows]

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths["data_root"]),
        "products": str(paths["products"]),
        "output_dir": str(paths["output_dir"]),
        "a2_checkpoint": str(checkpoint_path),
        "alphas": alphas,
        "variants": variants,
        "baseline_variant": baseline_variant,
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
    pd.DataFrame(target_rows).to_csv(paths["output_dir"] / "target_item_profile.csv", index=False)
    target_gt.to_csv(paths["output_dir"] / "target_ground_truth_profile.csv", index=False)

    print("Amazon-M2 warm delta alpha response")
    print("output_dir:", paths["output_dir"])
    print("checkpoint:", checkpoint_path)
    print("alphas:", alphas)
    print("baseline_variant:", baseline_variant)
    print("warm_gt_rows:", len(warm_gt), "target_items:", len(target_rows), "test_users:", len(target_user_ids))

    if args.check_only:
        print("CHECK_ONLY: inputs resolved; no prediction run.")
        return

    _, warm_projected, cold_projected = degraded.load_projected_base_embeddings(
        checkpoint_path,
        paths["warm_embeddings"],
        paths["cold_embeddings"],
    )

    summaries: list[pd.DataFrame] = []
    details: list[pd.DataFrame] = []
    for run_idx, variant in enumerate(variants, start=1):
        alpha = alpha_by_variant[variant]
        print(f"\n===== [{run_idx}/{len(variants)}] variant={variant} alpha={alpha:g} START =====")
        recommender = degraded.load_a2_recommender(
            checkpoint_path=checkpoint_path,
            warm_embeddings=warm_projected,
            cold_embeddings=cold_projected,
            topk=args.topk,
        )
        scale_target_delta(recommender.model, target_item_ids_ordered, alpha=alpha)
        recommendations = degraded.predict_recommendations(
            recommender=recommender,
            test_interactions=test_subset,
            ground_truth=warm_gt,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        detail = degraded.build_hit_detail(recommendations, target_gt, topk=args.topk)
        detail["variant"] = variant
        detail["alpha"] = alpha
        summary = degraded.summarize_metrics(variant, detail, topk=args.topk)
        summary["alpha"] = alpha
        detail.to_csv(paths["output_dir"] / f"hit_detail_{variant}.csv", index=False)
        summaries.append(summary)
        details.append(detail)
        print(f"===== [{run_idx}/{len(variants)}] variant={variant} DONE =====")

    summary = pd.concat(summaries, ignore_index=True)
    hit_detail = pd.concat(details, ignore_index=True)
    response, fixed_summary, oracle_summary, item_response = build_response_tables(
        hit_detail,
        topk=args.topk,
        baseline_variant=baseline_variant,
    )
    feature_table = build_feature_table(target_rows)
    per_item, selector = selector_summary(
        item_response,
        feature_table,
        baseline_variant=baseline_variant,
        random_state=args.random_state,
        n_splits=args.n_splits,
    )

    summary.to_csv(paths["output_dir"] / "alpha_metrics.csv", index=False)
    hit_detail.to_csv(paths["output_dir"] / "alpha_hit_detail_all.csv", index=False)
    response.to_csv(paths["output_dir"] / "alpha_record_response.csv", index=False)
    fixed_summary.to_csv(paths["output_dir"] / "alpha_fixed_summary.csv", index=False)
    oracle_summary.to_csv(paths["output_dir"] / "alpha_oracle_summary.csv", index=False)
    item_response.to_csv(paths["output_dir"] / "alpha_item_response.csv", index=False)
    per_item.to_csv(paths["output_dir"] / "alpha_item_oracle.csv", index=False)
    selector.to_csv(paths["output_dir"] / "alpha_item_selector_cv_summary.csv", index=False)

    report_lines = [
        "# Amazon-M2 warm delta alpha response",
        "",
        "## Overall Metrics",
        "",
        markdown_table(summary[summary["field_group"] == "all"]),
        "",
        "## Fixed Alpha Summary",
        "",
        markdown_table(fixed_summary),
        "",
        "## Record-Level Oracle",
        "",
        markdown_table(oracle_summary),
        "",
        "## Best Item-Level Selector",
        "",
        markdown_table(selector.head(1)),
        "",
        "## Notes",
        "",
        "- Baseline is the largest alpha in the run, normally alpha=1.0, i.e. original A2 warm delta.",
        "- This is a warm-target local response diagnostic. It does not yet prove strict cold transfer.",
    ]
    (paths["output_dir"] / "2026-06-07 Amazon-M2 warm-delta alpha response 结果.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("DONE: warm delta alpha response outputs generated.")
    print(fixed_summary.to_string(index=False))
    print(oracle_summary.to_string(index=False))
    if len(selector):
        print(selector.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
