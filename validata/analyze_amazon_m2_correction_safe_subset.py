"""Amazon-M2 correction-safe subset oracle 诊断。

这个脚本不训练模型，也不重新 fit PCA。它复用 cold-delta generation probe 的
A2 checkpoint 和 generated cold delta 逻辑，进一步检查：

1. generated cold delta 虽然整体失败，是否存在局部受益 cold item 子集；
2. 这些受益/受损样本是否能被可观测特征分桶解释。

如果找不到稳定的 correction-safe 子集，就应停止 cold-delta 方法线。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import numpy as np
import pandas as pd
import polars as pl

from validata.evaluate_amazon_m2_generated_cold_delta_probe import (
    DEFAULT_A2_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_PRODUCTS_PATH,
    FIELD_GROUP_ORDER,
    TEMP_ROOT,
    build_count_table,
    build_ground_truth_groups,
    generate_neighbor_delta,
    import_letitgo_runtime,
    limit_eval_users,
    load_a2_recommender,
    load_pickle,
    load_projected_embeddings,
    load_warm_delta,
    normalize_item2index,
    parse_float_list,
    parse_int_list,
    predict_recommendations,
    read_cold_products,
    require_file,
    require_files,
    resolve_paths,
    safe_row_cosine,
)


DEFAULT_OUTPUT_DIR = TEMP_ROOT / "correction-safe-subset-oracle"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether generated cold delta has correction-safe subsets."
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


def add_recommendation_rank(recommendations: pl.DataFrame) -> pl.DataFrame:
    """按模型输出顺序给每个 user 的推荐列表补 rank。

    Lightning predict 输出已经是按分数降序截断后的 top-k 列表。这里保留原始行顺序，
    用 groupby cumcount 得到 rank，避免不同库的 rank tie 处理带来额外差异。
    """
    pdf = recommendations.select(["user_id", "item_id", "rating"]).to_pandas()
    pdf["rank"] = pdf.groupby("user_id", sort=False).cumcount() + 1
    return pl.from_pandas(pdf)


def build_record_metrics(
    group: str,
    recommendations: pl.DataFrame,
    cold_ground_truth: pl.DataFrame,
    metric_topk: int,
) -> pl.DataFrame:
    """把推荐列表转成每条 cold ground-truth 的 hit/rank/NDCG。"""
    ranked = add_recommendation_rank(recommendations).select(
        ["user_id", "item_id", "rank", "rating"]
    )
    gt_columns = [
        column
        for column in ["user_id", "item_id", "field_group", "present_field_count"]
        if column in cold_ground_truth.columns
    ]
    records = cold_ground_truth.select(gt_columns).join(
        ranked,
        on=["user_id", "item_id"],
        how="left",
    )
    return records.with_columns(
        [
            pl.lit(group).alias("group"),
            pl.col("rank").is_not_null().cast(pl.Int64).alias("hit"),
            pl.when(pl.col("rank").is_not_null())
            .then(1.0 / ((pl.col("rank") + 1).cast(pl.Float64).log(base=2)))
            .otherwise(0.0)
            .alias("ndcg"),
            pl.lit(metric_topk).alias("metric_topk"),
        ]
    )


def compare_against_baseline(
    baseline_records: pl.DataFrame,
    generated_records: pl.DataFrame,
) -> pl.DataFrame:
    """比较 generated 和 A2 zero-delta 在每条 cold GT 上的变化。"""
    baseline = baseline_records.select(
        [
            "user_id",
            "item_id",
            pl.col("rank").alias("a2_rank"),
            pl.col("hit").alias("a2_hit"),
            pl.col("ndcg").alias("a2_ndcg"),
        ]
    )
    compared = generated_records.join(baseline, on=["user_id", "item_id"], how="left")
    return compared.with_columns(
        [
            (pl.col("ndcg") - pl.col("a2_ndcg")).alias("delta_ndcg"),
            (pl.col("hit") - pl.col("a2_hit")).alias("delta_hit"),
        ]
    ).with_columns(
        pl.when(pl.col("delta_ndcg") > EPS)
        .then(pl.lit("better"))
        .when(pl.col("delta_ndcg") < -EPS)
        .then(pl.lit("worse"))
        .otherwise(pl.lit("same"))
        .alias("oracle_status")
    )


def summarize_oracle_records(records: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """汇总每个桶里 generated 相对 A2 的受益/受损比例。"""
    return (
        records.group_by(by)
        .agg(
            [
                pl.len().alias("records"),
                (pl.col("delta_ndcg") > EPS).sum().alias("better_records"),
                (pl.col("delta_ndcg") < -EPS).sum().alias("worse_records"),
                (pl.col("delta_ndcg").abs() <= EPS).sum().alias("same_records"),
                (pl.col("delta_hit") > 0).sum().alias("hit_gain_records"),
                (pl.col("delta_hit") < 0).sum().alias("hit_loss_records"),
                pl.col("delta_ndcg").mean().alias("mean_delta_ndcg"),
                pl.col("delta_ndcg").median().alias("median_delta_ndcg"),
                pl.col("delta_hit").mean().alias("mean_delta_hit"),
            ]
        )
        .with_columns(
            [
                (pl.col("better_records") / pl.col("records")).alias("better_rate"),
                (pl.col("worse_records") / pl.col("records")).alias("worse_rate"),
                (pl.col("same_records") / pl.col("records")).alias("same_rate"),
                (pl.col("hit_gain_records") / pl.col("records")).alias("hit_gain_rate"),
                (pl.col("hit_loss_records") / pl.col("records")).alias("hit_loss_rate"),
            ]
        )
        .sort(by)
    )


def build_cold_item_feature_table(
    cold_rows: list[dict[str, object]],
    cold_embeddings: np.ndarray,
) -> pl.DataFrame:
    rows = []
    for row in cold_rows:
        item_id = int(row["item_id"])
        position = int(row["position_in_embedding_file"])
        embedding_norm = float(np.linalg.norm(cold_embeddings[position]))
        rows.append(
            {
                "item_id": item_id,
                "raw_item_id": row["raw_item_id"],
                "title_len": len(str(row.get("title") or "")),
                "brand_present": bool(row.get("brand_present")),
                "author_present": bool(row.get("author_present")),
                "metadata_found": bool(row.get("metadata_found")),
                "cold_content_norm": embedding_norm,
            }
        )
    return pl.from_dicts(rows)


def build_generated_feature_table(
    group: str,
    cold_embeddings: np.ndarray,
    generated_delta: np.ndarray,
    details: pd.DataFrame,
) -> pl.DataFrame:
    content_final_cosine = safe_row_cosine(cold_embeddings, cold_embeddings + generated_delta)
    feature_table = pl.from_pandas(details).with_columns(pl.lit(group).alias("group"))
    # 中文注释：content-final cosine 按 cold embedding 文件顺序排列；
    # details 也是同一顺序生成，所以可以直接按 item_id 对齐。
    cosine_table = pl.DataFrame(
        {
            "item_id": details["item_id"].astype(int).to_numpy(),
            "cold_content_final_cosine": content_final_cosine.astype(float),
        }
    )
    return feature_table.join(cosine_table, on="item_id", how="left")


def add_quantile_bucket(
    records: pl.DataFrame,
    column: str,
    bucket_column: str,
    labels: tuple[str, str, str] = ("low", "mid", "high"),
) -> pl.DataFrame:
    if column not in records.columns:
        return records
    pdf = records.to_pandas()
    values = pd.to_numeric(pdf[column], errors="coerce")
    if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
        pdf[bucket_column] = "all"
    else:
        try:
            bucketed = pd.qcut(values, q=3, labels=labels, duplicates="drop")
            pdf[bucket_column] = bucketed.astype(str).replace("nan", "missing")
        except ValueError:
            pdf[bucket_column] = "all"
        pdf.loc[values.isna(), bucket_column] = "missing"
    return pl.from_pandas(pdf)


def add_oracle_buckets(records: pl.DataFrame) -> pl.DataFrame:
    output = records
    bucket_specs = [
        ("nearest_warm_cosine", "nearest_warm_cosine_bucket"),
        ("neighbor_cosine_mean", "neighbor_cosine_mean_bucket"),
        ("generated_delta_norm", "generated_delta_norm_bucket"),
        ("cold_content_final_cosine", "cold_content_final_cosine_bucket"),
        ("title_len", "title_len_bucket"),
        ("cold_content_norm", "cold_content_norm_bucket"),
    ]
    for column, bucket_column in bucket_specs:
        output = add_quantile_bucket(output, column, bucket_column)
    return output


def summarize_bucket_records(records: pl.DataFrame, bucket_columns: list[str]) -> pl.DataFrame:
    tables = []
    for bucket_column in bucket_columns:
        if bucket_column not in records.columns:
            continue
        summary = summarize_oracle_records(records, by=["group", bucket_column]).rename(
            {bucket_column: "bucket"}
        )
        tables.append(summary.with_columns(pl.lit(bucket_column).alias("bucket_feature")))
    if not tables:
        return pl.DataFrame()
    return pl.concat(tables).select(
        [
            "group",
            "bucket_feature",
            "bucket",
            "records",
            "better_records",
            "worse_records",
            "same_records",
            "better_rate",
            "worse_rate",
            "mean_delta_ndcg",
            "median_delta_ndcg",
            "mean_delta_hit",
        ]
    )


def format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_无记录_"
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_result_markdown(output_dir: Path, key_summary: pl.DataFrame, field_summary: pl.DataFrame) -> None:
    created = datetime.now().strftime("%Y-%m-%d %H%M%S")
    path = output_dir / f"{created} Amazon-M2 correction-safe subset oracle 诊断结果.md"
    key_preview = key_summary.sort(["mean_delta_ndcg"], descending=True).head(12).to_pandas()
    field_preview = field_summary.sort(["group", "field_group"]).to_pandas()

    path.write_text(
        "\n".join(
            [
                f"# {created} Amazon-M2 correction-safe subset oracle 诊断结果",
                "",
                f"创建时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 这次在问什么",
                "",
                "这次不重新训练，也不重新 fit PCA。它接着 `cold-delta generation probe` 的负结果继续问：",
                "",
                "> generated cold delta 虽然整体不如 A2 zero delta，但有没有某些 cold item 子集其实被它帮到了？",
                "",
                "如果存在稳定、可观测的受益子集，后面才有理由做 evidence-aware / uncertainty-aware gate。"
                "如果不存在，就应该停止 cold-delta 方法线。",
                "",
                "## 关键文件",
                "",
                "```text",
                "oracle_record_delta.csv",
                "oracle_group_summary.csv",
                "oracle_field_group_summary.csv",
                "oracle_bucket_summary.csv",
                "```",
                "",
                "## 整体 oracle 摘要",
                "",
                "> [!info] 表格字段说明",
                "> `group`：generated cold delta 组。  ",
                "> `records`：参与比较的 cold ground-truth 记录数。  ",
                "> `better_records / worse_records / same_records`：相对 A2 zero delta 的 NDCG 变好、变差、相同记录数。  ",
                "> `better_rate / worse_rate`：变好/变差比例。  ",
                "> `mean_delta_ndcg`：generated NDCG 减去 A2 NDCG 的平均值；大于 0 才代表平均受益。",
                "",
                dataframe_to_markdown(key_preview),
                "",
                "## weak / mid / strong 分组摘要",
                "",
                "> [!info] 表格字段说明",
                "> `field_group`：字段完整度分组。  ",
                "> `mean_delta_ndcg`：该分组内 generated 相对 A2 的平均 NDCG 变化。  ",
                "> `better_rate / worse_rate`：该分组内变好/变差比例。",
                "",
                dataframe_to_markdown(field_preview),
                "",
                "## 临时判读",
                "",
                "正式结论需要结合 CSV 全表看。粗略判断时重点看：",
                "",
                "- 是否存在 `mean_delta_ndcg > 0` 且 `better_rate` 明显高于 `worse_rate` 的稳定组；",
                "- 这种稳定组是否集中在可观测桶里，而不是只出现在某一个偶然的 `topk/alpha`；",
                "- weak_0_1 是否有稳定受益。如果 weak 没有受益，就不能接回 weak-evidence robustness 主线。",
                "",
            ]
        ),
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
    cold_ground_truth = ground_truth_groups.filter(pl.col("is_cold"))
    count_table = build_count_table(cold_rows, ground_truth_groups)

    warm_embeddings, cold_embeddings = load_projected_embeddings(
        checkpoint_path=checkpoint_path,
        warm_embeddings_path=paths["warm_embeddings"],
        cold_embeddings_path=paths["cold_embeddings"],
    )
    warm_delta = load_warm_delta(checkpoint_path, expected_warm_items=warm_embeddings.shape[0])
    item_features = build_cold_item_feature_table(cold_rows, cold_embeddings)

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

    print("Amazon-M2 correction-safe subset oracle diagnostic")
    print("data_root:", paths["data_root"])
    print("products:", paths["products"])
    print("output_dir:", output_dir)
    print("a2_checkpoint:", checkpoint_path)
    print("cold_ground_truth_rows:", len(cold_ground_truth))
    print("warm_embedding_shape:", warm_embeddings.shape)
    print("cold_embedding_shape:", cold_embeddings.shape)
    print("warm_delta_shape:", warm_delta.shape)
    print(count_table)

    if args.check_only:
        write_manifest(output_dir, manifest)
        print("CHECK_ONLY: 输入、字段分组、embedding 和 delta 形状检查完成；未跑预测。")
        return

    all_generated_records = []
    all_feature_tables = []

    print()
    print("===== [baseline] A2_original_zero_delta START =====", flush=True)
    baseline_start = time.perf_counter()
    baseline_recommender = load_a2_recommender(
        checkpoint_path=checkpoint_path,
        warm_embeddings=warm_embeddings,
        cold_embeddings=cold_embeddings,
        cold_delta=np.zeros_like(cold_embeddings, dtype=np.float32),
        metric_topk=args.metric_topk,
    )
    baseline_recommendations = predict_recommendations(
        recommender=baseline_recommender,
        test_interactions=test_interactions,
        ground_truth=ground_truth,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    baseline_records = build_record_metrics(
        group="A2_original_zero_delta",
        recommendations=baseline_recommendations,
        cold_ground_truth=cold_ground_truth,
        metric_topk=args.metric_topk,
    )
    baseline_records.write_csv(output_dir / "oracle_a2_record_metrics.csv")
    print(
        f"===== [baseline] A2_original_zero_delta DONE, "
        f"elapsed={time.perf_counter() - baseline_start:.2f}s =====",
        flush=True,
    )

    total_runs = len(topk_values) * len(alpha_values)
    run_index = 0
    for neighbor_topk in topk_values:
        for alpha in alpha_values:
            run_index += 1
            group = f"generated_top{neighbor_topk}_alpha{alpha:g}"
            run_start = time.perf_counter()
            print()
            print(f"===== [{run_index}/{total_runs}] group={group} START =====", flush=True)

            generated_delta, details = generate_neighbor_delta(
                warm_content=warm_embeddings,
                cold_content=cold_embeddings,
                warm_delta=warm_delta,
                topk=neighbor_topk,
                alpha=alpha,
                batch_size=args.neighbor_batch_size,
            )
            generated_features = build_generated_feature_table(
                group=group,
                cold_embeddings=cold_embeddings,
                generated_delta=generated_delta,
                details=details,
            )
            recommender = load_a2_recommender(
                checkpoint_path=checkpoint_path,
                warm_embeddings=warm_embeddings,
                cold_embeddings=cold_embeddings,
                cold_delta=generated_delta,
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
            generated_records = build_record_metrics(
                group=group,
                recommendations=recommendations,
                cold_ground_truth=cold_ground_truth,
                metric_topk=args.metric_topk,
            )
            compared = compare_against_baseline(baseline_records, generated_records)
            compared = (
                compared.join(generated_features, on=["group", "item_id"], how="left")
                .join(item_features, on="item_id", how="left")
                .with_columns(
                    [
                        pl.lit(neighbor_topk).alias("topk"),
                        pl.lit(alpha).alias("alpha"),
                    ]
                )
            )
            all_generated_records.append(compared)
            all_feature_tables.append(generated_features)

            generated_records.write_csv(output_dir / f"{group}_record_metrics.csv")
            print(
                f"===== [{run_index}/{total_runs}] group={group} DONE, "
                f"elapsed={time.perf_counter() - run_start:.2f}s =====",
                flush=True,
            )

    oracle_records = add_oracle_buckets(pl.concat(all_generated_records))
    generated_feature_table = pl.concat(all_feature_tables)
    bucket_columns = [
        "nearest_warm_cosine_bucket",
        "neighbor_cosine_mean_bucket",
        "generated_delta_norm_bucket",
        "cold_content_final_cosine_bucket",
        "title_len_bucket",
        "cold_content_norm_bucket",
    ]
    group_summary = summarize_oracle_records(oracle_records, by=["group"])
    field_group_summary = summarize_oracle_records(oracle_records, by=["group", "field_group"])
    bucket_summary = summarize_bucket_records(oracle_records, bucket_columns=bucket_columns)

    oracle_records.write_csv(output_dir / "oracle_record_delta.csv")
    generated_feature_table.write_csv(output_dir / "oracle_generated_delta_features.csv")
    group_summary.write_csv(output_dir / "oracle_group_summary.csv")
    field_group_summary.write_csv(output_dir / "oracle_field_group_summary.csv")
    if len(bucket_summary) > 0:
        bucket_summary.write_csv(output_dir / "oracle_bucket_summary.csv")
    write_manifest(output_dir, manifest)
    write_result_markdown(output_dir, group_summary, field_group_summary)
    print("DONE: correction-safe subset oracle outputs 已生成。")


if __name__ == "__main__":
    main()
