"""Amazon-M2 自然分组第二层命中明细诊断。

本脚本不训练模型，也不重新预测；它读取已经导出的 top-k 推荐明细，
把每条 cold ground-truth 是否被命中、命中排名、粗条件分层差距整理出来。
用途：检查 weak/mid/strong 的差距是否在用户-物品命中层面仍然成立。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent
TEMP_ROOT = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260605"

DEFAULT_RECOMMENDATION_DIR = TEMP_ROOT / "自然分组命中明细诊断" / "recommendation_export"
DEFAULT_FIELD_OUTPUT_DIR = (
    TEMP_ROOT / "自然字段完整度分组评测_FR口径修正版" / "field_group_eval_outputs"
)
DEFAULT_CONFOUND_TABLE = TEMP_ROOT / "自然分组混杂因素诊断" / "cold_item_confound_table.csv"
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "自然分组命中明细诊断"

FIELD_GROUP_ORDER = ["weak_0_1", "mid_2", "strong_3_4", "missing_metadata"]
PROXY_COLUMNS = [
    "title_len_bucket",
    "brand_present",
    "desc_len_bucket",
    "author_present",
    "price_bucket",
    "raw_id_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze item/user-level recommendation hits for Amazon-M2 natural groups."
    )
    parser.add_argument(
        "--recommendations",
        type=Path,
        default=DEFAULT_RECOMMENDATION_DIR / "recommendations_A2.csv.gz",
    )
    parser.add_argument(
        "--ground-truth-groups",
        type=Path,
        default=DEFAULT_FIELD_OUTPUT_DIR / "ground_truth_cold_with_field_group.csv",
    )
    parser.add_argument(
        "--field-profile",
        type=Path,
        default=DEFAULT_FIELD_OUTPUT_DIR / "field_profile_cold_items.csv",
    )
    parser.add_argument("--confound-table", type=Path, default=DEFAULT_CONFOUND_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="A2")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查输入文件是否存在，不生成结果。",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.expanduser().is_file():
        raise FileNotFoundError(f"缺少输入文件：{label}: {path.expanduser().resolve()}")


def load_recommendations(path: Path, topk: int) -> pd.DataFrame:
    recommendations = pd.read_csv(path)
    required = {"user_id", "item_id", "rating"}
    missing = sorted(required - set(recommendations.columns))
    if missing:
        raise ValueError(f"recommendations 缺少列：{missing}")

    # 中文注释：这里对齐 replay.metrics 的 Polars 排序语义：
    # 先按 rating 降序，再按 item_id 降序。这样分数并列时的排名也和原指标一致。
    recommendations = recommendations.sort_values(
        ["user_id", "rating", "item_id"],
        ascending=[True, False, False],
    ).copy()
    recommendations["rank"] = recommendations.groupby("user_id").cumcount() + 1
    return recommendations[recommendations["rank"] <= topk]


def build_hit_detail(
    recommendations: pd.DataFrame,
    ground_truth_groups: pd.DataFrame,
    field_profile: pd.DataFrame,
    confound_table: pd.DataFrame,
    topk: int,
) -> pd.DataFrame:
    cold_gt = ground_truth_groups[ground_truth_groups["is_cold"].astype(bool)].copy()
    cold_gt = cold_gt[["user_id", "item_id", "is_cold", "field_group", "present_field_count"]]

    rec_hit_cols = recommendations[["user_id", "item_id", "rank", "rating"]].copy()
    detail = cold_gt.merge(rec_hit_cols, on=["user_id", "item_id"], how="left")
    detail["hit"] = detail["rank"].notna()
    detail["rank"] = detail["rank"].astype("Int64")
    detail[f"recall_contribution@{topk}"] = detail["hit"].astype(float)
    detail[f"ndcg_contribution@{topk}"] = np.where(
        detail["hit"],
        1.0 / np.log2(detail["rank"].astype(float) + 1.0),
        0.0,
    )

    profile_cols = [
        column
        for column in ["item_id", "raw_item_id", "metadata_found", "color_present", "size_present", "model_present", "material_present"]
        if column in field_profile.columns
    ]
    if profile_cols:
        detail = detail.merge(field_profile[profile_cols], on="item_id", how="left")

    confound_cols = [
        "item_id",
        "title_tokens",
        "title_len_bucket",
        "brand_present",
        "desc_tokens",
        "desc_len_bucket",
        "author_present",
        "price_bucket",
        "raw_id_type",
        "cold_ground_truth_rows",
    ]
    confound_cols = [column for column in confound_cols if column in confound_table.columns]
    if confound_cols:
        detail = detail.merge(confound_table[confound_cols], on="item_id", how="left")

    return detail.sort_values(["field_group", "user_id", "item_id"]).reset_index(drop=True)


def summarize_field_groups(hit_detail: pd.DataFrame, topk: int) -> pd.DataFrame:
    rows = []
    for group_name in FIELD_GROUP_ORDER:
        group = hit_detail[hit_detail["field_group"] == group_name]
        rows.append(
            {
                "field_group": group_name,
                "cold_ground_truth_rows": len(group),
                "cold_gt_items": group["item_id"].nunique() if len(group) else 0,
                "hit_rows": int(group["hit"].sum()) if len(group) else 0,
                f"cold_Recall@{topk}": float(group[f"recall_contribution@{topk}"].mean())
                if len(group)
                else 0.0,
                f"cold_NDCG@{topk}": float(group[f"ndcg_contribution@{topk}"].mean())
                if len(group)
                else 0.0,
                "mean_hit_rank": float(group.loc[group["hit"], "rank"].mean())
                if group["hit"].any()
                else np.nan,
            }
        )

    return round_float_columns(pd.DataFrame(rows))


def summarize_item_concentration(hit_detail: pd.DataFrame, topk: int) -> pd.DataFrame:
    item_summary = (
        hit_detail.groupby(["field_group", "item_id"], dropna=False)
        .agg(
            raw_item_id=("raw_item_id", "first"),
            gt_rows=("user_id", "size"),
            hit_rows=("hit", "sum"),
            ndcg_sum=(f"ndcg_contribution@{topk}", "sum"),
            mean_hit_rank=("rank", "mean"),
            title_len_bucket=("title_len_bucket", "first"),
            desc_len_bucket=("desc_len_bucket", "first"),
            raw_id_type=("raw_id_type", "first"),
        )
        .reset_index()
    )
    item_summary["hit_rate"] = item_summary["hit_rows"] / item_summary["gt_rows"].replace(0, np.nan)
    item_summary = item_summary.sort_values(
        ["field_group", "hit_rows", "ndcg_sum", "gt_rows"],
        ascending=[True, False, False, False],
    )
    return round_float_columns(item_summary)


def summarize_group_concentration(item_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in FIELD_GROUP_ORDER:
        group = item_summary[item_summary["field_group"] == group_name].copy()
        total_hits = float(group["hit_rows"].sum()) if len(group) else 0.0
        total_gt = float(group["gt_rows"].sum()) if len(group) else 0.0
        top_hit_items = group.sort_values("hit_rows", ascending=False).head(10)
        rows.append(
            {
                "field_group": group_name,
                "gt_rows": int(total_gt),
                "hit_rows": int(total_hits),
                "item_count": len(group),
                "items_with_hit": int((group["hit_rows"] > 0).sum()) if len(group) else 0,
                "top1_item_hit_share": float(top_hit_items.head(1)["hit_rows"].sum() / total_hits)
                if total_hits
                else 0.0,
                "top5_item_hit_share": float(top_hit_items.head(5)["hit_rows"].sum() / total_hits)
                if total_hits
                else 0.0,
                "top10_item_hit_share": float(top_hit_items["hit_rows"].sum() / total_hits)
                if total_hits
                else 0.0,
            }
        )
    return round_float_columns(pd.DataFrame(rows))


def summarize_proxy_metrics(hit_detail: pd.DataFrame, topk: int) -> pd.DataFrame:
    rows = []
    for proxy in PROXY_COLUMNS:
        if proxy not in hit_detail.columns:
            continue
        data = hit_detail.copy()
        data[proxy] = data[proxy].fillna("missing").astype(str)
        grouped = data.groupby([proxy, "field_group"], dropna=False)
        for (bucket, group_name), group in grouped:
            rows.append(
                {
                    "proxy": proxy,
                    "proxy_bucket": bucket,
                    "field_group": group_name,
                    "cold_ground_truth_rows": len(group),
                    "cold_gt_items": group["item_id"].nunique(),
                    "hit_rows": int(group["hit"].sum()),
                    f"cold_Recall@{topk}": float(group[f"recall_contribution@{topk}"].mean()),
                    f"cold_NDCG@{topk}": float(group[f"ndcg_contribution@{topk}"].mean()),
                }
            )
    return round_float_columns(pd.DataFrame(rows))


def summarize_proxy_gaps(proxy_metrics: pd.DataFrame, topk: int) -> pd.DataFrame:
    if proxy_metrics.empty:
        return pd.DataFrame()

    pivot = proxy_metrics.pivot_table(
        index=["proxy", "proxy_bucket"],
        columns="field_group",
        values=[f"cold_NDCG@{topk}", f"cold_Recall@{topk}", "cold_ground_truth_rows"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns]
    pivot = pivot.reset_index()

    for group_name in ["weak_0_1", "strong_3_4"]:
        for metric in [f"cold_NDCG@{topk}", f"cold_Recall@{topk}", "cold_ground_truth_rows"]:
            column = f"{metric}_{group_name}"
            if column not in pivot.columns:
                pivot[column] = np.nan

    pivot["has_weak_and_strong_gt"] = (
        pivot["cold_ground_truth_rows_weak_0_1"].fillna(0) > 0
    ) & (pivot["cold_ground_truth_rows_strong_3_4"].fillna(0) > 0)
    pivot[f"ndcg_gap_strong_minus_weak@{topk}"] = (
        pivot[f"cold_NDCG@{topk}_strong_3_4"] - pivot[f"cold_NDCG@{topk}_weak_0_1"]
    )
    pivot[f"recall_gap_strong_minus_weak@{topk}"] = (
        pivot[f"cold_Recall@{topk}_strong_3_4"] - pivot[f"cold_Recall@{topk}_weak_0_1"]
    )
    return round_float_columns(
        pivot.sort_values(["proxy", "proxy_bucket"]).reset_index(drop=True)
    )


def round_float_columns(table: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    result = table.copy()
    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = result[column].round(digits)
    return result


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if table.empty:
        return "_无数据_"
    view = table.loc[:, [column for column in columns if column in table.columns]].head(max_rows)
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(format_cell(row[column]) for column in view.columns) + " |")
    return "\n".join([header, separator, *rows])


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_result_markdown(
    output_dir: Path,
    model: str,
    topk: int,
    group_metrics: pd.DataFrame,
    group_concentration: pd.DataFrame,
    proxy_gaps: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    a = group_metrics.set_index("field_group")
    weak_ndcg = float(a.loc["weak_0_1", f"cold_NDCG@{topk}"]) if "weak_0_1" in a.index else 0.0
    strong_ndcg = (
        float(a.loc["strong_3_4", f"cold_NDCG@{topk}"]) if "strong_3_4" in a.index else 0.0
    )
    weak_recall = (
        float(a.loc["weak_0_1", f"cold_Recall@{topk}"]) if "weak_0_1" in a.index else 0.0
    )
    strong_recall = (
        float(a.loc["strong_3_4", f"cold_Recall@{topk}"]) if "strong_3_4" in a.index else 0.0
    )

    comparable = proxy_gaps[proxy_gaps["has_weak_and_strong_gt"].astype(bool)].copy()
    if not comparable.empty:
        gap_col = f"ndcg_gap_strong_minus_weak@{topk}"
        comparable = comparable.sort_values(gap_col, ascending=False)

    content = f"""---
title: 2026-06-05 Amazon-M2 自然分组第二层命中明细诊断结果
date: 2026-06-05
created: 2026-06-05
tags:
  - let-it-go
  - amazon-m2
  - 自然分组
  - 命中明细
  - 混杂诊断
---

# 2026-06-05 Amazon-M2 自然分组第二层命中明细诊断结果

## 一句话结论

> [!note] 第二层诊断
> 本次补的是用户-物品命中明细层面的诊断：读取 `{model}` 的 top-{topk} 推荐列表，检查每条 cold ground-truth 是否被命中、命中排名是多少，以及 strong/weak 差距是否在相似粗条件下仍然存在。没有重新训练模型，也没有重新生成 embedding。

## 通俗解释

第一层只是在看“weak/mid/strong 三组平均分差很多”。第二层再往下问：

```text
1. strong 组高，是不是很多 ground-truth 都真的被推荐中了？
2. 还是只有少数几个热门/容易 item 撑起了分数？
3. 在 title 长度、desc 长度、价格、raw id 类型差不多的粗条件里，strong 是否仍然比 weak 好？
```

## 关键结果

```text
{model} weak_0_1   NDCG@{topk}={weak_ndcg:.4f}, Recall@{topk}={weak_recall:.4f}
{model} strong_3_4 NDCG@{topk}={strong_ndcg:.4f}, Recall@{topk}={strong_recall:.4f}
strong - weak      NDCG@{topk}={strong_ndcg - weak_ndcg:.4f}, Recall@{topk}={strong_recall - weak_recall:.4f}
```

## 分组命中明细复算

这张表用推荐明细逐条复算，应该和自然分组评测里的 A2 主表基本一致。

{markdown_table(group_metrics, ["field_group", "cold_ground_truth_rows", "cold_gt_items", "hit_rows", f"cold_Recall@{topk}", f"cold_NDCG@{topk}", "mean_hit_rank"])}

## item 命中集中度

如果 top1/top5 item 命中占比很高，说明某组分数可能被少数 item 撑起来；如果占比不高，说明命中更分散。

{markdown_table(group_concentration, ["field_group", "gt_rows", "hit_rows", "item_count", "items_with_hit", "top1_item_hit_share", "top5_item_hit_share", "top10_item_hit_share"])}

## 粗条件内 strong-weak gap

这里只展示 weak 和 strong 都有 ground-truth 的 bucket。它不是严格因果控制，但比第一层更接近“在相似条件下再比较”。

{markdown_table(comparable, ["proxy", "proxy_bucket", "cold_ground_truth_rows_weak_0_1", "cold_ground_truth_rows_strong_3_4", f"cold_NDCG@{topk}_weak_0_1", f"cold_NDCG@{topk}_strong_3_4", f"ndcg_gap_strong_minus_weak@{topk}"], max_rows=40)}

## 当前能说明什么

```text
1. 如果明细复算和第一层指标一致，说明自然分组差距不是整理表格造成的。
2. 如果 strong 组命中不是集中在极少数 item，说明 strong 组优势不是完全由一两个 item 撑起。
3. 如果多个粗条件内 strong 仍高于 weak，说明自然分组更像 cold item difficulty/content evidence strata。
```

## 当前不能说明什么

```text
1. 仍然不能证明 color/size/model/material 四字段完整度是因果原因；
2. 仍然不能证明手工 q score 方法成立；
3. 仍然不能替代后续真正的方法设计或 full-pipeline retrain。
```

## 输出文件

```text
hit_detail_{model}.csv
field_group_hit_metrics_{model}.csv
item_hit_summary_{model}.csv
group_hit_concentration_{model}.csv
proxy_bucket_hit_metrics_{model}.csv
proxy_bucket_gap_{model}.csv
second_layer_manifest.json
```

## 运行口径

```text
recommendations: {manifest["inputs"]["recommendations"]}
ground_truth_groups: {manifest["inputs"]["ground_truth_groups"]}
confound_table: {manifest["inputs"]["confound_table"]}
generated_at: {manifest["generated_at"]}
```
"""
    (output_dir / f"2026-06-05 Amazon-M2 自然分组第二层命中明细诊断结果.md").write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    recommendations_path = args.recommendations.expanduser().resolve()
    ground_truth_path = args.ground_truth_groups.expanduser().resolve()
    field_profile_path = args.field_profile.expanduser().resolve()
    confound_table_path = args.confound_table.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    require_file(recommendations_path, "recommendations")
    require_file(ground_truth_path, "ground_truth_groups")
    require_file(field_profile_path, "field_profile")
    require_file(confound_table_path, "confound_table")

    print("Amazon-M2 natural group second-layer hit diagnostic")
    print("recommendations:", recommendations_path)
    print("ground_truth_groups:", ground_truth_path)
    print("field_profile:", field_profile_path)
    print("confound_table:", confound_table_path)
    print("output_dir:", output_dir)

    if args.check_only:
        print("CHECK_ONLY: 输入文件检查通过；未生成结果。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    recommendations = load_recommendations(recommendations_path, args.topk)
    ground_truth_groups = pd.read_csv(ground_truth_path)
    field_profile = pd.read_csv(field_profile_path)
    confound_table = pd.read_csv(confound_table_path)

    hit_detail = build_hit_detail(
        recommendations=recommendations,
        ground_truth_groups=ground_truth_groups,
        field_profile=field_profile,
        confound_table=confound_table,
        topk=args.topk,
    )
    group_metrics = summarize_field_groups(hit_detail, args.topk)
    item_summary = summarize_item_concentration(hit_detail, args.topk)
    group_concentration = summarize_group_concentration(item_summary)
    proxy_metrics = summarize_proxy_metrics(hit_detail, args.topk)
    proxy_gaps = summarize_proxy_gaps(proxy_metrics, args.topk)

    hit_detail.to_csv(output_dir / f"hit_detail_{args.model}.csv", index=False)
    group_metrics.to_csv(output_dir / f"field_group_hit_metrics_{args.model}.csv", index=False)
    item_summary.to_csv(output_dir / f"item_hit_summary_{args.model}.csv", index=False)
    group_concentration.to_csv(
        output_dir / f"group_hit_concentration_{args.model}.csv",
        index=False,
    )
    proxy_metrics.to_csv(output_dir / f"proxy_bucket_hit_metrics_{args.model}.csv", index=False)
    proxy_gaps.to_csv(output_dir / f"proxy_bucket_gap_{args.model}.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "topk": args.topk,
        "output_dir": str(output_dir),
        "inputs": {
            "recommendations": str(recommendations_path),
            "ground_truth_groups": str(ground_truth_path),
            "field_profile": str(field_profile_path),
            "confound_table": str(confound_table_path),
        },
    }
    (output_dir / "second_layer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_result_markdown(
        output_dir=output_dir,
        model=args.model,
        topk=args.topk,
        group_metrics=group_metrics,
        group_concentration=group_concentration,
        proxy_gaps=proxy_gaps,
        manifest=manifest,
    )
    print("DONE: 第二层命中明细诊断已输出。")


if __name__ == "__main__":
    main()
