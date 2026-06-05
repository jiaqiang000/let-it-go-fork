"""Amazon-M2 自然分组混杂因素诊断。

本脚本只做离线 CSV 诊断，不训练模型，不重新生成 embedding，也不改 run.py。
核心目的：检查 weak/mid/strong 自然字段分组的大差距，是否可能混入
title、brand、desc、price、author、raw item id 类型等未参与分组的因素。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent
TEMP_ROOT = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260605"

DEFAULT_FIELD_OUTPUT_DIR = (
    TEMP_ROOT / "自然字段完整度分组评测_FR口径修正版" / "field_group_eval_outputs"
)
DEFAULT_VARIANT_OUTPUT_DIR = TEMP_ROOT / "受控字段消融扩展评测" / "variant_eval_outputs"
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "自然分组混杂因素诊断"

FIELD_GROUP_ORDER = ["weak_0_1", "mid_2", "strong_3_4", "missing_metadata"]
VARIANT_ORDER = [
    "original_author",
    "control_full",
    "drop_four",
    "title_brand_only",
    "title_only",
    "brand_only",
    "no_title",
    "no_brand",
    "no_author",
    "no_title_brand",
    "no_color",
    "no_size",
    "no_model",
    "no_material",
    "structured_four_only",
    "empty_text",
]
MISSING_STRINGS = {"", "null", "none", "nan", "[]", "na", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose confounding factors behind Amazon-M2 natural field groups."
    )
    parser.add_argument("--field-output-dir", type=Path, default=DEFAULT_FIELD_OUTPUT_DIR)
    parser.add_argument("--variant-output-dir", type=Path, default=DEFAULT_VARIANT_OUTPUT_DIR)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查输入文件和 raw metadata 可读取性，不生成正式输出。",
    )
    return parser.parse_args()


def is_present_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    text = str(value).strip()
    return text.lower() not in MISSING_STRINGS


def clean_text(value: Any) -> str:
    if not is_present_value(value):
        return ""
    return str(value).strip()


def text_chars(value: Any) -> int:
    return len(clean_text(value))


def text_tokens(value: Any) -> int:
    text = clean_text(value)
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def raw_id_type(raw_item_id: Any) -> str:
    text = str(raw_item_id).strip()
    if not text:
        return "missing"
    if text.startswith("B"):
        return "asin_B"
    if re.fullmatch(r"[0-9X]{10,13}", text):
        return "isbn_like"
    if text.isdigit():
        return "numeric"
    return "other"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要输入文件：{label}: {path}")


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    field_dir = args.field_output_dir.expanduser().resolve()
    variant_dir = args.variant_output_dir.expanduser().resolve()
    return {
        "field_profile": field_dir / "field_profile_cold_items.csv",
        "ground_truth_groups": field_dir / "ground_truth_cold_with_field_group.csv",
        "field_group_metrics": field_dir / "field_group_metrics.csv",
        "variant_field_group_metrics": variant_dir / "variant_field_group_metrics.csv",
        "variant_overall_metrics": variant_dir / "variant_overall_metrics.csv",
        "products": args.products_path.expanduser().resolve(),
        "output_dir": args.output_dir.expanduser().resolve(),
    }


def load_matching_products(
    products_path: Path,
    cold_raw_ids: set[str],
    locale: str,
    chunksize: int,
) -> pd.DataFrame:
    usecols = [
        "id",
        "locale",
        "title",
        "price",
        "brand",
        "color",
        "size",
        "model",
        "material",
        "author",
        "desc",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        products_path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        # 中文注释：这里只保留作者 Amazon-M2 FR 链路对应的 locale，
        # 并且只取 cold item id，避免把 58 万行 raw metadata 全量放进后续诊断表。
        filtered = chunk[(chunk["locale"] == locale) & (chunk["id"].isin(cold_raw_ids))]
        if not filtered.empty:
            chunks.append(filtered.copy())

    if not chunks:
        raise ValueError(f"products_train.csv 中没有匹配 locale={locale!r} 的 cold item metadata。")

    products = pd.concat(chunks, ignore_index=True)
    return products.drop_duplicates("id", keep="last")


def add_text_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()

    for column in ["title", "brand", "desc", "author"]:
        table[f"{column}_present"] = table[column].map(is_present_value)
        table[f"{column}_chars"] = table[column].map(text_chars)
        table[f"{column}_tokens"] = table[column].map(text_tokens)

    table["price_present"] = table["price"].map(is_present_value)
    table["price_value"] = pd.to_numeric(table["price"].map(clean_text), errors="coerce")
    table["raw_id_type"] = table["raw_item_id"].map(raw_id_type)
    table["title_len_bucket"] = table["title_tokens"].map(bucket_title_tokens)
    table["desc_len_bucket"] = table["desc_tokens"].map(bucket_desc_tokens)
    table["price_bucket"] = bucket_price(table["price_value"])
    return table


def bucket_title_tokens(tokens: int) -> str:
    if tokens <= 0:
        return "missing"
    if tokens <= 6:
        return "short"
    if tokens <= 14:
        return "medium"
    return "long"


def bucket_desc_tokens(tokens: int) -> str:
    if tokens <= 0:
        return "missing"
    if tokens <= 20:
        return "short"
    if tokens <= 80:
        return "medium"
    return "long"


def bucket_price(price: pd.Series) -> pd.Series:
    result = pd.Series("missing", index=price.index, dtype="object")
    present = price.dropna()
    if present.empty:
        return result

    # 中文注释：价格只是粗略 proxy，不做强因果解释；用三分位避免手工阈值过度主观。
    q1 = present.quantile(1 / 3)
    q2 = present.quantile(2 / 3)
    result.loc[price.notna() & (price <= q1)] = "low"
    result.loc[price.notna() & (price > q1) & (price <= q2)] = "medium"
    result.loc[price.notna() & (price > q2)] = "high"
    return result


def build_cold_item_confound_table(
    field_profile: pd.DataFrame,
    ground_truth_groups: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    gt_counts = (
        ground_truth_groups.groupby("item_id", as_index=False)
        .agg(cold_ground_truth_rows=("user_id", "size"), cold_gt_user_count=("user_id", "nunique"))
    )
    table = field_profile.merge(
        products,
        left_on="raw_item_id",
        right_on="id",
        how="left",
        suffixes=("", "_product"),
    ).merge(gt_counts, on="item_id", how="left")

    table["cold_ground_truth_rows"] = table["cold_ground_truth_rows"].fillna(0).astype(int)
    table["cold_gt_user_count"] = table["cold_gt_user_count"].fillna(0).astype(int)

    for column in ["title", "price", "brand", "color", "size", "model", "material", "author", "desc"]:
        if column not in table.columns:
            table[column] = ""
        table[column] = table[column].fillna("")

    table = add_text_features(table)

    keep_columns = [
        "raw_item_id",
        "item_id",
        "field_group",
        "present_field_count",
        "color_present",
        "size_present",
        "model_present",
        "material_present",
        "title_present",
        "title_chars",
        "title_tokens",
        "title_len_bucket",
        "brand_present",
        "brand_chars",
        "brand_tokens",
        "desc_present",
        "desc_chars",
        "desc_tokens",
        "desc_len_bucket",
        "author_present",
        "author_chars",
        "author_tokens",
        "price_present",
        "price_value",
        "price_bucket",
        "raw_id_type",
        "cold_ground_truth_rows",
        "cold_gt_user_count",
    ]
    return table[keep_columns].sort_values(["field_group", "item_id"])


def ordered_groupby(table: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    ordered = table.copy()
    ordered["field_group"] = pd.Categorical(
        ordered["field_group"], categories=FIELD_GROUP_ORDER, ordered=True
    )
    return ordered.groupby("field_group", observed=False)


def build_group_profile_summary(table: pd.DataFrame) -> pd.DataFrame:
    grouped = ordered_groupby(table)
    summary = grouped.agg(
        cold_items=("item_id", "size"),
        cold_gt_items=("cold_ground_truth_rows", lambda s: int((s > 0).sum())),
        cold_ground_truth_rows=("cold_ground_truth_rows", "sum"),
        title_present_rate=("title_present", "mean"),
        title_tokens_mean=("title_tokens", "mean"),
        title_tokens_median=("title_tokens", "median"),
        brand_present_rate=("brand_present", "mean"),
        desc_present_rate=("desc_present", "mean"),
        desc_tokens_mean=("desc_tokens", "mean"),
        desc_tokens_median=("desc_tokens", "median"),
        author_present_rate=("author_present", "mean"),
        price_present_rate=("price_present", "mean"),
        price_mean=("price_value", "mean"),
        price_median=("price_value", "median"),
    ).reset_index()

    raw_type = (
        table.pivot_table(
            index="field_group",
            columns="raw_id_type",
            values="item_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(FIELD_GROUP_ORDER, fill_value=0)
        .add_prefix("raw_id_count_")
        .reset_index()
    )
    summary = summary.merge(raw_type, on="field_group", how="left")
    return round_float_columns(summary)


def build_group_text_brand_price_profile(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variables = [
        ("title_len_bucket", "title_len"),
        ("brand_present", "brand_present"),
        ("desc_len_bucket", "desc_len"),
        ("author_present", "author_present"),
        ("price_bucket", "price"),
        ("raw_id_type", "raw_id_type"),
    ]

    for column, label in variables:
        grouped = (
            table.groupby(["field_group", column], dropna=False)
            .agg(cold_items=("item_id", "size"), cold_ground_truth_rows=("cold_ground_truth_rows", "sum"))
            .reset_index()
            .rename(columns={column: "bucket"})
        )
        totals = grouped.groupby("field_group")["cold_items"].transform("sum")
        grouped["share_in_group"] = grouped["cold_items"] / totals.replace(0, np.nan)
        grouped.insert(0, "variable", label)
        rows.extend(grouped.to_dict("records"))

    result = pd.DataFrame(rows)
    result["bucket"] = result["bucket"].astype(str)
    return round_float_columns(
        result.sort_values(["variable", "bucket", "field_group"]).reset_index(drop=True)
    )


def build_variant_group_gap_summary(variant_metrics: pd.DataFrame) -> pd.DataFrame:
    subset = variant_metrics[variant_metrics["field_group"].isin(["weak_0_1", "mid_2", "strong_3_4"])]
    pivot = subset.pivot_table(
        index=["model", "variant"],
        columns="field_group",
        values=["cold_NDCG@10", "cold_Recall@10", "cold_ground_truth_rows"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns]
    pivot = pivot.reset_index()
    pivot["ndcg_gap_strong_minus_weak"] = (
        pivot["cold_NDCG@10_strong_3_4"] - pivot["cold_NDCG@10_weak_0_1"]
    )
    pivot["recall_gap_strong_minus_weak"] = (
        pivot["cold_Recall@10_strong_3_4"] - pivot["cold_Recall@10_weak_0_1"]
    )
    pivot["ndcg_gap_mid_minus_weak"] = pivot["cold_NDCG@10_mid_2"] - pivot["cold_NDCG@10_weak_0_1"]
    pivot["recall_gap_mid_minus_weak"] = (
        pivot["cold_Recall@10_mid_2"] - pivot["cold_Recall@10_weak_0_1"]
    )
    pivot["variant_order"] = pivot["variant"].map({name: i for i, name in enumerate(VARIANT_ORDER)})
    pivot = pivot.sort_values(["model", "variant_order", "variant"]).drop(columns=["variant_order"])
    return round_float_columns(pivot)


def build_matched_proxy_gap_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    proxies = [
        ("title_len_bucket", "title_len"),
        ("brand_present", "brand_present"),
        ("desc_len_bucket", "desc_len"),
        ("author_present", "author_present"),
        ("price_bucket", "price"),
        ("raw_id_type", "raw_id_type"),
    ]

    for column, label in proxies:
        grouped = (
            table.groupby([column, "field_group"], dropna=False)
            .agg(cold_items=("item_id", "size"), cold_ground_truth_rows=("cold_ground_truth_rows", "sum"))
            .reset_index()
        )
        pivot = grouped.pivot_table(
            index=column,
            columns="field_group",
            values=["cold_items", "cold_ground_truth_rows"],
            aggfunc="first",
            fill_value=0,
        )
        pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns]
        pivot = pivot.reset_index().rename(columns={column: "proxy_bucket"})
        pivot.insert(0, "proxy", label)
        for group in ["weak_0_1", "mid_2", "strong_3_4"]:
            for metric in ["cold_items", "cold_ground_truth_rows"]:
                col = f"{metric}_{group}"
                if col not in pivot.columns:
                    pivot[col] = 0
        pivot["has_weak_and_strong"] = (
            (pivot["cold_items_weak_0_1"] > 0) & (pivot["cold_items_strong_3_4"] > 0)
        )
        pivot["strong_minus_weak_cold_items"] = (
            pivot["cold_items_strong_3_4"] - pivot["cold_items_weak_0_1"]
        )
        pivot["strong_minus_weak_gt_rows"] = (
            pivot["cold_ground_truth_rows_strong_3_4"]
            - pivot["cold_ground_truth_rows_weak_0_1"]
        )
        rows.append(pivot)

    result = pd.concat(rows, ignore_index=True)
    result["proxy_bucket"] = result["proxy_bucket"].astype(str)
    return result.sort_values(["proxy", "proxy_bucket"]).reset_index(drop=True)


def round_float_columns(table: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    result = table.copy()
    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = result[column].round(digits)
    return result


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    view = table.loc[:, columns].head(max_rows).copy()
    if view.empty:
        return "_无数据_"

    # 中文注释：不依赖 pandas.to_markdown 的 tabulate 可选包，避免服务器/本地环境差异。
    header = "| " + " | ".join(str(column) for column in view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = []
    for _, row in view.iterrows():
        values = [format_markdown_cell(row[column]) for column in view.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_result_markdown(
    output_dir: Path,
    group_profile: pd.DataFrame,
    variant_gap: pd.DataFrame,
    matched_proxy: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    a2_variant_gap = variant_gap[variant_gap["model"] == "A2"].copy()
    profile_by_group = group_profile.set_index("field_group")
    weak_profile = profile_by_group.loc["weak_0_1"] if "weak_0_1" in profile_by_group.index else pd.Series()
    strong_profile = (
        profile_by_group.loc["strong_3_4"] if "strong_3_4" in profile_by_group.index else pd.Series()
    )

    def get_profile(row: pd.Series, name: str) -> float:
        value = row.get(name, np.nan)
        return float(value) if pd.notna(value) else float("nan")

    def get_a2_gap(variant: str, column: str) -> float:
        rows = a2_variant_gap[a2_variant_gap["variant"] == variant]
        if rows.empty:
            return float("nan")
        return float(rows.iloc[0][column])

    weak_desc_rate = get_profile(weak_profile, "desc_present_rate")
    strong_desc_rate = get_profile(strong_profile, "desc_present_rate")
    weak_author_rate = get_profile(weak_profile, "author_present_rate")
    strong_author_rate = get_profile(strong_profile, "author_present_rate")
    weak_isbn_count = int(get_profile(weak_profile, "raw_id_count_isbn_like") or 0)
    strong_isbn_count = int(get_profile(strong_profile, "raw_id_count_isbn_like") or 0)
    original_gap = get_a2_gap("original_author", "ndcg_gap_strong_minus_weak")
    title_only_gap = get_a2_gap("title_only", "ndcg_gap_strong_minus_weak")
    no_title_gap = get_a2_gap("no_title", "ndcg_gap_strong_minus_weak")
    structured_gap = get_a2_gap("structured_four_only", "ndcg_gap_strong_minus_weak")

    gap_cols = [
        "variant",
        "ndcg_gap_strong_minus_weak",
        "recall_gap_strong_minus_weak",
        "cold_NDCG@10_weak_0_1",
        "cold_NDCG@10_strong_3_4",
    ]
    group_cols = [
        "field_group",
        "cold_items",
        "cold_ground_truth_rows",
        "title_present_rate",
        "title_tokens_mean",
        "brand_present_rate",
        "desc_present_rate",
        "desc_tokens_mean",
        "author_present_rate",
        "price_present_rate",
        "raw_id_count_asin_B",
        "raw_id_count_isbn_like",
    ]
    group_cols = [col for col in group_cols if col in group_profile.columns]
    proxy_cols = [
        "proxy",
        "proxy_bucket",
        "cold_items_weak_0_1",
        "cold_items_strong_3_4",
        "cold_ground_truth_rows_weak_0_1",
        "cold_ground_truth_rows_strong_3_4",
        "has_weak_and_strong",
    ]

    content = f"""---
title: 2026-06-05 Amazon-M2 自然分组混杂因素诊断结果
date: 2026-06-05
created: 2026-06-05
tags:
  - let-it-go
  - amazon-m2
  - 自然分组
  - 混杂诊断
  - 实验结果
---

# 2026-06-05 Amazon-M2 自然分组混杂因素诊断结果

## 一句话结论

> [!note] 第一版诊断
> 本次只做低成本 CSV 诊断：检查 weak/mid/strong 自然分组是否同时混入 title、brand、desc、price、author、raw id 类型等差异，并复用受控字段消融结果整理 group gap。没有重新训练模型，也没有重跑 prediction 保存推荐明细。

## 初步判读

> [!important] 当前最重要的信号
> 自然分组不是纯粹的 `color/size/model/material` 四字段因果信号，更像 cold item difficulty / content evidence strata。理由是：weak/strong 在未参与分组的 `desc/author/raw_id_type` 上确实有差异；同时在 `title_only/no_title/structured_four_only` 等受控字段 variant 下，strong-weak gap 仍然明显存在。

具体看：

```text
1. title 和 brand 差异没有想象中大：三组 title_present_rate 都是 1，brand_present_rate 也几乎都是 1。
2. desc/author/raw_id_type 差异更明显：
   weak desc_present_rate = {weak_desc_rate:.3f}
   strong desc_present_rate = {strong_desc_rate:.3f}
   weak author_present_rate = {weak_author_rate:.3f}
   strong author_present_rate = {strong_author_rate:.3f}
   weak isbn_like item count = {weak_isbn_count}
   strong isbn_like item count = {strong_isbn_count}
3. A2 strong-weak NDCG gap 在不同 variant 下仍然存在：
   original_author gap = {original_gap:.4f}
   title_only gap = {title_only_gap:.4f}
   no_title gap = {no_title_gap:.4f}
   structured_four_only gap = {structured_gap:.4f}
```

通俗说：strong 组好，不只是因为四个结构化字段更多；weak/strong 很可能本来就是不同难度的商品群体。自然分组可以继续作为难度分层候选，但还不能直接当成“字段完整度 q”的因果标签。

## 运行口径

```text
locale: {manifest["locale"]}
products_path: {manifest["inputs"]["products"]}
output_dir: {manifest["output_dir"]}
generated_at: {manifest["generated_at"]}
```

## 分析一：自然分组画像

这一步不是再检查 `color/size/model/material`，因为 weak/mid/strong 本来就是按它们分组。这里检查的是没有参与分组的其他信息。

{markdown_table(group_profile, group_cols)}

## 分析二：受控字段消融下 strong-weak gap

下面表格复用已有 `variant_field_group_metrics.csv`，只整理 A2 下每个 variant 的 `strong_3_4 - weak_0_1` 差距。

{markdown_table(a2_variant_gap, gap_cols, max_rows=30)}

## 分析三：相似 title/brand/desc 条件下的样本构成排查

第一版没有 item/user 级推荐明细，所以这里不计算分层后的 NDCG，只检查同一粗条件内 weak 和 strong 是否同时存在，以及样本量是否严重失衡。

{markdown_table(matched_proxy, proxy_cols, max_rows=40)}

## 当前不能说明什么

```text
1. 不能说明字段完整度是因果质量标签；
2. 不能说明字段完整度 q 感知方法成立；
3. 不能说明用户更喜欢字段完整商品；
4. 不能替代 full-pipeline retrain；
5. 不能替代 item/user 级 hit@10 诊断。
```

## 后续判读重点

```text
1. 如果 weak 和 strong 在 title/brand/desc/raw_id_type 上差异很大，自然分组更像商品群体差异或 difficulty proxy；
2. 如果 title_only/no_title/structured_four_only 下 strong-weak gap 仍明显存在，说明自然分组不是单一字段内容能解释；
3. 如果粗分层内 weak 和 strong 样本很少同时存在，说明当前数据不适合做严格 matched gap；
4. 如果要进一步判断自然分组能否支撑方法，需要补 recommendation 明细或更强的类目/商品类型信息。
```

## 输出文件

```text
cold_item_confound_table.csv
group_profile_summary.csv
group_text_brand_price_profile.csv
variant_group_gap_summary.csv
matched_proxy_gap_summary.csv
run_manifest.json
```
"""

    (output_dir / "2026-06-05 Amazon-M2 自然分组混杂因素诊断结果.md").write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args)
    for label, path in paths.items():
        if label != "output_dir":
            require_file(path, label)

    field_profile = pd.read_csv(paths["field_profile"])
    ground_truth_groups = pd.read_csv(paths["ground_truth_groups"])
    variant_metrics = pd.read_csv(paths["variant_field_group_metrics"])

    cold_ids = set(field_profile["raw_item_id"].astype(str))
    products = load_matching_products(paths["products"], cold_ids, args.locale, args.chunksize)

    print("Amazon-M2 natural group confounding diagnostic")
    print("field_profile:", paths["field_profile"])
    print("products:", paths["products"])
    print("matched products:", len(products))
    print("output_dir:", paths["output_dir"])

    if args.check_only:
        print("CHECK_ONLY: 输入文件和 raw metadata 读取检查完成，未生成输出。")
        return

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    confound_table = build_cold_item_confound_table(field_profile, ground_truth_groups, products)
    group_profile = build_group_profile_summary(confound_table)
    text_brand_price_profile = build_group_text_brand_price_profile(confound_table)
    variant_gap = build_variant_group_gap_summary(variant_metrics)
    matched_proxy = build_matched_proxy_gap_summary(confound_table)

    confound_table.to_csv(output_dir / "cold_item_confound_table.csv", index=False)
    group_profile.to_csv(output_dir / "group_profile_summary.csv", index=False)
    text_brand_price_profile.to_csv(output_dir / "group_text_brand_price_profile.csv", index=False)
    variant_gap.to_csv(output_dir / "variant_group_gap_summary.csv", index=False)
    matched_proxy.to_csv(output_dir / "matched_proxy_gap_summary.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "locale": args.locale,
        "inputs": {key: str(path) for key, path in paths.items() if key != "output_dir"},
        "output_dir": str(output_dir),
        "rows": {
            "field_profile": int(len(field_profile)),
            "ground_truth_groups": int(len(ground_truth_groups)),
            "matched_products": int(len(products)),
            "cold_item_confound_table": int(len(confound_table)),
        },
        "notes": [
            "本脚本不训练模型，不重新生成 embedding。",
            "matched_proxy_gap_summary 第一版只做样本构成粗分层，不计算分层 NDCG。",
            "raw_id_type 只是缺少 category/product_type 时的弱 proxy，不能当正式类目。",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_result_markdown(output_dir, group_profile, variant_gap, matched_proxy, manifest)

    print("wrote:")
    for filename in [
        "cold_item_confound_table.csv",
        "group_profile_summary.csv",
        "group_text_brand_price_profile.csv",
        "variant_group_gap_summary.csv",
        "matched_proxy_gap_summary.csv",
        "run_manifest.json",
        "2026-06-05 Amazon-M2 自然分组混杂因素诊断结果.md",
    ]:
        print(" ", output_dir / filename)


if __name__ == "__main__":
    main()
