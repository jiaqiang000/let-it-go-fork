"""Amazon-M2 natural weak/mid/strong 全套字段与商品类型画像诊断。

本脚本只做离线 CSV 分析，不训练模型，不重新生成 embedding，不改 run.py/source。
目的：检查 natural weak/mid/strong 的大差距是否混入商品类型、文本结构、
数据来源、字段完整度之外的 proxy，而不是继续把四字段完整度当成因果 q。
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
TEMP_20260605 = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260605"
TEMP_20260606 = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260606"

DEFAULT_CONFOUND_TABLE = TEMP_20260605 / "自然分组混杂因素诊断" / "cold_item_confound_table.csv"
DEFAULT_HIT_DETAIL = TEMP_20260605 / "自然分组命中明细诊断" / "hit_detail_A2.csv"
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
DEFAULT_OUTPUT_DIR = TEMP_20260606 / "natural-group-full-profile-diagnostic"

FIELD_GROUP_ORDER = ["weak_0_1", "mid_2", "strong_3_4", "missing_metadata"]
MISSING_STRINGS = {"", "null", "none", "nan", "[]", "na", "n/a"}
FOUR_FIELDS = ["color_present", "size_present", "model_present", "material_present"]

PROXY_COLUMNS = [
    "raw_id_type",
    "book_like_proxy",
    "book_like_strict",
    "book_like_score_bucket",
    "book_like_source",
    "author_present",
    "publisher_brand_proxy",
    "title_book_terms_proxy",
    "desc_len_bucket",
    "title_len_bucket",
    "text_evidence_bucket",
    "field_pattern",
    "present_field_count",
    "price_bucket",
    "gt_rows_bucket",
]

PUBLISHER_BRAND_RE = re.compile(
    r"hachette|folio|larousse|gallimard|flammarion|nathan|pocket|livre|"
    r"albin michel|robert laffont|hatier|casterman|gl[eé]nat|dargaud|dupuis|"
    r"dunod|eyrolles|puf|seuil|actes sud|j'ai lu|j ai lu|harper|penguin",
    re.IGNORECASE,
)
BOOK_TITLE_RE = re.compile(
    r"bibliocoll[eè]ge|tome|roman|livre|livres|[eé]dition|[eé]ditions|"
    r"folio|poche|manga|manuel|cahier|lecture|contes?|recueil|chapitre|"
    r"harry potter|pok[eé]mon|album|bd\\b|baccalaur[eé]at|bac\\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze natural weak/mid/strong field and product-type profile for Amazon-M2."
    )
    parser.add_argument("--confound-table", type=Path, default=DEFAULT_CONFOUND_TABLE)
    parser.add_argument("--hit-detail", type=Path, default=DEFAULT_HIT_DETAIL)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locale", default="FR")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in MISSING_STRINGS else text


def is_present(value: Any) -> bool:
    return bool(clean_text(value))


def raw_id_type(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "missing"
    if text.startswith("B"):
        return "asin_B"
    if re.fullmatch(r"[0-9X]{10,13}", text):
        return "isbn_like"
    if text.isdigit():
        return "numeric"
    return "other"


def bucket_gt_rows(rows: Any) -> str:
    value = int(rows) if pd.notna(rows) else 0
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2_4"
    return "5_plus"


def bucket_score(score: Any) -> str:
    value = int(score) if pd.notna(score) else 0
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    return "2_plus"


def add_book_like_features(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    for column in ["title", "brand", "author", "raw_item_id"]:
        if column not in result.columns:
            result[column] = ""

    if "raw_id_type" not in result.columns:
        result["raw_id_type"] = result["raw_item_id"].map(raw_id_type)

    result["book_like_raw_id"] = result["raw_id_type"].astype(str).eq("isbn_like")
    result["book_like_author"] = result["author"].map(is_present)
    result["book_like_publisher_brand"] = result["brand"].map(
        lambda text: bool(PUBLISHER_BRAND_RE.search(clean_text(text)))
    )
    result["book_like_title_terms"] = result["title"].map(
        lambda text: bool(BOOK_TITLE_RE.search(clean_text(text)))
    )
    result["book_like_score"] = (
        result[
            [
                "book_like_raw_id",
                "book_like_author",
                "book_like_publisher_brand",
                "book_like_title_terms",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    result["book_like_proxy"] = result["book_like_score"] > 0
    result["book_like_strict"] = (
        result["book_like_raw_id"]
        | result["book_like_author"]
        | result["book_like_publisher_brand"]
    )
    result["publisher_brand_proxy"] = result["book_like_publisher_brand"]
    result["title_book_terms_proxy"] = result["book_like_title_terms"]
    result["book_like_score_bucket"] = result["book_like_score"].map(bucket_score)

    def source(row: pd.Series) -> str:
        labels = []
        if bool(row["book_like_raw_id"]):
            labels.append("raw_id")
        if bool(row["book_like_author"]):
            labels.append("author")
        if bool(row["book_like_publisher_brand"]):
            labels.append("publisher_brand")
        if bool(row["book_like_title_terms"]):
            labels.append("title_terms")
        return "+".join(labels) if labels else "none"

    result["book_like_source"] = result.apply(source, axis=1)
    return result


def load_products_for_items(
    products_path: Path,
    raw_ids: set[str],
    locale: str,
    chunksize: int,
) -> pd.DataFrame:
    usecols = ["id", "locale", "title", "brand", "author", "desc", "price"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        products_path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        # 中文注释：严格使用 FR locale，避免把其他国家的同 id metadata 混进自然分组画像。
        filtered = chunk[(chunk["locale"] == locale) & chunk["id"].isin(raw_ids)]
        if not filtered.empty:
            chunks.append(filtered.copy())
    if not chunks:
        raise ValueError(f"products_train.csv 中没有匹配 locale={locale!r} 的 cold item metadata。")
    return pd.concat(chunks, ignore_index=True).drop_duplicates("id", keep="last")


def build_full_item_profile(confound_table: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    table = confound_table.copy()
    raw_cols = products[["id", "title", "brand", "author", "desc", "price"]].rename(
        columns={
            "id": "raw_item_id",
            "title": "title",
            "brand": "brand",
            "author": "author",
            "desc": "desc",
            "price": "raw_price",
        }
    )
    drop_existing = [column for column in ["title", "brand", "author", "desc", "raw_price"] if column in table.columns]
    table = table.drop(columns=drop_existing, errors="ignore").merge(raw_cols, on="raw_item_id", how="left")
    for column in ["title", "brand", "author", "desc"]:
        table[column] = table[column].fillna("")
    table["brand_norm"] = table["brand"].map(lambda text: clean_text(text).lower() or "missing")
    table["title_norm"] = table["title"].map(lambda text: clean_text(text).lower() or "missing")
    table["desc_present_raw"] = table["desc"].map(is_present)
    table = add_book_like_features(table)

    table["field_pattern"] = table.apply(build_field_pattern, axis=1)
    table["gt_rows_bucket"] = table["cold_ground_truth_rows"].map(bucket_gt_rows)
    table["text_evidence_count"] = (
        table[["title_present", "brand_present", "desc_present", "author_present"]].astype(int).sum(axis=1)
    )
    table["text_evidence_bucket"] = table["text_evidence_count"].map(
        lambda value: "low_0_2" if value <= 2 else ("mid_3" if value == 3 else "high_4")
    )
    return table


def build_field_pattern(row: pd.Series) -> str:
    labels = []
    for column, label in [
        ("color_present", "C"),
        ("size_present", "S"),
        ("model_present", "Mo"),
        ("material_present", "Ma"),
    ]:
        if bool(row.get(column, False)):
            labels.append(label)
    return "+".join(labels) if labels else "none"


def summarize_group_profile(item_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in FIELD_GROUP_ORDER[:-1]:
        group = item_profile[item_profile["field_group"] == group_name]
        rows.append(
            {
                "field_group": group_name,
                "cold_items": len(group),
                "gt_items": int((group["cold_ground_truth_rows"] > 0).sum()) if len(group) else 0,
                "gt_rows": int(group["cold_ground_truth_rows"].sum()) if len(group) else 0,
                "color_rate": mean_bool(group, "color_present"),
                "size_rate": mean_bool(group, "size_present"),
                "model_rate": mean_bool(group, "model_present"),
                "material_rate": mean_bool(group, "material_present"),
                "title_present_rate": mean_bool(group, "title_present"),
                "title_tokens_mean": mean_number(group, "title_tokens"),
                "title_tokens_median": median_number(group, "title_tokens"),
                "brand_present_rate": mean_bool(group, "brand_present"),
                "desc_present_rate": mean_bool(group, "desc_present"),
                "desc_tokens_mean": mean_number(group, "desc_tokens"),
                "desc_tokens_median": median_number(group, "desc_tokens"),
                "author_present_rate": mean_bool(group, "author_present"),
                "isbn_like_rate": float((group["raw_id_type"] == "isbn_like").mean()) if len(group) else 0.0,
                "asin_B_rate": float((group["raw_id_type"] == "asin_B").mean()) if len(group) else 0.0,
                "book_like_proxy_rate": mean_bool(group, "book_like_proxy"),
                "book_like_strict_rate": mean_bool(group, "book_like_strict"),
                "book_like_score_mean": mean_number(group, "book_like_score"),
                "price_median": median_number(group, "price_value"),
                "gt_rows_per_item_mean": mean_number(group, "cold_ground_truth_rows"),
            }
        )
    return round_float_columns(pd.DataFrame(rows))


def summarize_group_profile_gt_weighted(item_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in FIELD_GROUP_ORDER[:-1]:
        group = item_profile[item_profile["field_group"] == group_name].copy()
        weights = group["cold_ground_truth_rows"].fillna(0).astype(float)
        rows.append(
            {
                "field_group": group_name,
                "gt_rows": int(weights.sum()),
                "title_tokens_gt_weighted_mean": weighted_mean(group, "title_tokens", weights),
                "desc_tokens_gt_weighted_mean": weighted_mean(group, "desc_tokens", weights),
                "desc_present_gt_weighted_rate": weighted_mean(group, "desc_present", weights),
                "author_present_gt_weighted_rate": weighted_mean(group, "author_present", weights),
                "isbn_like_gt_weighted_rate": weighted_series((group["raw_id_type"] == "isbn_like"), weights),
                "book_like_proxy_gt_weighted_rate": weighted_mean(group, "book_like_proxy", weights),
                "book_like_strict_gt_weighted_rate": weighted_mean(group, "book_like_strict", weights),
            }
        )
    return round_float_columns(pd.DataFrame(rows))


def mean_bool(table: pd.DataFrame, column: str) -> float:
    return float(table[column].astype(bool).mean()) if len(table) and column in table.columns else 0.0


def mean_number(table: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(table[column], errors="coerce").mean()) if len(table) and column in table.columns else 0.0


def median_number(table: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(table[column], errors="coerce").median()) if len(table) and column in table.columns else 0.0


def weighted_series(values: pd.Series, weights: pd.Series) -> float:
    if weights.sum() <= 0:
        return np.nan
    numeric = values.astype(float)
    return float((numeric * weights).sum() / weights.sum())


def weighted_mean(table: pd.DataFrame, column: str, weights: pd.Series) -> float:
    if column not in table.columns or weights.sum() <= 0:
        return np.nan
    return weighted_series(table[column], weights)


def summarize_proxy_distribution(item_profile: pd.DataFrame, proxies: list[str]) -> pd.DataFrame:
    rows = []
    for proxy in proxies:
        if proxy not in item_profile.columns:
            continue
        data = item_profile.copy()
        data[proxy] = data[proxy].fillna("missing").astype(str)
        grouped = (
            data.groupby(["field_group", proxy], dropna=False)
            .agg(cold_items=("item_id", "size"), gt_rows=("cold_ground_truth_rows", "sum"))
            .reset_index()
            .rename(columns={proxy: "proxy_bucket"})
        )
        totals = grouped.groupby("field_group")["cold_items"].transform("sum").replace(0, np.nan)
        grouped["share_in_group"] = grouped["cold_items"] / totals
        grouped.insert(0, "proxy", proxy)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return round_float_columns(pd.concat(rows, ignore_index=True).sort_values(["proxy", "proxy_bucket", "field_group"]))


def merge_hit_detail_with_profile(hit_detail: pd.DataFrame, item_profile: pd.DataFrame) -> pd.DataFrame:
    profile_cols = [
        "item_id",
        "raw_item_id",
        "brand_norm",
        "field_pattern",
        "gt_rows_bucket",
        "text_evidence_count",
        "text_evidence_bucket",
        "book_like_raw_id",
        "book_like_author",
        "book_like_publisher_brand",
        "book_like_title_terms",
        "book_like_score",
        "book_like_score_bucket",
        "book_like_proxy",
        "book_like_strict",
        "publisher_brand_proxy",
        "title_book_terms_proxy",
        "book_like_source",
    ]
    profile_cols = [column for column in profile_cols if column in item_profile.columns]
    drop_cols = [column for column in profile_cols if column != "item_id" and column in hit_detail.columns]
    merged = hit_detail.drop(columns=drop_cols, errors="ignore").merge(
        item_profile[profile_cols],
        on="item_id",
        how="left",
    )
    return merged


def build_proxy_hit_metrics(hit_detail: pd.DataFrame, proxies: list[str], topk: int) -> pd.DataFrame:
    rows = []
    for proxy in proxies:
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
                    "gt_rows": len(group),
                    "gt_items": group["item_id"].nunique(),
                    "hit_rows": int(group["hit"].sum()),
                    f"Recall@{topk}": float(group[f"recall_contribution@{topk}"].mean()) if len(group) else 0.0,
                    f"NDCG@{topk}": float(group[f"ndcg_contribution@{topk}"].mean()) if len(group) else 0.0,
                }
            )
    return round_float_columns(pd.DataFrame(rows))


def build_proxy_gap_summary(hit_detail: pd.DataFrame, proxies: list[str], topk: int) -> pd.DataFrame:
    metrics = build_proxy_hit_metrics(hit_detail, proxies, topk)
    if metrics.empty:
        return pd.DataFrame()
    pivot = metrics.pivot_table(
        index=["proxy", "proxy_bucket"],
        columns="field_group",
        values=[f"NDCG@{topk}", f"Recall@{topk}", "gt_rows", "gt_items"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns]
    pivot = pivot.reset_index()
    for group in ["weak_0_1", "mid_2", "strong_3_4"]:
        for metric in [f"NDCG@{topk}", f"Recall@{topk}", "gt_rows", "gt_items"]:
            column = f"{metric}_{group}"
            if column not in pivot.columns:
                pivot[column] = np.nan
    pivot["has_weak_and_strong_gt"] = (
        pivot["gt_rows_weak_0_1"].fillna(0) > 0
    ) & (pivot["gt_rows_strong_3_4"].fillna(0) > 0)
    pivot[f"ndcg_gap_strong_minus_weak@{topk}"] = (
        pivot[f"NDCG@{topk}_strong_3_4"] - pivot[f"NDCG@{topk}_weak_0_1"]
    )
    pivot[f"recall_gap_strong_minus_weak@{topk}"] = (
        pivot[f"Recall@{topk}_strong_3_4"] - pivot[f"Recall@{topk}_weak_0_1"]
    )
    pivot[f"ndcg_gap_mid_minus_weak@{topk}"] = (
        pivot[f"NDCG@{topk}_mid_2"] - pivot[f"NDCG@{topk}_weak_0_1"]
    )
    return round_float_columns(pivot.sort_values(["proxy", "proxy_bucket"]).reset_index(drop=True))


def default_subsets(hit_detail: pd.DataFrame) -> dict[str, pd.Series]:
    subsets: dict[str, pd.Series] = {"all": pd.Series(True, index=hit_detail.index)}
    bool_cols = [
        "book_like_proxy",
        "book_like_strict",
        "author_present",
        "desc_present",
        "brand_present",
    ]
    for column in bool_cols:
        if column in hit_detail.columns:
            values = hit_detail[column].fillna(False).astype(bool)
            subsets[f"{column}=True"] = values
            subsets[f"{column}=False"] = ~values
    if "raw_id_type" in hit_detail.columns:
        subsets["raw_id_type=asin_B"] = hit_detail["raw_id_type"].astype(str).eq("asin_B")
        subsets["raw_id_type!=isbn_like"] = ~hit_detail["raw_id_type"].astype(str).eq("isbn_like")
    if "desc_len_bucket" in hit_detail.columns:
        subsets["desc_len_bucket!=missing"] = ~hit_detail["desc_len_bucket"].astype(str).eq("missing")
        subsets["desc_len_bucket=missing"] = hit_detail["desc_len_bucket"].astype(str).eq("missing")
    if "title_len_bucket" in hit_detail.columns:
        subsets["title_len_bucket=long"] = hit_detail["title_len_bucket"].astype(str).eq("long")
        subsets["title_len_bucket!=long"] = ~hit_detail["title_len_bucket"].astype(str).eq("long")
    if {"book_like_proxy", "desc_present"}.issubset(hit_detail.columns):
        subsets["non_book_like_and_desc_present"] = (
            ~hit_detail["book_like_proxy"].fillna(False).astype(bool)
        ) & hit_detail["desc_present"].fillna(False).astype(bool)
    if {"raw_id_type", "desc_present", "author_present"}.issubset(hit_detail.columns):
        subsets["asin_B_desc_present_author_absent"] = (
            hit_detail["raw_id_type"].astype(str).eq("asin_B")
            & hit_detail["desc_present"].fillna(False).astype(bool)
            & ~hit_detail["author_present"].fillna(False).astype(bool)
        )
    return subsets


def summarize_subset_metrics(
    hit_detail: pd.DataFrame,
    subsets: dict[str, pd.Series],
    topk: int,
) -> pd.DataFrame:
    rows = []
    for subset_name, mask in subsets.items():
        data = hit_detail[mask].copy()
        row: dict[str, Any] = {
            "subset": subset_name,
            "total_gt_rows": len(data),
            "total_gt_items": data["item_id"].nunique() if len(data) else 0,
        }
        for group_name in FIELD_GROUP_ORDER[:-1]:
            group = data[data["field_group"] == group_name]
            row[f"gt_rows_{group_name}"] = len(group)
            row[f"gt_items_{group_name}"] = group["item_id"].nunique() if len(group) else 0
            row[f"Recall@{topk}_{group_name}"] = (
                float(group[f"recall_contribution@{topk}"].mean()) if len(group) else np.nan
            )
            row[f"NDCG@{topk}_{group_name}"] = (
                float(group[f"ndcg_contribution@{topk}"].mean()) if len(group) else np.nan
            )
            # 兼容旧命名，方便人工读表。
            row[f"cold_Recall@{topk}_{group_name}"] = row[f"Recall@{topk}_{group_name}"]
            row[f"cold_NDCG@{topk}_{group_name}"] = row[f"NDCG@{topk}_{group_name}"]

        row[f"ndcg_gap_strong_minus_weak@{topk}"] = (
            row[f"NDCG@{topk}_strong_3_4"] - row[f"NDCG@{topk}_weak_0_1"]
        )
        row[f"recall_gap_strong_minus_weak@{topk}"] = (
            row[f"Recall@{topk}_strong_3_4"] - row[f"Recall@{topk}_weak_0_1"]
        )
        row[f"ndcg_gap_mid_minus_weak@{topk}"] = (
            row[f"NDCG@{topk}_mid_2"] - row[f"NDCG@{topk}_weak_0_1"]
        )
        rows.append(row)
    return round_float_columns(pd.DataFrame(rows))


def summarize_top_values(item_profile: pd.DataFrame, column: str, topn: int = 20) -> pd.DataFrame:
    data = item_profile.copy()
    data[column] = data[column].fillna("missing").astype(str)
    grouped = (
        data.groupby(["field_group", column], dropna=False)
        .agg(cold_items=("item_id", "size"), gt_rows=("cold_ground_truth_rows", "sum"))
        .reset_index()
        .rename(columns={column: "value"})
    )
    grouped["share_in_group"] = grouped["cold_items"] / grouped.groupby("field_group")["cold_items"].transform("sum")
    return round_float_columns(
        grouped.sort_values(["field_group", "cold_items", "gt_rows"], ascending=[True, False, False])
        .groupby("field_group", group_keys=False)
        .head(topn)
    )


def build_examples(item_profile: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "field_group",
        "book_like_proxy",
        "book_like_source",
        "raw_item_id",
        "item_id",
        "cold_ground_truth_rows",
        "title",
        "brand",
        "author",
        "desc_len_bucket",
        "title_len_bucket",
        "field_pattern",
    ]
    rows = []
    for group_name in FIELD_GROUP_ORDER[:-1]:
        group = item_profile[item_profile["field_group"] == group_name].copy()
        for book_like in [True, False]:
            subset = group[group["book_like_proxy"].astype(bool) == book_like]
            if subset.empty:
                continue
            rows.append(
                subset.sort_values(["cold_ground_truth_rows", "raw_item_id"], ascending=[False, True])
                .head(10)[keep]
            )
    if not rows:
        return pd.DataFrame(columns=keep)
    return pd.concat(rows, ignore_index=True)


def round_float_columns(table: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    result = table.copy()
    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = result[column].round(digits)
    return result


def dataframe_to_markdown(table: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if table.empty:
        return "_无数据_"
    view = table.copy()
    if columns is not None:
        view = view[[column for column in columns if column in view.columns]]
    view = view.head(max_rows)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(format_cell(row[col]) for col in view.columns) + " |")
    return "\n".join([header, sep, *rows])


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_result_markdown(
    output_dir: Path,
    group_profile: pd.DataFrame,
    gt_weighted: pd.DataFrame,
    subset_summary: pd.DataFrame,
    proxy_gaps: pd.DataFrame,
    top_brands: pd.DataFrame,
    manifest: dict[str, Any],
    topk: int,
) -> None:
    timestamp = manifest["generated_at"].replace("T", " ").replace(":", "")[:17]
    subset_view = subset_summary.sort_values(
        [f"ndcg_gap_strong_minus_weak@{topk}", "total_gt_rows"],
        ascending=[False, False],
    )
    comparable_proxy = proxy_gaps[proxy_gaps["has_weak_and_strong_gt"].fillna(False).astype(bool)].copy()
    comparable_proxy = comparable_proxy.sort_values(
        [f"ndcg_gap_strong_minus_weak@{topk}", "gt_rows_weak_0_1"],
        ascending=[False, False],
    )
    profile = group_profile.set_index("field_group")
    weak_book = float(profile.loc["weak_0_1", "book_like_proxy_rate"]) if "weak_0_1" in profile.index else np.nan
    strong_book = float(profile.loc["strong_3_4", "book_like_proxy_rate"]) if "strong_3_4" in profile.index else np.nan
    weak_desc = float(profile.loc["weak_0_1", "desc_present_rate"]) if "weak_0_1" in profile.index else np.nan
    strong_desc = float(profile.loc["strong_3_4", "desc_present_rate"]) if "strong_3_4" in profile.index else np.nan

    content = f"""---
title: 2026-06-06 Amazon-M2 natural group 商品类型与字段画像全套诊断结果
date: 2026-06-06
created: 2026-06-06
created_at: {manifest["generated_at"]}
tags:
  - let-it-go
  - amazon-m2
  - natural-group
  - 商品类型
  - 混杂诊断
---

# 2026-06-06 Amazon-M2 natural group 商品类型与字段画像全套诊断结果

## 一句话结论

> [!important] 严格解释边界
> 本次全套诊断不是证明“字段完整度导致推荐失败”，而是检查 natural weak / mid / strong 是否混入商品类型、文本结构、数据来源和命中样本差异。结果应作为机制诊断，不应直接变成 q-aware 方法变量。

当前最值得注意的是：

```text
weak book_like_proxy_rate = {weak_book:.4f}
strong book_like_proxy_rate = {strong_book:.4f}
weak desc_present_rate = {weak_desc:.4f}
strong desc_present_rate = {strong_desc:.4f}
```

通俗说：如果 weak 的 book-like / author / ISBN / 出版社品牌 proxy 明显更多，那 weak 组就不只是“字段少”，还可能是混进了不同商品类型或数据来源。

## 字段说明

> [!note] 关键字段
> `book_like_proxy`：只要命中 ISBN-like raw id、author 存在、publisher-like brand、book-like title terms 任一条件，就认为可能是书籍/出版物类 proxy。
> `book_like_strict`：只使用 ISBN-like raw id、author、publisher-like brand，不使用 title terms，较保守。
> `field_pattern`：color / size / model / material 四字段的存在组合。
> `subset_*`：不是重新训练，只是在已有 A2 hit detail 中筛选子集后重新算 NDCG/Recall。

## 1. 三组字段与商品类型画像

{dataframe_to_markdown(group_profile, max_rows=20)}

## 2. GT 加权画像

这张表只按有 ground-truth 的用户-物品记录加权，避免大量无 GT 的 cold item 影响判断。

{dataframe_to_markdown(gt_weighted, max_rows=20)}

## 3. 剔除/保留不同子集后的 weak/strong gap

如果剔除 book-like、ISBN-like、author-present 后 strong-weak gap 明显缩小，说明原始 natural gap 可能被商品类型混杂放大。如果 gap 仍然大，说明商品类型只能解释一部分。

{dataframe_to_markdown(subset_view, ["subset", "total_gt_rows", "gt_rows_weak_0_1", "gt_rows_mid_2", "gt_rows_strong_3_4", f"NDCG@{topk}_weak_0_1", f"NDCG@{topk}_mid_2", f"NDCG@{topk}_strong_3_4", f"ndcg_gap_strong_minus_weak@{topk}", f"recall_gap_strong_minus_weak@{topk}"], max_rows=30)}

## 4. proxy bucket 内部 strong-weak gap

这一步看“在相似 proxy bucket 内 strong 是否仍然更好”。它不是严格因果匹配，但比只看总体均值更稳。

{dataframe_to_markdown(comparable_proxy, ["proxy", "proxy_bucket", "gt_rows_weak_0_1", "gt_rows_strong_3_4", f"NDCG@{topk}_weak_0_1", f"NDCG@{topk}_strong_3_4", f"ndcg_gap_strong_minus_weak@{topk}"], max_rows=40)}

## 5. 各组 Top brand

{dataframe_to_markdown(top_brands, ["field_group", "value", "cold_items", "gt_rows", "share_in_group"], max_rows=45)}

## 当前能说明什么

```text
1. 可以判断 natural weak/mid/strong 是否混入商品类型和文本结构差异；
2. 可以判断 strong-weak gap 在剔除某些 proxy 后是否仍然存在；
3. 可以为后续是否继续把 natural group 当 difficulty strata 提供依据。
```

## 当前不能说明什么

```text
1. 不能证明 book-like item 本身导致推荐失败；
2. 不能证明字段完整度是因果质量标签；
3. 不能直接推出一个 q-aware / gate / delta 方法；
4. 不能替代真正的类目字段或人工商品类型标注。
```

## 输出文件

```text
natural_group_full_item_profile.csv
group_completeness_length_profile.csv
group_gt_weighted_profile.csv
proxy_distribution_by_group.csv
proxy_hit_metrics.csv
proxy_gap_within_bucket.csv
subset_recomputed_group_metrics.csv
top_brand_by_group.csv
diagnostic_examples.csv
run_manifest.json
```
"""
    (output_dir / f"{timestamp} Amazon-M2 natural group 商品类型与字段画像全套诊断结果.md").write_text(
        content,
        encoding="utf-8",
    )


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{label}: {path}")
    return path


def main() -> None:
    args = parse_args()
    confound_path = require_file(args.confound_table, "confound_table")
    hit_detail_path = require_file(args.hit_detail, "hit_detail")
    products_path = require_file(args.products_path, "products_train.csv")
    output_dir = args.output_dir.expanduser().resolve()

    confound = pd.read_csv(confound_path)
    hit_detail = pd.read_csv(hit_detail_path)
    raw_ids = set(confound["raw_item_id"].astype(str))

    print("Amazon-M2 natural group full profile diagnostic")
    print("confound_table:", confound_path)
    print("hit_detail:", hit_detail_path)
    print("products:", products_path)
    print("output_dir:", output_dir)
    print("cold_items:", len(confound))
    print("hit_detail_rows:", len(hit_detail))

    if args.check_only:
        print("CHECK_ONLY: 输入文件检查通过；未生成结果。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    products = load_products_for_items(products_path, raw_ids, args.locale, args.chunksize)
    item_profile = build_full_item_profile(confound, products)
    hit_profile = merge_hit_detail_with_profile(hit_detail, item_profile)

    group_profile = summarize_group_profile(item_profile)
    gt_weighted = summarize_group_profile_gt_weighted(item_profile)
    proxy_distribution = summarize_proxy_distribution(item_profile, PROXY_COLUMNS)
    proxy_metrics = build_proxy_hit_metrics(hit_profile, PROXY_COLUMNS, args.topk)
    proxy_gaps = build_proxy_gap_summary(hit_profile, PROXY_COLUMNS, args.topk)
    subset_summary = summarize_subset_metrics(hit_profile, default_subsets(hit_profile), args.topk)
    top_brands = summarize_top_values(item_profile, "brand_norm", topn=15)
    examples = build_examples(item_profile)

    item_profile.to_csv(output_dir / "natural_group_full_item_profile.csv", index=False)
    group_profile.to_csv(output_dir / "group_completeness_length_profile.csv", index=False)
    gt_weighted.to_csv(output_dir / "group_gt_weighted_profile.csv", index=False)
    proxy_distribution.to_csv(output_dir / "proxy_distribution_by_group.csv", index=False)
    proxy_metrics.to_csv(output_dir / "proxy_hit_metrics.csv", index=False)
    proxy_gaps.to_csv(output_dir / "proxy_gap_within_bucket.csv", index=False)
    subset_summary.to_csv(output_dir / "subset_recomputed_group_metrics.csv", index=False)
    top_brands.to_csv(output_dir / "top_brand_by_group.csv", index=False)
    examples.to_csv(output_dir / "diagnostic_examples.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "locale": args.locale,
        "topk": args.topk,
        "output_dir": str(output_dir),
        "inputs": {
            "confound_table": str(confound_path),
            "hit_detail": str(hit_detail_path),
            "products": str(products_path),
        },
        "rows": {
            "cold_items": int(len(confound)),
            "hit_detail_rows": int(len(hit_detail)),
            "matched_products": int(len(products)),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_result_markdown(
        output_dir=output_dir,
        group_profile=group_profile,
        gt_weighted=gt_weighted,
        subset_summary=subset_summary,
        proxy_gaps=proxy_gaps,
        top_brands=top_brands,
        manifest=manifest,
        topk=args.topk,
    )
    print("DONE: natural group full profile diagnostic outputs 已生成。")


if __name__ == "__main__":
    main()
