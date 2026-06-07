"""Amazon-M2 response-selector 可行性诊断。

本脚本是一个严格的事后诊断脚本，不训练模型，也不修改 Let It Go 推荐器。
它读取已经完成的 generated cold-delta / oracle 记录，检查不同 cold item
对 generated delta 的响应差异是否足够大、且是否能被可观测特征预测。

注意：这里的正向 selector 结果只说明“已有日志里存在可预测的响应异质性”，
不能直接证明未来 strict cold item 也能安全使用 generated delta。因此本脚本
只用于判断是否值得继续做 response-selector 方法线，不应被解释成最终方法结果。
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
TEMP_20260606 = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260606"
TEMP_20260607 = RESEARCH_ROOT / "temp_202606_实验文件记录" / "temp_20260607"
DEFAULT_INPUT_DIR = TEMP_20260606 / "correction-safe-subset-oracle"
DEFAULT_OUTPUT_DIR = TEMP_20260607 / "response-selector-feasibility"

KEY_COLUMNS = ["user_id", "item_id"]
ITEM_KEY_COLUMNS = ["item_id"]
CAT_FEATURES = ["field_group", "brand_present", "author_present", "metadata_found"]
NUM_FEATURES = [
    "present_field_count",
    "nearest_warm_cosine",
    "neighbor_cosine_mean",
    "title_len",
    "cold_content_norm",
]
SELECTION_FRACTIONS = [0.05, 0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze response-selector feasibility from existing Amazon-M2 oracle deltas."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def bool_to_string(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column].astype(str)
    return frame


def load_oracle_records(input_dir: Path) -> pd.DataFrame:
    path = require_file(input_dir / "oracle_record_delta.csv", "oracle_record_delta.csv")
    records = pd.read_csv(path)
    required = set(KEY_COLUMNS + CAT_FEATURES + NUM_FEATURES + ["group", "a2_ndcg", "ndcg", "delta_ndcg"])
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"oracle_record_delta.csv missing columns: {missing}")
    return records


def build_fixed_feature_table(records: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "a2_ndcg": ("a2_ndcg", "first"),
        "field_group": ("field_group", "first"),
        "present_field_count": ("present_field_count", "first"),
        "nearest_warm_cosine": ("nearest_warm_cosine", "max"),
        "neighbor_cosine_mean": ("neighbor_cosine_mean", "max"),
        "title_len": ("title_len", "first"),
        "brand_present": ("brand_present", "first"),
        "author_present": ("author_present", "first"),
        "metadata_found": ("metadata_found", "first"),
        "cold_content_norm": ("cold_content_norm", "first"),
    }
    return records.groupby(KEY_COLUMNS).agg(**aggregations).reset_index()


def compute_oracle_gap(records: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_generated = (
        records.sort_values(KEY_COLUMNS + ["ndcg"], ascending=[True, True, False])
        .groupby(KEY_COLUMNS)
        .head(1)[KEY_COLUMNS + ["group", "ndcg", "delta_ndcg"]]
        .rename(
            columns={
                "group": "best_generated_group",
                "ndcg": "best_generated_ndcg",
                "delta_ndcg": "best_generated_delta",
            }
        )
    )
    per_record = features.merge(best_generated, on=KEY_COLUMNS, how="left")
    per_record["oracle_gain"] = np.maximum(per_record["best_generated_delta"], 0.0)
    per_record["oracle_positive"] = per_record["best_generated_delta"] > 0
    per_record["oracle_ndcg"] = per_record["a2_ndcg"] + per_record["oracle_gain"]
    summary = pd.DataFrame(
        [
            {
                "records": len(per_record),
                "a2_mean_ndcg": per_record["a2_ndcg"].mean(),
                "oracle_mean_ndcg": per_record["oracle_ndcg"].mean(),
                "oracle_mean_gain": per_record["oracle_gain"].mean(),
                "oracle_positive_rate": per_record["oracle_positive"].mean(),
                "oracle_positive_records": int(per_record["oracle_positive"].sum()),
                "oracle_zero_or_negative_rate": 1.0 - per_record["oracle_positive"].mean(),
                "oracle_gain_p50": per_record["oracle_gain"].quantile(0.50),
                "oracle_gain_p90": per_record["oracle_gain"].quantile(0.90),
                "oracle_gain_p95": per_record["oracle_gain"].quantile(0.95),
            }
        ]
    )
    return summary, per_record


def compute_topk_alpha_oracle(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topk, topk_records in records.groupby("topk", sort=True):
        fixed_rows = []
        for alpha, alpha_records in topk_records.groupby("alpha", sort=True):
            fixed_rows.append(
                {
                    "topk": int(topk),
                    "policy": f"fixed_alpha_{alpha:g}",
                    "alpha": float(alpha),
                    "records": int(len(alpha_records)),
                    "mean_delta_ndcg": float(alpha_records["delta_ndcg"].mean()),
                    "positive_rate": float((alpha_records["delta_ndcg"] > 0).mean()),
                    "negative_rate": float((alpha_records["delta_ndcg"] < 0).mean()),
                }
            )
        fixed_table = pd.DataFrame(fixed_rows)
        best_fixed = fixed_table.sort_values("mean_delta_ndcg", ascending=False).head(1)

        best_per_record = (
            topk_records.sort_values(KEY_COLUMNS + ["delta_ndcg"], ascending=[True, True, False])
            .groupby(KEY_COLUMNS)
            .head(1)
        )
        oracle_gain = np.maximum(best_per_record["delta_ndcg"].to_numpy(dtype=float), 0.0)
        rows.extend(fixed_rows)
        rows.append(
            {
                "topk": int(topk),
                "policy": "oracle_choose_alpha_or_a2",
                "alpha": np.nan,
                "records": int(len(best_per_record)),
                "mean_delta_ndcg": float(oracle_gain.mean()),
                "positive_rate": float((oracle_gain > 0).mean()),
                "negative_rate": 0.0,
                "best_fixed_policy": best_fixed["policy"].iloc[0],
                "oracle_minus_best_fixed": float(oracle_gain.mean() - best_fixed["mean_delta_ndcg"].iloc[0]),
                "nonzero_alpha_chosen_rate": float((best_per_record["delta_ndcg"] > 0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    for column in ["best_fixed_policy", "oracle_minus_best_fixed", "nonzero_alpha_chosen_rate"]:
        if column not in result.columns:
            result[column] = np.nan
    return result.sort_values(["topk", "policy"])


def feature_rule_summary(per_record: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in NUM_FEATURES:
        values = per_record[feature].astype(float)
        for direction, ascending in [("high", False), ("low", True)]:
            for fraction in SELECTION_FRACTIONS:
                k = max(1, int(round(len(per_record) * fraction)))
                selected_index = values.sort_values(ascending=ascending).index[:k]
                selected = per_record.loc[selected_index]
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "fraction": fraction,
                        "selected_records": int(len(selected)),
                        "precision": float(selected["oracle_positive"].mean()),
                        "mean_oracle_gain": float(selected["oracle_gain"].mean()),
                        "baseline_positive_rate": float(per_record["oracle_positive"].mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["precision", "mean_oracle_gain"], ascending=[False, False]
    )


def cross_validated_probabilities(
    data: pd.DataFrame,
    label: pd.Series,
    random_state: int,
    n_splits: int,
) -> dict[str, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    x = bool_to_string(data[CAT_FEATURES + NUM_FEATURES], CAT_FEATURES)
    y = label.astype(int).to_numpy()
    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)
    splits = min(n_splits, positive_count, negative_count)
    if splits < 2:
        return {}

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


def evaluate_ranked_selection(
    records: pd.DataFrame,
    scores: np.ndarray,
    delta_column: str,
    label_column: str,
    model_name: str,
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = np.argsort(scores)[::-1]
    base_delta = records[delta_column].astype(float).to_numpy()
    labels = records[label_column].astype(bool).to_numpy()
    for fraction in SELECTION_FRACTIONS:
        k = max(1, int(round(len(records) * fraction)))
        selected_idx = order[:k]
        selected_delta = base_delta[selected_idx]
        policy_gain = float(selected_delta.sum() / len(records))
        rows.append(
            {
                "variant": variant,
                "model": model_name,
                "fraction": fraction,
                "selected_records": int(k),
                "positive_rate": float(labels[selected_idx].mean()),
                "selected_mean_delta": float(selected_delta.mean()),
                "policy_mean_gain_vs_a2": policy_gain,
                "random_same_fraction_expected_gain": float(fraction * base_delta.mean()),
                "variant_fixed_mean_delta": float(base_delta.mean()),
            }
        )
    return rows


def compute_variant_gate_cv(
    records: pd.DataFrame,
    features: pd.DataFrame,
    random_state: int,
    n_splits: int,
    min_positive: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_features = features[KEY_COLUMNS + CAT_FEATURES + NUM_FEATURES]
    for variant, variant_rows in records.groupby("group", sort=True):
        variant_frame = base_features.merge(
            variant_rows[KEY_COLUMNS + ["delta_ndcg"]],
            on=KEY_COLUMNS,
            how="inner",
        )
        variant_frame["positive"] = variant_frame["delta_ndcg"] > 0
        positive_count = int(variant_frame["positive"].sum())
        if positive_count < min_positive:
            rows.append(
                {
                    "variant": variant,
                    "model": "skipped",
                    "fraction": np.nan,
                    "selected_records": 0,
                    "positive_rate": np.nan,
                    "selected_mean_delta": np.nan,
                    "policy_mean_gain_vs_a2": np.nan,
                    "random_same_fraction_expected_gain": np.nan,
                    "variant_fixed_mean_delta": float(variant_frame["delta_ndcg"].mean()),
                    "skip_reason": f"positive_count<{min_positive}",
                }
            )
            continue

        probabilities = cross_validated_probabilities(
            variant_frame,
            variant_frame["positive"],
            random_state=random_state,
            n_splits=n_splits,
        )
        for model_name, score in probabilities.items():
            rows.extend(
                evaluate_ranked_selection(
                    records=variant_frame,
                    scores=score,
                    delta_column="delta_ndcg",
                    label_column="positive",
                    model_name=model_name,
                    variant=variant,
                )
            )
    result = pd.DataFrame(rows)
    if "skip_reason" not in result.columns:
        result["skip_reason"] = ""
    else:
        result["skip_reason"] = result["skip_reason"].fillna("")
    return result.sort_values(
        ["policy_mean_gain_vs_a2", "selected_mean_delta"],
        ascending=[False, False],
        na_position="last",
    )


def build_item_level_tables(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_features = (
        records.groupby(ITEM_KEY_COLUMNS)
        .agg(
            a2_ndcg=("a2_ndcg", "mean"),
            field_group=("field_group", "first"),
            present_field_count=("present_field_count", "first"),
            nearest_warm_cosine=("nearest_warm_cosine", "max"),
            neighbor_cosine_mean=("neighbor_cosine_mean", "max"),
            title_len=("title_len", "first"),
            brand_present=("brand_present", "first"),
            author_present=("author_present", "first"),
            metadata_found=("metadata_found", "first"),
            cold_content_norm=("cold_content_norm", "first"),
            ground_truth_records=("user_id", "nunique"),
        )
        .reset_index()
    )
    item_variant = (
        records.groupby(ITEM_KEY_COLUMNS + ["group"])
        .agg(
            delta_ndcg=("delta_ndcg", "mean"),
            ndcg=("ndcg", "mean"),
            a2_ndcg=("a2_ndcg", "mean"),
            positive_records=("delta_ndcg", lambda values: int((values > 0).sum())),
            negative_records=("delta_ndcg", lambda values: int((values < 0).sum())),
            records=("delta_ndcg", "size"),
        )
        .reset_index()
    )
    return item_features, item_variant


def compute_item_oracle_gap(
    item_variant: pd.DataFrame,
    item_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_generated = (
        item_variant.sort_values(ITEM_KEY_COLUMNS + ["delta_ndcg"], ascending=[True, False])
        .groupby(ITEM_KEY_COLUMNS)
        .head(1)[ITEM_KEY_COLUMNS + ["group", "delta_ndcg", "ndcg"]]
        .rename(
            columns={
                "group": "best_generated_group",
                "delta_ndcg": "best_generated_delta",
                "ndcg": "best_generated_ndcg",
            }
        )
    )
    per_item = item_features.merge(best_generated, on=ITEM_KEY_COLUMNS, how="left")
    per_item["oracle_gain"] = np.maximum(per_item["best_generated_delta"], 0.0)
    per_item["oracle_positive"] = per_item["best_generated_delta"] > 0
    per_item["oracle_ndcg"] = per_item["a2_ndcg"] + per_item["oracle_gain"]
    summary = pd.DataFrame(
        [
            {
                "items": len(per_item),
                "ground_truth_records": int(per_item["ground_truth_records"].sum()),
                "a2_mean_item_ndcg": per_item["a2_ndcg"].mean(),
                "oracle_mean_item_ndcg": per_item["oracle_ndcg"].mean(),
                "oracle_mean_gain": per_item["oracle_gain"].mean(),
                "oracle_positive_item_rate": per_item["oracle_positive"].mean(),
                "oracle_positive_items": int(per_item["oracle_positive"].sum()),
                "oracle_gain_p50": per_item["oracle_gain"].quantile(0.50),
                "oracle_gain_p90": per_item["oracle_gain"].quantile(0.90),
                "oracle_gain_p95": per_item["oracle_gain"].quantile(0.95),
            }
        ]
    )
    return summary, per_item


def compute_item_variant_gate_cv(
    item_variant: pd.DataFrame,
    item_features: pd.DataFrame,
    random_state: int,
    n_splits: int,
    min_positive: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_features = item_features[ITEM_KEY_COLUMNS + CAT_FEATURES + NUM_FEATURES]
    for variant, variant_rows in item_variant.groupby("group", sort=True):
        variant_frame = base_features.merge(
            variant_rows[ITEM_KEY_COLUMNS + ["delta_ndcg"]],
            on=ITEM_KEY_COLUMNS,
            how="inner",
        )
        variant_frame["positive"] = variant_frame["delta_ndcg"] > 0
        positive_count = int(variant_frame["positive"].sum())
        if positive_count < min_positive:
            rows.append(
                {
                    "variant": variant,
                    "model": "skipped",
                    "fraction": np.nan,
                    "selected_records": 0,
                    "positive_rate": np.nan,
                    "selected_mean_delta": np.nan,
                    "policy_mean_gain_vs_a2": np.nan,
                    "random_same_fraction_expected_gain": np.nan,
                    "variant_fixed_mean_delta": float(variant_frame["delta_ndcg"].mean()),
                    "skip_reason": f"positive_count<{min_positive}",
                }
            )
            continue
        probabilities = cross_validated_probabilities(
            variant_frame,
            variant_frame["positive"],
            random_state=random_state,
            n_splits=n_splits,
        )
        for model_name, score in probabilities.items():
            rows.extend(
                evaluate_ranked_selection(
                    records=variant_frame,
                    scores=score,
                    delta_column="delta_ndcg",
                    label_column="positive",
                    model_name=model_name,
                    variant=variant,
                )
            )
    result = pd.DataFrame(rows)
    if "skip_reason" not in result.columns:
        result["skip_reason"] = ""
    else:
        result["skip_reason"] = result["skip_reason"].fillna("")
    return result.sort_values(
        ["policy_mean_gain_vs_a2", "selected_mean_delta"],
        ascending=[False, False],
        na_position="last",
    )


def compute_any_oracle_cv(
    per_record: pd.DataFrame,
    random_state: int,
    n_splits: int,
) -> pd.DataFrame:
    probabilities = cross_validated_probabilities(
        per_record,
        per_record["oracle_positive"],
        random_state=random_state,
        n_splits=n_splits,
    )
    rows: list[dict[str, Any]] = []
    for model_name, score in probabilities.items():
        rows.extend(
            evaluate_ranked_selection(
                records=per_record,
                scores=score,
                delta_column="oracle_gain",
                label_column="oracle_positive",
                model_name=model_name,
                variant="oracle_any_generated",
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["policy_mean_gain_vs_a2", "selected_mean_delta"], ascending=[False, False]
    )


def write_report(
    output_dir: Path,
    oracle_summary: pd.DataFrame,
    topk_alpha_oracle: pd.DataFrame,
    rule_summary: pd.DataFrame,
    any_oracle_cv: pd.DataFrame,
    variant_gate_cv: pd.DataFrame,
    item_oracle_summary: pd.DataFrame,
    item_any_oracle_cv: pd.DataFrame,
    item_variant_gate_cv: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        text_frame = frame.copy()
        for column in text_frame.columns:
            if pd.api.types.is_float_dtype(text_frame[column]):
                text_frame[column] = text_frame[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
                )
            else:
                text_frame[column] = text_frame[column].map(
                    lambda value: "" if pd.isna(value) else str(value)
                )
        headers = list(text_frame.columns)
        rows = text_frame.values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    best_variant_gate = variant_gate_cv.dropna(subset=["policy_mean_gain_vs_a2"]).head(1)
    best_any_gate = any_oracle_cv.head(1)
    best_item_variant_gate = item_variant_gate_cv.dropna(subset=["policy_mean_gain_vs_a2"]).head(1)
    best_item_any_gate = item_any_oracle_cv.head(1)
    best_rule = rule_summary.head(1)
    oracle_alpha_rows = topk_alpha_oracle[
        topk_alpha_oracle["policy"] == "oracle_choose_alpha_or_a2"
    ]
    lines = [
        "# Amazon-M2 response selector feasibility",
        "",
        "## Oracle gap",
        "",
        markdown_table(oracle_summary),
        "",
        "## TopK alpha oracle gap",
        "",
        markdown_table(oracle_alpha_rows),
        "",
        "## Best simple feature rule",
        "",
        markdown_table(best_rule),
        "",
        "## Best cross-validated selector over oracle-any labels",
        "",
        markdown_table(best_any_gate),
        "",
        "## Best cross-validated selector over deployable fixed variants",
        "",
        markdown_table(best_variant_gate) if len(best_variant_gate) else "No valid variant gate.",
        "",
        "## Item-level oracle gap",
        "",
        markdown_table(item_oracle_summary),
        "",
        "## Best item-level selector over oracle-any labels",
        "",
        markdown_table(best_item_any_gate),
        "",
        "## Best item-level selector over deployable fixed variants",
        "",
        markdown_table(best_item_variant_gate) if len(best_item_variant_gate) else "No valid item-level variant gate.",
        "",
        "## Gate decision notes",
        "",
        "- This is a feasibility diagnostic over existing generated-delta variants, not a final local-response experiment.",
        "- Positive policy gain here only means that the logged generated-delta probe contains predictable response heterogeneity.",
        "- It does not prove that content/CF endpoint interpolation will transfer to future strict cold items.",
    ]
    (output_dir / "2026-06-07 Amazon-M2 response-selector feasibility 结果.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_oracle_records(input_dir)
    features = build_fixed_feature_table(records)
    oracle_summary, per_record = compute_oracle_gap(records, features)
    topk_alpha_oracle = compute_topk_alpha_oracle(records)
    rules = feature_rule_summary(per_record)
    any_oracle_cv = compute_any_oracle_cv(
        per_record,
        random_state=args.random_state,
        n_splits=args.n_splits,
    )
    variant_gate_cv = compute_variant_gate_cv(
        records,
        features,
        random_state=args.random_state,
        n_splits=args.n_splits,
        min_positive=args.min_positive,
    )
    item_features, item_variant = build_item_level_tables(records)
    item_oracle_summary, per_item = compute_item_oracle_gap(item_variant, item_features)
    item_any_oracle_cv = compute_any_oracle_cv(
        per_item,
        random_state=args.random_state,
        n_splits=args.n_splits,
    )
    item_variant_gate_cv = compute_item_variant_gate_cv(
        item_variant,
        item_features,
        random_state=args.random_state,
        n_splits=args.n_splits,
        min_positive=args.min_positive,
    )

    oracle_summary.to_csv(output_dir / "oracle_gap_summary.csv", index=False)
    topk_alpha_oracle.to_csv(output_dir / "topk_alpha_oracle_summary.csv", index=False)
    per_record.to_csv(output_dir / "oracle_per_record_best_generated.csv", index=False)
    rules.to_csv(output_dir / "feature_rule_summary.csv", index=False)
    any_oracle_cv.to_csv(output_dir / "oracle_any_selector_cv_summary.csv", index=False)
    variant_gate_cv.to_csv(output_dir / "variant_gate_cv_summary.csv", index=False)
    item_oracle_summary.to_csv(output_dir / "item_level_oracle_gap_summary.csv", index=False)
    per_item.to_csv(output_dir / "item_level_oracle_per_item_best_generated.csv", index=False)
    item_any_oracle_cv.to_csv(output_dir / "item_level_oracle_any_selector_cv_summary.csv", index=False)
    item_variant_gate_cv.to_csv(output_dir / "item_level_variant_gate_cv_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "oracle_record_delta": str(input_dir / "oracle_record_delta.csv"),
        "records": int(len(per_record)),
        "items": int(len(per_item)),
        "variants": sorted(records["group"].unique().tolist()),
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "selection_fractions": SELECTION_FRACTIONS,
        "random_state": args.random_state,
        "n_splits": args.n_splits,
        "min_positive": args.min_positive,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir,
        oracle_summary,
        topk_alpha_oracle,
        rules,
        any_oracle_cv,
        variant_gate_cv,
        item_oracle_summary,
        item_any_oracle_cv,
        item_variant_gate_cv,
    )

    print("Wrote response-selector feasibility outputs to", output_dir)
    print(oracle_summary.to_string(index=False))
    print("\nBest deployable variant gate:")
    print(variant_gate_cv.dropna(subset=["policy_mean_gain_vs_a2"]).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
