"""汇总 degraded-view 方法 pilot 的本地预检证据。

本脚本不训练模型。它只读取已经沉淀的 CSV，回答一个严格问题：

现有 Let It Go 实验数据是否足够直接宣称 degraded-view 方法成立？

当前答案通常应是：不够。已有数据只能支持 title_trunc_8 作为温和退化视图
进入服务器训练 pilot；Pilot 2/3 仍需要训练结果。脚本会把这个边界写进
CSV、run_manifest 和结果 MD，避免把诊断证据误写成方法贡献。
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
TEMP_ROOT = RESEARCH_ROOT / "temp_202606_实验文件记录"
DEFAULT_OUTPUT_DIR = TEMP_ROOT / f"temp_{datetime.now().strftime('%Y%m%d')}" / "degraded-view-training-pilot-preflight"
DEFAULT_DEGRADED_DIR = TEMP_ROOT / "temp_20260606" / "degraded-view-sanity-check"
DEFAULT_CONTROLLED_DIR = TEMP_ROOT / "temp_20260605" / "受控字段消融扩展评测" / "variant_eval_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze degraded-view training pilot readiness.")
    parser.add_argument("--degraded-metrics", type=Path, default=DEFAULT_DEGRADED_DIR / "degraded_view_metrics.csv")
    parser.add_argument("--degraded-deltas", type=Path, default=DEFAULT_DEGRADED_DIR / "degraded_view_metric_deltas.csv")
    parser.add_argument("--controlled-overall", type=Path, default=DEFAULT_CONTROLLED_DIR / "variant_overall_metrics.csv")
    parser.add_argument("--controlled-field", type=Path, default=DEFAULT_CONTROLLED_DIR / "variant_field_group_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def require_csv(path: Path, label: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    return pd.read_csv(path)


def metric_col(metric: str, topk: int) -> str:
    return f"{metric}@{topk}"


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator.abs() > 1e-12, numerator / denominator, np.nan)


def summarize_degraded_retention(metrics: pd.DataFrame, topk: int) -> pd.DataFrame:
    ndcg = metric_col("NDCG", topk)
    recall = metric_col("Recall", topk)
    baseline = metrics[metrics["variant"] == "full_content_zero_delta"][
        ["field_group", ndcg, recall]
    ].rename(columns={ndcg: f"baseline_{ndcg}", recall: f"baseline_{recall}"})
    if baseline.empty:
        raise ValueError("degraded_view_metrics.csv 缺少 full_content_zero_delta baseline。")

    merged = metrics.merge(baseline, on="field_group", how="left")
    merged = merged.rename(columns={ndcg: f"variant_{ndcg}", recall: f"variant_{recall}"})
    merged[f"drop_{ndcg}"] = merged[f"baseline_{ndcg}"] - merged[f"variant_{ndcg}"]
    merged[f"drop_{recall}"] = merged[f"baseline_{recall}"] - merged[f"variant_{recall}"]
    merged[f"retention_{ndcg}"] = safe_ratio(merged[f"variant_{ndcg}"], merged[f"baseline_{ndcg}"])
    merged[f"retention_{recall}"] = safe_ratio(merged[f"variant_{recall}"], merged[f"baseline_{recall}"])
    merged["candidate_role"] = np.select(
        [
            merged["variant"].eq("full_content_zero_delta"),
            merged["variant"].eq("A2_original_warm_delta"),
            merged[f"drop_{ndcg}"].between(0.02, 0.08, inclusive="both"),
            merged[f"drop_{ndcg}"].gt(0.12),
        ],
        ["content_only_baseline", "warm_delta_reference", "mild_degraded_candidate", "destructive_negative_control"],
        default="other",
    )
    return merged.sort_values(["variant", "field_group"]).reset_index(drop=True)


def summarize_controlled_ablation(overall: pd.DataFrame, field: pd.DataFrame, topk: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ndcg = metric_col("NDCG", topk)
    cold_ndcg = f"cold_{ndcg}"
    cold_recall = f"cold_Recall@{topk}"
    a2 = overall[overall["model"] == "A2"].copy()
    baseline = a2[a2["variant"] == "control_full"][[cold_ndcg, cold_recall]]
    if baseline.empty:
        baseline = a2[a2["variant"] == "original_author"][[cold_ndcg, cold_recall]]
    if baseline.empty:
        raise ValueError("variant_overall_metrics.csv 缺少 A2 control_full/original_author baseline。")
    baseline_row = baseline.iloc[0]
    a2[f"drop_{cold_ndcg}"] = float(baseline_row[cold_ndcg]) - a2[cold_ndcg]
    a2[f"drop_{cold_recall}"] = float(baseline_row[cold_recall]) - a2[cold_recall]
    a2["evidence_role"] = np.select(
        [
            a2["variant"].isin(["control_full", "original_author"]),
            a2["variant"].isin(["drop_four", "title_brand_only"]),
            a2["variant"].isin(["no_title", "no_title_brand", "empty_text"]),
        ],
        ["baseline", "controlled_field_ablation", "textual_evidence_stress"],
        default="other_ablation",
    )

    a2_field = field[(field["model"] == "A2") & (field["variant"].isin(["original_author", "control_full"]))].copy()
    if "field_group" in a2_field.columns and not a2_field.empty:
        pivot = a2_field.pivot_table(
            index="variant",
            columns="field_group",
            values=cold_ndcg,
            aggfunc="first",
        ).reset_index()
        if {"weak_0_1", "strong_3_4"}.issubset(pivot.columns):
            pivot["natural_strong_minus_weak_NDCG@10"] = pivot["strong_3_4"] - pivot["weak_0_1"]
    else:
        pivot = pd.DataFrame()
    return a2.sort_values("drop_" + cold_ndcg).reset_index(drop=True), pivot


def value_at(df: pd.DataFrame, variant: str, field_group: str, column: str) -> float:
    rows = df[(df["variant"] == variant) & (df["field_group"] == field_group)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][column])


def build_pilot_gate_summary(retention: pd.DataFrame, controlled: pd.DataFrame, topk: int) -> pd.DataFrame:
    ndcg = metric_col("NDCG", topk)
    drop_col = f"drop_{ndcg}"
    title_all_drop = value_at(retention, "title_trunc_8_zero_delta", "all", drop_col)
    title_weak_drop = value_at(retention, "title_trunc_8_zero_delta", "weak_0_1", drop_col)
    title_strong_drop = value_at(retention, "title_trunc_8_zero_delta", "strong_3_4", drop_col)
    no_title_all_drop = value_at(retention, "no_title_zero_delta", "all", drop_col)

    text_drop = float("nan")
    if not controlled.empty and "variant" in controlled.columns:
        no_title = controlled[controlled["variant"] == "no_title"]
        if not no_title.empty and "drop_cold_NDCG@10" in no_title.columns:
            text_drop = float(no_title.iloc[0]["drop_cold_NDCG@10"])

    if title_all_drop >= 0.03 and no_title_all_drop > title_all_drop * 2:
        pilot1_status = "weak_go"
        pilot1_decision = "允许进入服务器训练 pilot，但只能用 title_trunc_8 作为温和退化主候选。"
    else:
        pilot1_status = "stop"
        pilot1_decision = "degraded view 本身信号不足，不建议继续训练方法。"

    rows: list[dict[str, Any]] = [
        {
            "pilot": "Pilot 1",
            "question": "人工 degraded view 是否能制造接近 cold difficulty 的推荐退化？",
            "status": pilot1_status,
            "local_evidence": (
                f"title_trunc_8 all drop={title_all_drop:.4f}; "
                f"weak drop={title_weak_drop:.4f}; strong drop={title_strong_drop:.4f}; "
                f"no_title all drop={no_title_all_drop:.4f}"
            ),
            "decision": pilot1_decision,
            "next_required_step": "服务器训练 title_trunc_8 / random_title_dropout / no_title / control_full 矩阵。",
        },
        {
            "pilot": "Pilot 2",
            "question": "Retention(p) 或退化敏感性是否能预测哪些 item 会从训练增强中受益？",
            "status": "not_tested",
            "local_evidence": "本地只能计算退化后的 ranking drop，尚未和训练后 item-level gain 关联。",
            "decision": "不能作为方法 gate 或样本权重。",
            "next_required_step": "拿到服务器训练后的 item/group gain，再做 retention-gain 相关与分桶。",
        },
        {
            "pilot": "Pilot 3",
            "question": "degraded-view 训练是否真的优于简单基线？",
            "status": "not_tested",
            "local_evidence": f"受控消融显示 title/text 很关键；A2 no_title cold NDCG drop={text_drop:.4f}。",
            "decision": "现有数据不能宣称方法成立。",
            "next_required_step": "至少比较 control_full、title_trunc_8、random_title_dropout_p30、no_title 的 cold overall 与 weak/mid/strong。",
        },
    ]
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "无数据。"
    view = df[columns].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else str(value))

    # 中文注释：不依赖 pandas.to_markdown/tabulate，保证服务器最小环境也能写结果 MD。
    widths = {
        column: max(len(str(column)), *(len(str(value)) for value in view[column].tolist()))
        for column in view.columns
    }
    header = "| " + " | ".join(str(column).ljust(widths[column]) for column in view.columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in view.columns) + " |"
    rows = [
        "| " + " | ".join(str(record[column]).ljust(widths[column]) for column in view.columns) + " |"
        for record in view.to_dict("records")
    ]
    return "\n".join([header, separator, *rows])


def write_result_md(
    output_dir: Path,
    timestamp: str,
    retention: pd.DataFrame,
    controlled: pd.DataFrame,
    gates: pd.DataFrame,
    manifest: dict[str, Any],
    topk: int,
) -> Path:
    ndcg = metric_col("NDCG", topk)
    result_path = output_dir / f"{timestamp} Amazon-M2 degraded-view training pilot 预检结果.md"
    key_retention = retention[
        retention["variant"].isin(["full_content_zero_delta", "title_trunc_8_zero_delta", "no_title_zero_delta"])
        & retention["field_group"].isin(["all", "weak_0_1", "mid_2", "strong_3_4"])
    ]
    key_controlled = controlled[
        controlled["variant"].isin(["control_full", "drop_four", "title_only", "no_title", "no_title_brand", "empty_text"])
    ]

    body = f"""# {timestamp} Amazon-M2 degraded-view training pilot 预检结果

创建时间：{timestamp}

结果目录：

```text
{output_dir}
```

## 1. 严格结论

现有本地数据**不足以证明 degraded-view training 方法成立**。它只支持一个较窄判断：

```text
title_trunc_8 可以作为温和 degraded view 候选进入服务器训练 pilot；
no_title 只能作为极端负面对照；
Pilot 2 / Pilot 3 还没有训练结果，不能写成方法贡献。
```

因此下一步不是继续解释已废弃的字段预算旧路线或 cold-delta 迁移路线，而是只跑一个最小训练矩阵，判断“训练期弱内容视图鲁棒性”有没有真实收益。

## 2. 本地 degraded-view 预检

{dataframe_to_markdown(
        key_retention,
        [
            "variant",
            "field_group",
            f"baseline_{ndcg}",
            f"variant_{ndcg}",
            f"drop_{ndcg}",
            f"retention_{ndcg}",
            "candidate_role",
        ],
    )}

## 3. 受控字段/文本消融背景

{dataframe_to_markdown(
        key_controlled,
        [
            "variant",
            "cold_NDCG@10",
            "drop_cold_NDCG@10",
            "cold_Recall@10",
            "drop_cold_Recall@10",
            "evidence_role",
        ],
    )}

## 4. Pilot gate

{dataframe_to_markdown(
        gates,
        ["pilot", "status", "local_evidence", "decision", "next_required_step"],
        max_rows=20,
    )}

## 5. 服务器训练入口

服务器上按已有沉淀方式启动主脚本：

```bash
cd /root/letitgo-runtime/let-it-go
mkdir -p /hy-tmp/letitgo_logs /hy-tmp/letitgo_ckpt

LOG=/hy-tmp/letitgo_logs/amazon_m2_degraded_view_training_pilot_outer_$(date +%Y%m%d_%H%M%S).log

nohup bash scripts/run_amazon_m2_degraded_view_training_pilot_2seeds.sh \\
  > "$LOG" 2>&1 &

echo "$LOG"
tail -f "$LOG"
```

## 6. manifest 摘要

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```
"""
    result_path.write_text(body, encoding="utf-8")
    return result_path


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d %H%M%S")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    degraded_metrics = require_csv(args.degraded_metrics, "degraded metrics")
    degraded_deltas = require_csv(args.degraded_deltas, "degraded metric deltas")
    controlled_overall = require_csv(args.controlled_overall, "controlled overall metrics")
    controlled_field = require_csv(args.controlled_field, "controlled field-group metrics")

    retention = summarize_degraded_retention(degraded_metrics, topk=args.topk)
    controlled_summary, natural_gap = summarize_controlled_ablation(
        controlled_overall,
        controlled_field,
        topk=args.topk,
    )
    gates = build_pilot_gate_summary(retention, controlled_summary, topk=args.topk)

    retention.to_csv(output_dir / "degraded_view_retention_summary.csv", index=False)
    controlled_summary.to_csv(output_dir / "controlled_ablation_context_summary.csv", index=False)
    natural_gap.to_csv(output_dir / "natural_group_gap_context.csv", index=False)
    gates.to_csv(output_dir / "pilot_gate_summary.csv", index=False)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script_role": "local preflight aggregation for degraded-view training pilot",
        "degraded_metrics": str(args.degraded_metrics.expanduser().resolve()),
        "degraded_deltas": str(args.degraded_deltas.expanduser().resolve()),
        "controlled_overall": str(args.controlled_overall.expanduser().resolve()),
        "controlled_field": str(args.controlled_field.expanduser().resolve()),
        "output_dir": str(output_dir),
        "topk": args.topk,
        "local_data_sufficiency": "insufficient_for_method_claim; sufficient_for_server_training_pilot",
        "result_files": [
            "degraded_view_retention_summary.csv",
            "controlled_ablation_context_summary.csv",
            "natural_group_gap_context.csv",
            "pilot_gate_summary.csv",
        ],
    }
    result_md = write_result_md(output_dir, timestamp, retention, controlled_summary, gates, manifest, args.topk)
    manifest["result_md"] = str(result_md)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote: {output_dir}")
    print(f"result_md: {result_md}")


if __name__ == "__main__":
    main()
