from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
HOME_ROOT = Path.home()
PARENT_ROOT = ROOT.parent

DEFAULT_META_PATH = ROOT / "row_data" / "meta_Beauty.json"
DEFAULT_PROCESSED_CANDIDATES = [
    ROOT / "data" / "beauty" / "processed",
    HOME_ROOT / "letitgo-data" / "data" / "beauty" / "processed",
    HOME_ROOT / "data" / "beauty" / "processed",
    PARENT_ROOT / "letitgo-data" / "data" / "beauty" / "processed",
    PARENT_ROOT / "data" / "beauty" / "processed",
]
DEFAULT_OUTPUT_DIR = ROOT / "quality_score_output" / "beauty"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
GENERIC_TITLES = {"item", "product", "test", "untitled", "na", "none", "unknown", "null"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Beauty quality score v1.")
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing item2index_warm.pkl and item2index_cold.pkl. "
            "If omitted, the script tries built-in Beauty locations such as "
            "~/letitgo-data/data/beauty/processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Defaults to <repo>/quality_score_output/beauty.",
    )
    return parser.parse_args()


def resolve_processed_dir(explicit_dir: Path | None) -> Path:
    if explicit_dir is not None:
        processed_dir = explicit_dir.expanduser().resolve()
        if not processed_dir.exists():
            raise FileNotFoundError(f"processed dir not found: {processed_dir}")
        return processed_dir

    for candidate in DEFAULT_PROCESSED_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    candidates = "\n".join(str(path) for path in DEFAULT_PROCESSED_CANDIDATES)
    raise FileNotFoundError(f"failed to locate processed dir, tried:\n{candidates}")


def parse_one_record(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return None


def load_mapping(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        mapping = pickle.load(f)

    if not isinstance(mapping, dict):
        raise ValueError(f"{path} is not a dict")

    return {str(key): int(value) for key, value in mapping.items()}


def normalize_text(value: object) -> str:
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_title_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def compute_category_depth(categories: object) -> int:
    if not isinstance(categories, list):
        return 0

    max_depth = 0
    for path in categories:
        if isinstance(path, list):
            max_depth = max(max_depth, len(path))
    return max_depth


def compute_clean_score(title: str, description: str, title_tokens: int) -> tuple[float, dict[str, int | float]]:
    penalty = 0.0
    combined = f"{title} {description}".strip()
    non_space = [char for char in combined if not char.isspace()]
    non_space_len = len(non_space)

    if title_tokens < 2:
        penalty += 0.25

    if normalize_title_key(title) in GENERIC_TITLES:
        penalty += 0.25

    if HTML_RE.search(combined) or URL_RE.search(combined):
        penalty += 0.25

    alpha_chars = sum(char.isalpha() for char in non_space)
    digit_chars = sum(char.isdigit() for char in non_space)
    symbol_chars = non_space_len - alpha_chars - digit_chars
    alpha_ratio = alpha_chars / non_space_len if non_space_len else 0.0
    noisy_ratio = (digit_chars + symbol_chars) / non_space_len if non_space_len else 0.0

    if non_space_len and (alpha_ratio < 0.55 or noisy_ratio > 0.45):
        penalty += 0.25

    return max(0.0, 1.0 - penalty), {
        "alpha_ratio": alpha_ratio,
        "noisy_ratio": noisy_ratio,
    }


def iter_meta_rows(meta_path: Path, keep_asins: set[str]) -> list[dict[str, object]]:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = parse_one_record(line)
            if obj is None:
                continue

            asin = str(obj.get("asin", "")).strip()
            if not asin or asin not in keep_asins:
                continue

            title = normalize_text(obj.get("title"))
            description = normalize_text(obj.get("description"))
            image_url = normalize_text(obj.get("imUrl"))
            categories = obj.get("categories", [])

            title_tokens = tokenize(title)
            desc_tokens = tokenize(description)
            clean_score, clean_meta = compute_clean_score(title, description, len(title_tokens))

            rows.append(
                {
                    "asin": asin,
                    "title": title,
                    "description": description,
                    "has_title": int(bool(title)),
                    "has_desc": int(bool(description)),
                    "has_image": int(bool(image_url)),
                    "has_categories": int(isinstance(categories, list) and len(categories) > 0),
                    "title_len": len(title),
                    "desc_len": len(description),
                    "title_tokens": len(title_tokens),
                    "desc_tokens": len(desc_tokens),
                    "text_tokens": len(title_tokens) + len(desc_tokens),
                    "category_depth": compute_category_depth(categories),
                    "clean_i": clean_score,
                    "alpha_ratio": clean_meta["alpha_ratio"],
                    "noisy_ratio": clean_meta["noisy_ratio"],
                }
            )

    return rows


def safe_log_quantile_score(values: pd.Series, p5: float, p95: float) -> pd.Series:
    if p95 <= p5:
        return (values > 0).astype(float)

    numerator = np.log1p(values) - np.log1p(p5)
    denominator = np.log1p(p95) - np.log1p(p5)
    return pd.Series(np.clip(numerator / denominator, 0.0, 1.0), index=values.index)


def safe_ratio_score(values: pd.Series, upper: float) -> pd.Series:
    if upper <= 0:
        return pd.Series(0.0, index=values.index)
    return pd.Series(np.clip(values / upper, 0.0, 1.0), index=values.index)


def ensure_contiguous_item_ids(df: pd.DataFrame, split_name: str) -> None:
    item_ids = df["item_id"].to_numpy()
    expected = np.arange(item_ids.min(), item_ids.max() + 1)
    if len(item_ids) != len(expected) or not np.array_equal(np.sort(item_ids), expected):
        raise ValueError(f"{split_name} item ids are not contiguous, cannot align quality array safely")


def assign_quality_buckets(scores: pd.Series, low_thr: float, high_thr: float) -> pd.Series:
    return pd.Series(
        np.where(scores < low_thr, "low", np.where(scores < high_thr, "mid", "high")),
        index=scores.index,
    )


def main() -> None:
    args = parse_args()
    meta_path = args.meta_path.expanduser().resolve()
    processed_dir = resolve_processed_dir(args.processed_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta path not found: {meta_path}")

    warm_mapping = load_mapping(processed_dir / "item2index_warm.pkl")
    cold_mapping = load_mapping(processed_dir / "item2index_cold.pkl")
    keep_asins = set(warm_mapping) | set(cold_mapping)

    meta_rows = iter_meta_rows(meta_path, keep_asins)
    meta_df = pd.DataFrame(meta_rows).drop_duplicates(subset=["asin"])

    mapping_rows = []
    for split_name, mapping in (("warm", warm_mapping), ("cold", cold_mapping)):
        for asin, item_id in mapping.items():
            mapping_rows.append({"split": split_name, "asin": asin, "item_id": item_id})

    items_df = pd.DataFrame(mapping_rows).merge(meta_df, on="asin", how="left")

    missing = items_df["has_title"].isna().sum()
    if missing:
        missing_asins = items_df.loc[items_df["has_title"].isna(), "asin"].head(10).tolist()
        raise ValueError(f"missing metadata for {missing} items, examples: {missing_asins}")

    warm_df = items_df[items_df["split"] == "warm"].copy()
    cold_df = items_df[items_df["split"] == "cold"].copy()

    for split_name, split_df in (("warm", warm_df), ("cold", cold_df)):
        ensure_contiguous_item_ids(split_df, split_name)

    text_p5 = float(warm_df["text_tokens"].quantile(0.05))
    text_p95 = float(warm_df["text_tokens"].quantile(0.95))
    attr_raw_warm = warm_df["has_categories"] + 0.5 * warm_df["category_depth"]
    attr_p95 = float(attr_raw_warm.quantile(0.95))

    items_df["cov_i"] = items_df[["has_title", "has_desc", "has_image", "has_categories"]].mean(axis=1)
    items_df["txt_i"] = safe_log_quantile_score(items_df["text_tokens"], text_p5, text_p95)
    items_df["att_i"] = safe_ratio_score(
        items_df["has_categories"] + 0.5 * items_df["category_depth"],
        attr_p95,
    )
    items_df["quality_score"] = (
        0.35 * items_df["cov_i"]
        + 0.35 * items_df["txt_i"]
        + 0.20 * items_df["att_i"]
        + 0.10 * items_df["clean_i"]
    ).clip(0.0, 1.0)

    warm_scores = items_df.loc[items_df["split"] == "warm", "quality_score"]
    low_thr = float(warm_scores.quantile(1.0 / 3.0))
    high_thr = float(warm_scores.quantile(2.0 / 3.0))
    items_df["quality_bucket"] = assign_quality_buckets(items_df["quality_score"], low_thr, high_thr)

    warm_sorted = items_df[items_df["split"] == "warm"].sort_values("item_id")
    cold_sorted = items_df[items_df["split"] == "cold"].sort_values("item_id")

    warm_quality = warm_sorted["quality_score"].to_numpy(dtype=np.float32)
    cold_quality = cold_sorted["quality_score"].to_numpy(dtype=np.float32)

    np.save(output_dir / "warm_quality.npy", warm_quality)
    np.save(output_dir / "cold_quality.npy", cold_quality)

    feature_columns = [
        "split",
        "asin",
        "item_id",
        "has_title",
        "has_desc",
        "has_image",
        "has_categories",
        "title_len",
        "desc_len",
        "title_tokens",
        "desc_tokens",
        "text_tokens",
        "category_depth",
        "alpha_ratio",
        "noisy_ratio",
        "cov_i",
        "txt_i",
        "att_i",
        "clean_i",
        "quality_score",
        "quality_bucket",
    ]
    items_df.sort_values(["split", "item_id"])[feature_columns].to_csv(
        output_dir / "quality_features.csv",
        index=False,
    )

    stats = {
        "meta_path": str(meta_path),
        "processed_dir": str(processed_dir),
        "warm_items": int(len(warm_quality)),
        "cold_items": int(len(cold_quality)),
        "text_token_p5": round(text_p5, 6),
        "text_token_p95": round(text_p95, 6),
        "attr_p95": round(attr_p95, 6),
        "quality_bucket_low_threshold": round(low_thr, 6),
        "quality_bucket_high_threshold": round(high_thr, 6),
        "warm_quality_mean": round(float(warm_quality.mean()), 6),
        "cold_quality_mean": round(float(cold_quality.mean()), 6),
    }
    with (output_dir / "quality_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    print("meta path:", meta_path)
    print("processed dir:", processed_dir)
    print("output dir:", output_dir)
    print("saved:", output_dir / "warm_quality.npy", warm_quality.shape)
    print("saved:", output_dir / "cold_quality.npy", cold_quality.shape)
    print("saved:", output_dir / "quality_features.csv")
    print("saved:", output_dir / "quality_stats.json")
    print("warm quality mean:", round(float(warm_quality.mean()), 6))
    print("cold quality mean:", round(float(cold_quality.mean()), 6))


if __name__ == "__main__":
    main()
