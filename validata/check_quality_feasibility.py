from pathlib import Path
import json
import gzip
import pickle
import ast
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

META_PATH = ROOT / "row_data" / "meta_Beauty.json"
WARM_PATH = ROOT / "data" / "beauty" / "processed" / "item2index_warm.pkl"
COLD_PATH = ROOT / "data" / "beauty" / "processed" / "item2index_cold.pkl"
WARM_EMB_PATH = ROOT / "data" / "beauty" / "item_embeddings" / "embeddings_warm.npy"
COLD_EMB_PATH = ROOT / "data" / "beauty" / "item_embeddings" / "embeddings_cold.npy"


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_one_record(text):
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


def iter_records(path):
    # 先按“每行一个对象”读
    with open_text(path) as f:
        line_success = False
        for line in f:
            obj = parse_one_record(line)
            if obj is not None:
                line_success = True
                yield obj
        if line_success:
            return

    # 如果逐行不行，再尝试整文件
    with open_text(path) as f:
        whole = f.read()

    obj = parse_one_record(whole)
    if obj is None:
        raise ValueError(f"无法解析文件: {path}")

    if isinstance(obj, list):
        for x in obj:
            yield x
    else:
        yield obj


def load_meta_df(path):
    rows = []
    for obj in iter_records(path):
        asin = obj.get("asin")
        if asin is None:
            continue

        title = obj.get("title", "")
        desc = obj.get("description", "")
        imUrl = obj.get("imUrl", "")
        categories = obj.get("categories", [])
        salesRank = obj.get("salesRank", {})

        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        else:
            desc = str(desc) if desc is not None else ""

        title = str(title) if title is not None else ""
        imUrl = str(imUrl) if imUrl is not None else ""

        # category depth: categories 可能是 list[list[str]]
        category_depth = 0
        if isinstance(categories, list):
            for x in categories:
                if isinstance(x, list):
                    category_depth = max(category_depth, len(x))

        rows.append({
            "asin": str(asin),
            "has_title": int(len(title.strip()) > 0),
            "title_len": len(title.strip()),
            "title_tokens": len(title.strip().split()),
            "has_desc": int(len(desc.strip()) > 0),
            "desc_len": len(desc.strip()),
            "desc_tokens": len(desc.strip().split()),
            "has_image": int(len(imUrl.strip()) > 0),
            "has_categories": int(isinstance(categories, list) and len(categories) > 0),
            "category_depth": category_depth,
            "has_salesRank": int(isinstance(salesRank, dict) and len(salesRank) > 0),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["asin"])
    return df


def load_mapping(path):
    with open(path, "rb") as f:
        mp = pickle.load(f)

    if not isinstance(mp, dict):
        raise ValueError(f"{path} 不是 dict")

    # 这里按你现在的输出，方向就是 raw asin -> index
    sample_k = next(iter(mp.keys()))
    sample_v = mp[sample_k]
    print(f"{path.name} sample: {repr(sample_k)} -> {repr(sample_v)}")

    return {str(k): int(v) for k, v in mp.items()}


def report_one_split(name, mp, meta_df):
    raw_ids = pd.DataFrame({
        "asin": list(mp.keys()),
        "item_id": list(mp.values()),
    })

    merged = raw_ids.merge(meta_df, on="asin", how="left")

    total = len(merged)
    covered = merged["has_title"].notna().sum()

    print(f"\n===== {name} =====")
    print("total items:", total)
    print("covered by meta:", covered)
    print("coverage rate:", round(covered / total, 6))

    missing = merged["has_title"].isna().sum()
    print("missing meta rows:", missing)

    valid = merged[merged["has_title"].notna()].copy()
    if len(valid) == 0:
        print("没有任何可用 metadata")
        return merged, valid

    print("has_title rate:", round(valid["has_title"].mean(), 6))
    print("has_desc rate:", round(valid["has_desc"].mean(), 6))
    print("has_image rate:", round(valid["has_image"].mean(), 6))
    print("has_categories rate:", round(valid["has_categories"].mean(), 6))
    print("has_salesRank rate:", round(valid["has_salesRank"].mean(), 6))

    print("avg title_len:", round(valid["title_len"].mean(), 2))
    print("avg desc_len:", round(valid["desc_len"].mean(), 2))
    print("avg title_tokens:", round(valid["title_tokens"].mean(), 2))
    print("avg desc_tokens:", round(valid["desc_tokens"].mean(), 2))
    print("avg category_depth:", round(valid["category_depth"].mean(), 2))

    print("\n5 rows with weakest metadata:")
    weak = valid.sort_values(
        by=["has_title", "has_desc", "has_image", "has_categories", "title_len", "desc_len"],
        ascending=[True, True, True, True, True, True]
    ).head(5)
    print(weak[[
        "asin", "item_id", "has_title", "has_desc", "has_image",
        "has_categories", "title_len", "desc_len", "category_depth"
    ]])

    return merged, valid


def check_embedding_alignment(warm_map, cold_map):
    warm_emb = np.load(WARM_EMB_PATH)
    cold_emb = np.load(COLD_EMB_PATH)

    print("\n===== EMBEDDING ALIGNMENT =====")
    print("warm map len:", len(warm_map))
    print("warm emb shape:", warm_emb.shape)
    print("cold map len:", len(cold_map))
    print("cold emb shape:", cold_emb.shape)

    print("warm aligned:", len(warm_map) == warm_emb.shape[0])
    print("cold aligned:", len(cold_map) == cold_emb.shape[0])


def main():
    meta_df = load_meta_df(META_PATH)
    print("meta rows:", len(meta_df))
    print("meta columns:", list(meta_df.columns))

    warm_map = load_mapping(WARM_PATH)
    cold_map = load_mapping(COLD_PATH)

    report_one_split("WARM", warm_map, meta_df)
    report_one_split("COLD", cold_map, meta_df)

    check_embedding_alignment(warm_map, cold_map)


if __name__ == "__main__":
    main()