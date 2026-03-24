from pathlib import Path
import json
import gzip
import pickle
import ast
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "meta_json": ROOT / "row_data" / "meta_Beauty.json",
    "reviews_json": ROOT / "row_data" / "Beauty_5.json",

    "train_parquet": ROOT / "data" / "beauty" / "processed" / "train_interactions.parquet",
    "val_parquet": ROOT / "data" / "beauty" / "processed" / "val_interactions.parquet",
    "test_parquet": ROOT / "data" / "beauty" / "processed" / "test_interactions.parquet",
    "gt_parquet": ROOT / "data" / "beauty" / "processed" / "ground_truth.parquet",

    "warm_pkl": ROOT / "data" / "beauty" / "processed" / "item2index_warm.pkl",
    "cold_pkl": ROOT / "data" / "beauty" / "processed" / "item2index_cold.pkl",

    "warm_npy": ROOT / "data" / "beauty" / "item_embeddings" / "embeddings_warm.npy",
    "cold_npy": ROOT / "data" / "beauty" / "item_embeddings" / "embeddings_cold.npy",
}

def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def parse_one_record(text):
    text = text.strip()
    if not text:
        return None

    # 先尝试严格 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试 Python dict literal（老 Amazon 数据常见）
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return None

def inspect_json(path, n=2):
    print(f"\n===== JSON/TEXT: {path} =====")

    # 先尝试按“每行一条记录”读取
    samples = []
    with open_text(path) as f:
        for line in f:
            obj = parse_one_record(line)
            if obj is not None:
                samples.append(obj)
                if len(samples) >= n:
                    break

    # 如果逐行失败，再尝试整文件读取
    if not samples:
        with open_text(path) as f:
            whole = f.read()

        obj = parse_one_record(whole)

        if obj is None:
            raise ValueError(
                f"{path} 既不是 JSON lines，也不是单个合法 JSON / Python dict 文件，请先打印文件前几行看看格式。"
            )

        if isinstance(obj, list):
            samples = obj[:n]
        else:
            samples = [obj]

    print(f"parsed {len(samples)} sample record(s)")
    for i, obj in enumerate(samples):
        if isinstance(obj, dict):
            print(f"[sample {i}] keys = {list(obj.keys())}")
        else:
            print(f"[sample {i}] type = {type(obj)}")
        print(obj)
        print("-" * 80)

def inspect_parquet(path):
    print(f"\n===== PARQUET: {path} =====")
    df = pd.read_parquet(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("head:")
    print(df.head())

def inspect_pkl(path):
    print(f"\n===== PKL: {path} =====")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print("type:", type(obj))
    if isinstance(obj, dict):
        print("len:", len(obj))
        for i, (k, v) in enumerate(obj.items()):
            if i >= 5:
                break
            print(repr(k), "->", repr(v))
    else:
        print(obj)

def inspect_npy(path):
    print(f"\n===== NPY: {path} =====")
    arr = np.load(path, allow_pickle=True)
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    if arr.ndim >= 2 and arr.shape[0] > 0:
        print("first row first 10 dims:", arr[0][:10])

def main():
    for name, path in FILES.items():
        if not path.exists():
            print(f"[MISSING] {name}: {path}")
            continue

        if path.suffix in [".json", ".gz"]:
            inspect_json(path)
        elif path.suffix == ".parquet":
            inspect_parquet(path)
        elif path.suffix == ".pkl":
            inspect_pkl(path)
        elif path.suffix == ".npy":
            inspect_npy(path)

if __name__ == "__main__":
    main()