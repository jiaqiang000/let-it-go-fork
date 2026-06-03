# Let It Go Amazon-M2 raw product text -> multilingual E5 embedding 验证

这个目录验证一件事：

> 能否从 Amazon-M2 原始 `products_train.csv` 的 FR 商品字段，按 Let It Go 的 `amazon_m2_fr` item 顺序重新得到作者提供的 content embeddings。

本验证目录只放脚本和输出，不修改 raw 数据、作者 processed 数据、训练代码或 notebook。

## 环境与存储

运行环境：

```bash
conda run -n let-it-go-py3.11 python ...
```

关键依赖：

```text
Python 3.11.15
numpy 1.26.4
pandas 2.2.2
pyarrow 19.0.1
polars 1.0.0
sentence-transformers 3.4.1
torch 2.6.0
tqdm 4.67.1
```

只读输入：

```text
let-it-go/row_data/amazon_m2_raw/
letitgo-data/data/amazon_m2_fr/processed/
letitgo-data/data/amazon_m2_fr/item_embeddings/
```

输出目录：

```text
outputs/amazon_m2_alignment_verify/
```

默认不会保存重新生成的全量 embedding `.npy`。本次全量编码只在内存里生成 embedding，并输出 JSON/CSV 对照摘要。

## 验证方法

Amazon-M2 preprocessing notebook 的关键逻辑是：

```python
products = pd.read_csv(products_train.csv)
products = products[products.locale == "FR"]
COLUMNS = ["title", "brand", "color", "size", "model", "material", "author"]
metadata = ""
for column in COLUMNS:
    metadata = metadata + (f"; {column}: " + products[column]).fillna("")
metadata = metadata.str.lstrip("; ")
model = SentenceTransformer("intfloat/multilingual-e5-base")
item_embeddings = model.encode(metadata.to_list(), normalize_embeddings=False)
```

脚本复刻了这个流程，包括 pandas 默认 NA 规则。因此 CSV 里的 `"NA"`、`"nan"` 等会被当作缺失值处理，而不是作为普通文本拼进 metadata。

## 命令

单元测试：

```bash
conda run -n let-it-go-py3.11 python -m unittest discover -s tests
```

轻量链路检查，不跑 E5：

```bash
conda run -n let-it-go-py3.11 python scripts/verify_amazon_m2_e5_embeddings.py
```

全量 E5 对照，不保存生成 embedding：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n let-it-go-py3.11 python scripts/verify_amazon_m2_e5_embeddings.py --encode --full
```

脚本会显示两个进度条：

- `Scanning products_train.csv`
- `Batches`，来自 SentenceTransformer 编码

## 当前结果

raw 文件完整性：

- expected files: 14
- expected files present: true
- `products_train.csv` 总商品行数：1,551,057
- FR 商品行数：44,577

item 侧覆盖：

- warm mapping: 42,647 个，`item_id = 1..42647`，连续且唯一
- cold mapping: 1,402 个，`item_id = 42648..44049`，连续且唯一
- warm+cold mapping 总数：44,049
- 在 FR `products_train.csv` 中匹配到：44,049
- 缺失商品元信息：0

embedding shape：

- `embeddings_warm.npy`: `(42647, 768)`
- `embeddings_cold.npy`: `(1402, 768)`
- warm/cold embedding 行数与 mapping 长度一致

processed parquet：

- `train_interactions.parquet`: `(405649, 3)`，item_id 范围 `1..42647`
- `val_interactions.parquet`: `(100584, 3)`，item_id 范围 `2..42643`
- `test_interactions.parquet`: `(48143, 4)`，item_id 范围 `5..44049`，含 `is_cold`
- `ground_truth.parquet`: `(12520, 3)`，item_id 范围 `5..44047`，含 `is_cold`

全量 E5 cosine 对照：

| split | count | mean | std | min | p50 | p95 | below 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| warm | 42647 | 0.999999821 | 1.37e-05 | 0.997565210 | 1.0 | 1.000000119 | 1 |
| cold | 1402 | 1.0 | 7.54e-08 | 0.999999881 | 1.0 | 1.000000119 | 0 |

唯一低于 0.999 的 item：

```text
split: warm
source_product_id: B09D44H52T
item_id: 29885
position: 29884
cosine: 0.9975652098655701
metadata_text: title: Sensodyne Dentifrice Soin Gel Fraîcheur Intense, Limitant la Sensibilité Dentaire, Lot de 3; brand: GSK; color: White; size: Lot de 3
```

对这个 item 额外测试过删字段、改 `color`、改 `brand` 等候选文本，当前 raw 文本仍是最接近作者 embedding 的版本。

## 结论

Amazon-M2 FR 这条数据链路是成立的：

```text
Amazon-M2 products_train.csv 的 FR 商品 id
        ↓
item2index_warm.pkl / item2index_cold.pkl
        ↓
processed parquet 的 numeric item_id
        ↓
title/brand/color/size/model/material/author 文本
        ↓
intfloat/multilingual-e5-base
        ↓
embeddings_warm.npy / embeddings_cold.npy
```

更严格地说：

- ID 覆盖和 item 顺序是完全对齐的；
- cold embedding 全量精确复现到浮点误差级别；
- warm embedding 中 42,646 / 42,647 个达到浮点误差级别；
- 剩余 1 个 warm item 有小差异，当前最可能是 Kaggle 镜像 raw 文件与作者当时 raw 文件的单条商品字段版本差异，或作者生成 embedding 时该条 metadata 有细微不同。

因此，这不是“乱接数据”。但如果论文里要写，建议表述为：

> Amazon-M2 FR 的 raw product metadata 与 Let It Go processed item mapping/embedding 基本全量对齐，并可用 notebook 所述 multilingual E5 流程复现作者 content embeddings；仅 1/44,049 个 item 出现轻微 cosine 差异。
