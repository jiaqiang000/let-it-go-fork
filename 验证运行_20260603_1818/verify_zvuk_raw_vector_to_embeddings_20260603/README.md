# Let It Go Zvuk raw track vector -> item embedding 验证

这个目录验证一件事：

> 能否从 Zvuk 原始 `zvuk-track_artist_embedding.parquet` 的 `track_id -> vector`，按 Let It Go 的 `zvuk` item 顺序重新得到作者提供的 `embeddings_warm.npy` 和 `embeddings_cold.npy`。

和 Beauty / Amazon-M2 不同，Zvuk 不是文本到 E5 embedding。它的 raw content 入口是预先给好的 128 维 track vector。

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
tqdm 4.67.1
```

只读输入：

```text
let-it-go/row_data/zvuk_raw/
letitgo-data/data/zvuk/processed/
letitgo-data/data/zvuk/item_embeddings/
```

raw 数据大小：

```text
zvuk-interactions.parquet              1.7G
zvuk-track_artist_embedding.parquet    807M
```

输出目录：

```text
outputs/zvuk_vector_alignment_verify/
```

本验证不会保存重新生成的全量 embedding `.npy`，只在内存里重建矩阵并输出 JSON/CSV 报告。

## 验证方法

Zvuk preprocessing notebook 的关键逻辑是：

```python
metadata = pl.read_parquet("zvuk-track_artist_embedding.parquet")
metadata = (
    metadata.rename({"track_id": "item_id"})
    .filter(pl.col("item_id").is_in(pl.concat((WARM_ITEMS, COLD_ITEMS))))
    .with_columns(pl.col("item_id").replace_strict({**item2index_warm, **item2index_cold}))
    .unique(["item_id", "vector"])
    .sort("item_id")
)
item_embeddings = np.vstack(metadata.get_column("vector").to_list())
```

本脚本复刻这个流程：

```text
raw track_id
  -> item2index_warm/cold.pkl
  -> numeric item_id
  -> sort by item_id
  -> stack vector
  -> split warm/cold
  -> compare with embeddings_warm/cold.npy
```

除了 cosine similarity，脚本还检查更严格的数值指标：

```text
np.allclose
max_abs_diff
mean_abs_diff
max_l2_diff
mean_l2_diff
```

## 命令

单元测试：

```bash
conda run -n let-it-go-py3.11 python -m unittest discover -s tests
```

完整验证：

```bash
conda run -n let-it-go-py3.11 python scripts/verify_zvuk_vectors.py
```

脚本会显示：

```text
Stacking ordered track vectors
```

## 当前结果

raw 文件：

- `zvuk-interactions.parquet`: `244,673,551 x 5`
- `zvuk-track_artist_embedding.parquet`: `2,199,876 x 4`
- track vector schema: `track_id Int32`, `vector List(Float32)`

mapping：

- warm mapping: 107,448 个，`item_id = 1..107448`，连续且唯一
- cold mapping: 23,637 个，`item_id = 107449..131085`，连续且唯一
- warm+cold mapping 总数：131,085

raw vector 覆盖：

- raw unique track-vector rows after filtering: 131,085
- joined rows: 131,085
- joined unique track ids: 131,085
- joined unique item ids: 131,085
- tracks with multiple unique vectors: 0

embedding shape：

- `embeddings_warm.npy`: `(107448, 128)`
- `embeddings_cold.npy`: `(23637, 128)`
- reconstructed full matrix: `(131085, 128)`

数值对照：

| split | count | allclose 1e-8 | max_abs_diff | max_l2_diff | cosine min | below 0.999 |
|---|---:|---:|---:|---:|---:|---:|
| warm | 107448 | true | 0.0 | 0.0 | 1.0 | 0 |
| cold | 23637 | true | 0.0 | 0.0 | 1.0 | 0 |
| full | 131085 | true | 0.0 | 0.0 | 1.0 | 0 |

processed parquet：

- `train_interactions.parquet`: `(2621480, 4)`，item_id 范围 `1..107448`
- `val_interactions.parquet`: `(263305, 4)`，item_id 范围 `1..107448`
- `test_interactions.parquet`: `(2644360, 6)`，item_id 范围 `1..131085`，含 `is_cold`
- `ground_truth.parquet`: `(4260, 6)`，item_id 范围 `55..131056`，含 `is_cold`

## 结论

Zvuk 这条数据链路完全对齐：

```text
zvuk-track_artist_embedding.parquet 的 track_id/vector
        ↓
item2index_warm.pkl / item2index_cold.pkl
        ↓
numeric item_id
        ↓
按 item_id 排序 stack vector
        ↓
embeddings_warm.npy / embeddings_cold.npy
```

更严格地说：

- ID 覆盖、item 顺序、embedding 行数全部一致；
- 重建出来的 full embedding matrix 与作者 warm+cold `.npy` 逐元素完全一致；
- `max_abs_diff = 0.0`，`max_l2_diff = 0.0`，`np.allclose(..., atol=1e-8) = true`。

因此，Zvuk 的 raw vector -> processed item embedding 链路不是推断出来的，而是可以逐元素精确复现。
