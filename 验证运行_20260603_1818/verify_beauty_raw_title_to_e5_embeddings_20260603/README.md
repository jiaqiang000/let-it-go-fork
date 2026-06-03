# Let It Go Beauty raw title -> E5 embedding 验证

这个目录只验证一件事：

> 能否从 Beauty 原始 `meta_Beauty.json` 的 `title` 字段，按 Let It Go 的 item 顺序重新得到作者实验入口所需的 E5 content embeddings。

它放在 Let It Go 仓库内的 `验证运行_20260603_1818/` 旁路目录下，只是为了贴近所依赖的数据切分、embedding 文件和原作者代码；它不放文献沉淀，不放正式实验结论，也不修改 `source/`、`scripts/`、`notebooks/` 里的原方法代码。

## 环境

本次验证使用全局 Conda 环境：

```bash
conda activate let-it-go-py3.11
```

已安装并确认可导入的关键依赖：

```text
python: /opt/anaconda3/envs/let-it-go-py3.11/bin/python
torch: 2.6.0
transformers: 4.49.0
sentence-transformers: 3.4.1
```

安装命令：

```bash
conda run -n let-it-go-py3.11 python -m pip install 'sentence-transformers==3.4.1' 'transformers==4.49.0'
```

不要用裸 `pip install`，避免包装到 Mac 其他 Python 环境里。

## 当前已完成

输出位置：

```text
outputs/beauty_title_e5_verify/
```

### 1. 非模型链路检查

运行命令：

```bash
conda run -n let-it-go-py3.11 python scripts/verify_beauty_e5_embeddings.py
```

检查结果：

- warm items: 11165
- cold items: 568
- matched metadata rows: 11733
- author warm embedding shape: 11165 x 768
- author cold embedding shape: 568 x 768
- warm/cold embedding 行数与 item 映射均一致

这说明 `meta_Beauty.json -> item2index_warm/cold.pkl -> embeddings_warm/cold.npy` 的 ID 和顺序链路是通的。

### 2. 全量 E5 编码对照

运行命令：

```bash
conda run -n let-it-go-py3.11 python scripts/verify_beauty_e5_embeddings.py --encode --full
```

它复刻 Beauty notebook 的做法：

```text
title -> intfloat/e5-base-v2 -> generated embedding
```

然后和作者给的 `embeddings_warm.npy`、`embeddings_cold.npy` 做 cosine similarity 对照。

编码规模：

- warm: 11165
- cold: 568

cosine similarity 结果：

| split | count | mean | std | min | p50 | max | below 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| warm | 11165 | 1.0 | 7.41e-08 | 0.99999988 | 1.0 | 1.00000024 | 0 |
| cold | 568 | 1.0 | 7.41e-08 | 0.99999988 | 1.0 | 1.00000024 | 0 |

缺失 title 的处理细节：

- 有 7 个 item 的原始 title 缺失。
- 直接把缺失 title 当成空字符串 `""` 会得到 cosine 约 0.7827。
- 作者 embedding 与字符串 `"None"` 的 E5 embedding 完全一致。
- 因此脚本用 `title_for_encoding` 字段复刻这一点：普通 title 原样编码，缺失 title 按 `"None"` 编码。

结论：在 Beauty 数据集上，作者提供的 `embeddings_warm.npy` 和 `embeddings_cold.npy` 可以由 `meta_Beauty.json` 的 `title` 字段经过 `intfloat/e5-base-v2` 全量精确复现。由此可确认，至少 Beauty 这条线的 raw title -> E5 content embedding 前半段是可复现的。

注意：这只证明 Beauty 的 `title -> E5 embedding` 入口可复现，不等于已经复现完整 Let It Go 训练流程，也不等于 Amazon-M2 或 Zvuk 的前处理都已验证。
