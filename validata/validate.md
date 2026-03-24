## Beauty 数据链路与 Quality Score v1 可行性结论总结

### 1. 任务背景

当前目标是判断：

1. `meta_Beauty.json`、`Beauty_5.json` 与处理后的 8 个文件之间，是否能建立清晰的数据关系；
2. 在此基础上，Beauty 数据集是否足以支撑一个规则型的 `quality score v1`。

这里的 8 个文件是：

- `train_interactions.parquet`
- `val_interactions.parquet`
- `test_interactions.parquet`
- `ground_truth.parquet`
- `item2index_warm.pkl`
- `item2index_cold.pkl`
- `embeddings_warm.npy`
- `embeddings_cold.npy`

---

### 2. 已确认的文件角色

#### 2.1 原始数据文件

- `row_data/meta_Beauty.json`
  - 保存商品元信息
  - 关键字段包括：
    - `asin`
    - `title`
    - `description`
    - `imUrl`
    - `categories`
    - `salesRank`

- `row_data/Beauty_5.json`
  - 保存交互数据
  - 对应 Beauty 的交互原始来源

#### 2.2 处理后文件

- `data/beauty/processed/*.parquet`
  - 是训练/验证/测试/GT 用的交互表
  - 其中 `item_id` 已经不是原始 `asin`，而是数值化后的索引

- `data/beauty/processed/item2index_warm.pkl`
- `data/beauty/processed/item2index_cold.pkl`
  - 是原始商品 ID 和数值索引之间的桥梁
  - 已确认方向为：

  ```python
  raw_asin -> numeric_item_id
  ```

  例如：

  ```python
  '9788072216' -> 1
  '9790794231' -> 11166
  ```

- `data/beauty/item_embeddings/embeddings_warm.npy`
- `data/beauty/item_embeddings/embeddings_cold.npy`
  - 是 warm / cold item 的内容向量
  - 行顺序应与 warm / cold 映射顺序一致

---

### 3. 目前已经建立的数据关系

当前已经明确以下数据主链路：

```text
meta_Beauty.json(asin)
    ↓
item2index_warm.pkl / item2index_cold.pkl
(raw asin -> numeric item_id)
    ↓
processed/*.parquet
(item_id 为数值索引)
    ↓
embeddings_warm.npy / embeddings_cold.npy
(按 warm/cold item 顺序排列)
```

也就是说：

- `meta_Beauty.json` 中的 `asin` 可以通过两个 `pkl` 映射到数值 `item_id`
- `processed` 中的 `item_id` 与该索引体系一致
- `embeddings_warm/cold.npy` 的行数与映射文件长度一致，说明 embedding 与映射顺序是对齐的

---

### 4. 核查结果

#### 4.1 metadata 总体规模

- `meta rows = 259204`

说明 `meta_Beauty.json` 中共有 259,204 条元信息记录。

#### 4.2 WARM 集合核查结果

- `total items = 11165`
- `covered by meta = 11165`
- `coverage rate = 1.0`
- `missing meta rows = 0`

字段覆盖率：

- `has_title rate = 0.999463`
- `has_desc rate = 0.921093`
- `has_image rate = 0.999463`
- `has_categories rate = 1.0`
- `has_salesRank rate = 0.981191`

平均信息量：

- `avg title_len = 65.14`
- `avg desc_len = 432.9`
- `avg title_tokens = 10.47`
- `avg desc_tokens = 67.11`
- `avg category_depth = 4.11`

#### 4.3 COLD 集合核查结果

- `total items = 568`
- `covered by meta = 568`
- `coverage rate = 1.0`
- `missing meta rows = 0`

字段覆盖率：

- `has_title rate = 0.998239`
- `has_desc rate = 0.929577`
- `has_image rate = 0.998239`
- `has_categories rate = 1.0`
- `has_salesRank rate = 0.991197`

平均信息量：

- `avg title_len = 118.18`
- `avg desc_len = 616.27`
- `avg title_tokens = 19.09`
- `avg desc_tokens = 97.1`
- `avg category_depth = 4.14`

#### 4.4 embedding 对齐结果

- `warm map len = 11165`
- `warm emb shape = (11165, 768)`
- `warm aligned = True`

- `cold map len = 568`
- `cold emb shape = (568, 768)`
- `cold aligned = True`

这说明：

- warm embedding 行数与 warm item 映射严格一致
- cold embedding 行数与 cold item 映射严格一致

---

### 5. 核心结论

### 5.1 关于数据关系

已经可以明确说：

> `meta_Beauty.json`、`Beauty_5.json`、8 个处理后文件之间，数据关系已经基本打通。

特别是：

- `meta_Beauty.json` 提供 item-side 元信息；
- `item2index_warm/cold.pkl` 负责把原始 `asin` 映射成处理后使用的数值 `item_id`；
- `processed/*.parquet` 使用该数值索引；
- `embeddings_warm/cold.npy` 与 warm/cold 映射长度严格一致。

因此，从“原始 item 元信息”到“模型输入文件”的桥梁已经找到。

---

### 5.2 关于 quality score v1 是否可做

最终结论是：

> **Beauty 数据集上，quality score v1 完全可做。**

而且不是“勉强能做”，而是：

- metadata 对 warm / cold item 的覆盖率都是 **100%**
- `title`、`image`、`categories` 几乎全覆盖
- `description` 覆盖率也超过 **92%**
- embedding 与 item 映射严格对齐

因此，构造一个基于 side information 完整度与信息量的规则型 `quality score v1`，在数据层面已经没有实质性障碍。

---

### 6. 研究层面的重要理解

这次核查还有一个重要启发：

> `cold item` 不一定意味着 `side information` 更差。

因为从统计上看，当前数据中：

- cold item 的平均 `title_len` 更长
- cold item 的平均 `desc_len` 更长
- cold item 的 `has_desc rate` 甚至略高

所以更合理的理解是：

> `quality score` 衡量的是 **item-level 的 side information 质量**，而不是简单地把它等同于 cold/warm 标签。

这意味着：

- 有些 cold item 质量很高
- 有些 warm item 质量很差

这正好支持后续做 `quality-aware` 设计。

---

### 7. 对 quality score v1 的建议

当前最稳妥的 `v1`，建议只使用以下字段：

1. `title`
   - 是否存在
   - 长度 / token 数

2. `description`
   - 是否存在
   - 长度 / token 数

3. `categories`
   - 是否存在
   - 层级深度

4. `imUrl`
   - 是否存在

### 暂不建议纳入：

- `salesRank`

原因：

- 虽然它覆盖率很高；
- 但它更接近“热门程度 / 销量信息”；
- 容易污染“quality score = side-info 质量”的研究叙事。

因此第一版更适合只保留：

- `title`
- `description`
- `categories`
- `image`

---

### 8. 当前阶段性结论

可以把本阶段结果概括为：

> 当前已经证明：Beauty 数据集在数据覆盖、字段可用性、ID 对齐关系、embedding 对齐关系四个层面，都足以支持一个规则型 `quality score v1` 的构建。

换句话说：

> 现在的问题已经不是“能不能做”，而是“quality score v1 具体怎么定义最合理”。

---

### 9. 下一步工作建议

下一步应直接进入实现阶段，编写 `build_quality_score_v1.py`，完成以下三件事：

1. 从 `meta_Beauty.json` 中抽取 item-side 质量特征；
2. 为每个 `asin` 计算 quality score；
3. 按 `item2index_warm.pkl` / `item2index_cold.pkl` 的索引顺序导出：
   - `warm_quality.npy`
   - `cold_quality.npy`

至此，quality score v1 就可以正式接入后续模型实验。

---

## 一句话总总结

> **Beauty 数据已经足够支撑 quality score v1；当前最关键的数据桥梁（asin → mapping → numeric item_id → embedding）已经打通，可以直接进入质量分数实现阶段。**