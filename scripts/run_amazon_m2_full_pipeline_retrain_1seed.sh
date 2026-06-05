#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

# 本脚本用于 Amazon-M2 受控字段 full-pipeline retrain。
# 它不生成 embedding，只读取 validata/build_amazon_m2_full_pipeline_variant_embeddings.py
# 已经生成好的 warm/cold variant embeddings，然后对每个 variant 重新训练 A2。

if [[ -z "${PYTHON_BIN:-}" && -f /root/letitgo-runtime/.venv/bin/activate ]]; then
  # 服务器默认兼容：如果用户没有显式指定 PYTHON_BIN，就沿用服务器 venv。
  source /root/letitgo-runtime/.venv/bin/activate
fi

PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-42}
DATA_ROOT=${DATA_ROOT:-/root/letitgo-data/data/amazon_m2_fr}
VARIANT_EMBEDDING_ROOT=${VARIANT_EMBEDDING_ROOT:-/hy-tmp/letitgo_outputs/amazon_m2_full_pipeline_variant_embeddings_20260606}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/hy-tmp/letitgo_ckpt/amazon_m2_full_pipeline_retrain_20260606}
PROJECT_NAME=${PROJECT_NAME:-letitgo_amazon_m2_full_pipeline_retrain}
MAX_EPOCHS=${MAX_EPOCHS:-100}
VARIANTS=${VARIANTS:-control_full,drop_four}
DRY_RUN=${DRY_RUN:-0}
CHECK_ONLY=${CHECK_ONLY:-0}

export CLEARML_OFFLINE_MODE=${CLEARML_OFFLINE_MODE:-1}

hydra_quoted_arg() {
  local key=$1
  local value=$2
  if [[ "$value" == *"'"* ]]; then
    echo "Hydra 参数暂不支持包含单引号的路径：$value" >&2
    return 1
  fi

  # 中文注释：本地 Obsidian 路径包含空格，Hydra override 需要显式 quoted string。
  printf "%s='%s'" "$key" "$value"
}

run_cmd() {
  echo
  echo ">>> $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

IFS=',' read -ra VARIANT_LIST <<< "$VARIANTS"
FILTERED_VARIANTS=()
for variant in "${VARIANT_LIST[@]}"; do
  variant=$(echo "$variant" | xargs)
  if [[ -n "$variant" ]]; then
    FILTERED_VARIANTS+=("$variant")
  fi
done

if [[ ${#FILTERED_VARIANTS[@]} -eq 0 ]]; then
  echo "VARIANTS 不能为空，例如 VARIANTS=control_full,drop_four。" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

echo "Amazon-M2 full-pipeline retrain single-seed run"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SEED=$SEED"
echo "DATA_ROOT=$DATA_ROOT"
echo "VARIANT_EMBEDDING_ROOT=$VARIANT_EMBEDDING_ROOT"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "VARIANTS=$VARIANTS"
echo "DRY_RUN=$DRY_RUN"
echo "CHECK_ONLY=$CHECK_ONLY"

if [[ "$CHECK_ONLY" == "1" ]]; then
  # 中文注释：只检查每个 variant 的 embedding 文件是否存在、shape 是否和 Amazon-M2 对齐。
  PROJECT_ROOT="$PROJECT_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  VARIANT_EMBEDDING_ROOT="$VARIANT_EMBEDDING_ROOT" \
  VARIANTS="$VARIANTS" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import numpy as np
import polars as pl


project_root = Path(os.environ["PROJECT_ROOT"])
data_root = Path(os.environ["DATA_ROOT"])
embedding_root = Path(os.environ["VARIANT_EMBEDDING_ROOT"])
variants = [item.strip() for item in os.environ["VARIANTS"].split(",") if item.strip()]

required_data = {
    "train": data_root / "processed" / "train_interactions.parquet",
    "val": data_root / "processed" / "val_interactions.parquet",
    "test": data_root / "processed" / "test_interactions.parquet",
    "ground_truth": data_root / "processed" / "ground_truth.parquet",
}
for name, path in required_data.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name} 文件不存在：{path}")

print("CHECK_ONLY: processed 数据文件存在。")
for name, path in required_data.items():
    frame = pl.read_parquet(path)
    print(f"  {name}: shape={frame.shape}, columns={frame.columns}")

for variant in variants:
    warm_path = embedding_root / variant / "item_embeddings" / "embeddings_warm.npy"
    cold_path = embedding_root / variant / "item_embeddings" / "embeddings_cold.npy"
    if not warm_path.is_file():
        raise FileNotFoundError(f"{variant} warm embedding 不存在：{warm_path}")
    if not cold_path.is_file():
        raise FileNotFoundError(f"{variant} cold embedding 不存在：{cold_path}")

    warm = np.load(warm_path)
    cold = np.load(cold_path)
    print(f"  {variant}: warm_shape={warm.shape}, cold_shape={cold.shape}")
    if warm.shape != (42647, 768):
        raise ValueError(f"{variant} warm embedding shape 应为 (42647, 768)，实际 {warm.shape}")
    if cold.shape != (1402, 768):
        raise ValueError(f"{variant} cold embedding shape 应为 (1402, 768)，实际 {cold.shape}")
    if warm.shape[1] != cold.shape[1]:
        raise ValueError(f"{variant} warm/cold embedding 维度不一致。")

print("CHECK_ONLY: full-pipeline retrain 入口检查通过。")
PY
  exit 0
fi

i=0
total=${#FILTERED_VARIANTS[@]}
for variant in "${FILTERED_VARIANTS[@]}"; do
  i=$((i + 1))
  warm_embedding="$VARIANT_EMBEDDING_ROOT/$variant/item_embeddings/embeddings_warm.npy"
  cold_embedding="$VARIANT_EMBEDDING_ROOT/$variant/item_embeddings/embeddings_cold.npy"

  if [[ ! -f "$warm_embedding" ]]; then
    echo "$variant warm embedding 不存在：$warm_embedding" >&2
    exit 1
  fi
  if [[ ! -f "$cold_embedding" ]]; then
    echo "$variant cold embedding 不存在：$cold_embedding" >&2
    exit 1
  fi

  echo
  echo "===== [${i}/${total}] variant=${variant} A2 full-pipeline retrain START ====="

  # 中文注释：这里是完整重训 A2。run.py 会用该 variant 的 warm embedding 重新 fit PCA/Normalizer，
  # 再在评测阶段追加同一 variant 的 cold embedding。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    dataset=amazon_m2 \
    "$(hydra_quoted_arg project_name "$PROJECT_NAME/$variant")" \
    "$(hydra_quoted_arg checkpoint_dir "$CHECKPOINT_DIR/$variant")" \
    "$(hydra_quoted_arg dataset.train_filepath "$DATA_ROOT/processed/train_interactions.parquet")" \
    "$(hydra_quoted_arg dataset.val_filepath "$DATA_ROOT/processed/val_interactions.parquet")" \
    "$(hydra_quoted_arg dataset.test_filepath "$DATA_ROOT/processed/test_interactions.parquet")" \
    "$(hydra_quoted_arg dataset.gt_filepath "$DATA_ROOT/processed/ground_truth.parquet")" \
    "$(hydra_quoted_arg dataset.item_embeddings.warm "$warm_embedding")" \
    "$(hydra_quoted_arg dataset.item_embeddings.cold "$cold_embedding")" \
    trainer.max_epochs="$MAX_EPOCHS" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    max_delta_norm=0.5

  echo "===== [${i}/${total}] variant=${variant} A2 full-pipeline retrain DONE ====="
done
