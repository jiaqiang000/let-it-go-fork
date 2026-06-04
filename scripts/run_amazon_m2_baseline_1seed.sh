#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

# 本脚本只用于打通 Amazon-M2 的 Let It Go 原版 baseline。
# 目的：不改 run.py / 模型 / evaluation，先确认 Amazon-M2 数据集能正常训练和评测。

if [[ -z "${PYTHON_BIN:-}" && -f /root/letitgo-runtime/.venv/bin/activate ]]; then
  # 服务器默认兼容：如果用户没有显式指定 PYTHON_BIN，就沿用服务器已配置的 venv。
  # 本地 smoke test 通常会显式传 PYTHON_BIN=/opt/anaconda3/envs/let-it-go-py3.11/bin/python。
  source /root/letitgo-runtime/.venv/bin/activate
fi

PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-42}
DATA_ROOT=${DATA_ROOT:-/root/letitgo-data/data/amazon_m2_fr}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/hy-tmp/letitgo_ckpt/amazon_m2_baseline_20260604}
PROJECT_NAME=${PROJECT_NAME:-letitgo_amazon_m2_baseline}
MAX_EPOCHS=${MAX_EPOCHS:-100}
RUN_GROUPS=${RUN_GROUPS:-A2}
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

  # 本地 Obsidian 路径包含空格，Hydra override 需要显式 quoted string。
  # 这里统一把路径/字符串值包装成 key='value'，避免本地 smoke test 解析失败。
  printf "%s='%s'" "$key" "$value"
}

COMMON_ARGS=(
  dataset=amazon_m2
  "$(hydra_quoted_arg project_name "$PROJECT_NAME")"
  "$(hydra_quoted_arg checkpoint_dir "$CHECKPOINT_DIR")"
  "$(hydra_quoted_arg dataset.train_filepath "$DATA_ROOT/processed/train_interactions.parquet")"
  "$(hydra_quoted_arg dataset.val_filepath "$DATA_ROOT/processed/val_interactions.parquet")"
  "$(hydra_quoted_arg dataset.test_filepath "$DATA_ROOT/processed/test_interactions.parquet")"
  "$(hydra_quoted_arg dataset.gt_filepath "$DATA_ROOT/processed/ground_truth.parquet")"
  "$(hydra_quoted_arg dataset.item_embeddings.warm "$DATA_ROOT/item_embeddings/embeddings_warm.npy")"
  "$(hydra_quoted_arg dataset.item_embeddings.cold "$DATA_ROOT/item_embeddings/embeddings_cold.npy")"
  trainer.max_epochs="$MAX_EPOCHS"
)

run_cmd() {
  echo
  echo ">>> $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

group_enabled() {
  local group=$1
  [[ ",$RUN_GROUPS," == *",$group,"* ]]
}

print_stage() {
  local current=$1
  local total=$2
  local message=$3
  echo
  echo "===== [${current}/${total}] ${message} ====="
}

GROUPS_TO_RUN=()
IFS=',' read -ra REQUESTED_GROUPS <<< "$RUN_GROUPS"
for group in "${REQUESTED_GROUPS[@]}"; do
  case "$group" in
    A0|A1|A2)
      GROUPS_TO_RUN+=("$group")
      ;;
    "")
      ;;
    *)
      echo "不支持的 RUN_GROUPS 分组：$group。当前脚本只支持 A0,A1,A2。" >&2
      exit 1
      ;;
  esac
done

if [[ ${#GROUPS_TO_RUN[@]} -eq 0 ]]; then
  echo "RUN_GROUPS 不能为空。当前脚本支持 A0,A1,A2。" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

echo "Amazon-M2 Let It Go baseline single-seed run"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SEED=$SEED"
echo "DATA_ROOT=$DATA_ROOT"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "RUN_GROUPS=$RUN_GROUPS"
echo "DRY_RUN=$DRY_RUN"
echo "CHECK_ONLY=$CHECK_ONLY"

if [[ "$CHECK_ONLY" == "1" ]]; then
  print_stage 1 1 "检查 Amazon-M2 数据、embedding、模型入口，不启动训练"
  # 本地轻量 smoke test：只验证路径、数据形状、embedding 形状、模型类和数据集构建。
  # 目的：确认 Amazon-M2 入口没有接错；正式训练仍交给服务器。
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" RUN_GROUPS="$RUN_GROUPS" "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf


project_root = Path(os.environ["PROJECT_ROOT"])
data_root = Path(os.environ["DATA_ROOT"])
groups = {group.strip() for group in os.environ["RUN_GROUPS"].split(",") if group.strip()}

paths = {
    "train": data_root / "processed" / "train_interactions.parquet",
    "val": data_root / "processed" / "val_interactions.parquet",
    "test": data_root / "processed" / "test_interactions.parquet",
    "ground_truth": data_root / "processed" / "ground_truth.parquet",
    "warm_embeddings": data_root / "item_embeddings" / "embeddings_warm.npy",
    "cold_embeddings": data_root / "item_embeddings" / "embeddings_cold.npy",
}

for name, path in paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name} 文件不存在：{path}")

frames = {
    name: pl.read_parquet(path)
    for name, path in paths.items()
    if name in {"train", "val", "test", "ground_truth"}
}
warm_embeddings = np.load(paths["warm_embeddings"])
cold_embeddings = np.load(paths["cold_embeddings"])

print("CHECK_ONLY: 数据文件存在。")
for name, frame in frames.items():
    print(f"  {name}: shape={frame.shape}, columns={frame.columns}")
    if "item_id" in frame.columns:
        item_min = frame.select(pl.col("item_id").min()).item()
        item_max = frame.select(pl.col("item_id").max()).item()
        print(f"    item_id_min={item_min}, item_id_max={item_max}")

print(f"  warm_embeddings_shape={warm_embeddings.shape}")
print(f"  cold_embeddings_shape={cold_embeddings.shape}")

if warm_embeddings.ndim != 2 or cold_embeddings.ndim != 2:
    raise ValueError("Amazon-M2 item embeddings 必须是二维矩阵。")
if warm_embeddings.shape[0] != 42647:
    raise ValueError(f"warm embedding 行数应为 42647，实际为 {warm_embeddings.shape[0]}")
if warm_embeddings.shape[1] != cold_embeddings.shape[1]:
    raise ValueError(
        "warm/cold embedding 维度不一致："
        f"{warm_embeddings.shape[1]} vs {cold_embeddings.shape[1]}"
    )

os.chdir(project_root / "scripts")
sys.path.insert(0, str(project_root / "scripts"))

from run import get_datamodule, get_model  # noqa: E402


datamodule_config = OmegaConf.create(
    {
        "dataset": {
            "train_filepath": str(paths["train"]),
            "val_filepath": str(paths["val"]),
            "max_length": 64,
        },
    }
)

datamodule = get_datamodule(datamodule_config)
datamodule.setup("fit")

print(f"  train_sequences={len(datamodule.train_dataset)}")
print(f"  val_sequences={len(datamodule.val_dataset)}")

group_configs = {
    "A0": {
        "use_pretrained_item_embeddings": False,
        "train_delta": False,
        "description": "init(rand) / 普通 SASRec",
    },
    "A1": {
        "use_pretrained_item_embeddings": True,
        "train_delta": False,
        "description": "init(text) / content initialization",
    },
    "A2": {
        "use_pretrained_item_embeddings": True,
        "train_delta": True,
        "description": "init(text)-delta-0.5 / trainable delta",
    },
}

for group in ("A0", "A1", "A2"):
    if group not in groups:
        continue

    config = OmegaConf.create(
        {
            "use_pretrained_item_embeddings": group_configs[group]["use_pretrained_item_embeddings"],
            "train_delta": group_configs[group]["train_delta"],
            "quality_aware_delta": False,
            "max_delta_norm": 0.5,
            "dataset": {
                "train_filepath": str(paths["train"]),
                "val_filepath": str(paths["val"]),
                "max_length": 64,
            },
            "model": {
                "num_items": 42647,
                "embedding_dim": 64,
                "num_blocks": 2,
                "num_heads": 1,
                "p": 0.3,
                "max_length": 64,
            },
        }
    )
    model = get_model(config)
    print(f"  {group}: {group_configs[group]['description']}")
    print(f"    model={model.__class__.__name__}")
    print(f"    use_pretrained_item_embeddings={config.use_pretrained_item_embeddings}")
    print(f"    train_delta={config.train_delta}")

print("CHECK_ONLY: Amazon-M2 baseline 入口检查通过。")
PY
  exit 0
fi

i=0
total=${#GROUPS_TO_RUN[@]}

if group_enabled A0; then
  i=$((i + 1))
  # A0：普通 SASRec baseline。
  # 目的：确认没有 content initialization 和 delta 时的 Amazon-M2 基础参照。
  print_stage "$i" "$total" "A0: init(rand) / 普通 SASRec；后续 epoch/batch 进度由 Lightning 打印"
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}"
fi

if group_enabled A1; then
  i=$((i + 1))
  # A1：content initialization。
  # 目的：确认作者给好的 Amazon-M2 content embeddings 进入 SASRec 后的参照。
  print_stage "$i" "$total" "A1: init(text) / content initialization；后续 epoch/batch 进度由 Lightning 打印"
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True
fi

if group_enabled A2; then
  i=$((i + 1))
  # A2：Amazon-M2 原版 Let It Go baseline：content initialization + trainable delta。
  # 这组对应 init(text)-delta-0.5，用来对齐论文主结果，不加入 q-aware 或字段扰动。
  print_stage "$i" "$total" "A2: init(text)-delta-0.5 / trainable delta；后续 epoch/batch 进度由 Lightning 打印"
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    max_delta_norm=0.5
fi
