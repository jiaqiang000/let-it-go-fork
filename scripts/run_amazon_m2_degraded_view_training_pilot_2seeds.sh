#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Amazon-M2 degraded-view training pilot。
# 本脚本遵循 full-pipeline retrain 的服务器沉淀方式：
# 1. 生成 degraded-view warm/cold embeddings，并单独记录 embedding 生成日志；
# 2. 从 scripts/ 目录调用 run.py，避免 source 导入失败；
# 3. 对 control_full/title_trunc_8/random_title_dropout_p30/no_title 跑 2 seeds；
# 4. 结束时打印必须下载的日志、manifest、ClearML offline zip 和可选 checkpoint。

if [[ -z "${PYTHON_BIN:-}" && -f /root/letitgo-runtime/.venv/bin/activate ]]; then
  # 服务器默认兼容：如果用户没有显式指定 PYTHON_BIN，就沿用服务器 venv。
  source /root/letitgo-runtime/.venv/bin/activate
fi

if [[ -z "${PYTHON_BIN:-}" && -x /opt/anaconda3/envs/let-it-go-py3.11/bin/python ]]; then
  PYTHON_BIN=/opt/anaconda3/envs/let-it-go-py3.11/bin/python
fi

PYTHON_BIN=${PYTHON_BIN:-python}
DATE_TAG=${DATE_TAG:-$(date +%Y%m%d)}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
DATA_ROOT=${DATA_ROOT:-/root/letitgo-data/data/amazon_m2_fr}
PRODUCTS_PATH=${PRODUCTS_PATH:-$PROJECT_ROOT/row_data/amazon_m2_raw/products_train.csv}
LOG_DIR=${LOG_DIR:-/hy-tmp/letitgo_logs}
RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR:-/hy-tmp/letitgo_outputs/amazon_m2_degraded_view_training_pilot_$DATE_TAG}
VARIANT_EMBEDDING_ROOT=${VARIANT_EMBEDDING_ROOT:-/hy-tmp/letitgo_outputs/amazon_m2_degraded_view_training_embeddings_$DATE_TAG}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/hy-tmp/letitgo_ckpt/amazon_m2_degraded_view_training_pilot_$DATE_TAG}
PROJECT_NAME=${PROJECT_NAME:-letitgo_amazon_m2_degraded_view_training_pilot}
MAX_EPOCHS=${MAX_EPOCHS:-100}
VARIANTS=${VARIANTS:-control_full,title_trunc_8,random_title_dropout_p30,no_title}
SEEDS=${SEEDS:-42,43}
TRAINER_DEVICES=${TRAINER_DEVICES:-[0]}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
MAX_DELTA_NORM=${MAX_DELTA_NORM:-0.5}
SENTENCE_CHECKPOINT=${SENTENCE_CHECKPOINT:-intfloat/multilingual-e5-base}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-256}
DRY_RUN=${DRY_RUN:-0}
CHECK_ONLY=${CHECK_ONLY:-0}
SKIP_EMBEDDING=${SKIP_EMBEDDING:-0}

export CLEARML_OFFLINE_MODE=${CLEARML_OFFLINE_MODE:-1}

EMBEDDING_LOG=$LOG_DIR/build_amazon_m2_degraded_view_training_embeddings_$RUN_TS.log
TRAINING_LOG=$LOG_DIR/amazon_m2_degraded_view_training_pilot_$RUN_TS.log
RUN_MANIFEST=$RUN_OUTPUT_DIR/server_run_manifest_$RUN_TS.json

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

write_server_manifest() {
  RUN_MANIFEST="$RUN_MANIFEST" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  PRODUCTS_PATH="$PRODUCTS_PATH" \
  LOG_DIR="$LOG_DIR" \
  RUN_OUTPUT_DIR="$RUN_OUTPUT_DIR" \
  VARIANT_EMBEDDING_ROOT="$VARIANT_EMBEDDING_ROOT" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  PROJECT_NAME="$PROJECT_NAME" \
  VARIANTS="$VARIANTS" \
  SEEDS="$SEEDS" \
  MAX_EPOCHS="$MAX_EPOCHS" \
  TRAINER_DEVICES="$TRAINER_DEVICES" \
  EMBEDDING_LOG="$EMBEDDING_LOG" \
  TRAINING_LOG="$TRAINING_LOG" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "script_role": "server runner for Amazon-M2 degraded-view training pilot",
    "project_root": os.environ["PROJECT_ROOT"],
    "data_root": os.environ["DATA_ROOT"],
    "products_path": os.environ["PRODUCTS_PATH"],
    "log_dir": os.environ["LOG_DIR"],
    "run_output_dir": os.environ["RUN_OUTPUT_DIR"],
    "variant_embedding_root": os.environ["VARIANT_EMBEDDING_ROOT"],
    "checkpoint_dir": os.environ["CHECKPOINT_DIR"],
    "project_name": os.environ["PROJECT_NAME"],
    "variants": [item.strip() for item in os.environ["VARIANTS"].split(",") if item.strip()],
    "seeds": [int(item.strip()) for item in os.environ["SEEDS"].split(",") if item.strip()],
    "max_epochs": int(os.environ["MAX_EPOCHS"]),
    "trainer_devices": os.environ["TRAINER_DEVICES"],
    "embedding_log": os.environ["EMBEDDING_LOG"],
    "training_log": os.environ["TRAINING_LOG"],
    "must_download": [
        os.environ["TRAINING_LOG"],
        os.environ["EMBEDDING_LOG"],
        str(Path(os.environ["VARIANT_EMBEDDING_ROOT"]) / "run_manifest.json"),
        str(Path(os.environ["VARIANT_EMBEDDING_ROOT"]) / "variant_embedding_summary.csv"),
        str(Path(os.environ["VARIANT_EMBEDDING_ROOT"]) / "degraded_view_training_profile_summary.csv"),
        str(Path(os.environ["VARIANT_EMBEDDING_ROOT"]) / "degraded_view_training_item_profile.csv"),
        "/root/.clearml/cache/offline/offline-*.zip",
    ],
    "optional_download": [os.environ["CHECKPOINT_DIR"]],
    "not_recommended_download": [
        str(Path(os.environ["VARIANT_EMBEDDING_ROOT"]) / "*" / "item_embeddings" / "*.npy")
    ],
}

path = Path(os.environ["RUN_MANIFEST"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"server manifest: {path}")
PY
}

check_only() {
  echo "Amazon-M2 degraded-view training pilot CHECK_ONLY"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "PRODUCTS_PATH=$PRODUCTS_PATH"
  echo "VARIANT_EMBEDDING_ROOT=$VARIANT_EMBEDDING_ROOT"
  echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
  echo "VARIANTS=$VARIANTS"
  echo "SEEDS=$SEEDS"

  PROJECT_ROOT="$PROJECT_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  PRODUCTS_PATH="$PRODUCTS_PATH" \
  VARIANT_EMBEDDING_ROOT="$VARIANT_EMBEDDING_ROOT" \
  VARIANTS="$VARIANTS" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(os.environ["PROJECT_ROOT"])
data_root = Path(os.environ["DATA_ROOT"])
products_path = Path(os.environ["PRODUCTS_PATH"])
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
if not products_path.is_file():
    raise FileNotFoundError(f"products_train.csv 不存在：{products_path}")
if not (project_root / "validata" / "build_amazon_m2_degraded_view_training_embeddings.py").is_file():
    raise FileNotFoundError("缺少 degraded-view embedding builder。")
if not (project_root / "scripts" / "run.py").is_file():
    raise FileNotFoundError("缺少 scripts/run.py。")

print("CHECK_ONLY: processed 数据文件存在。")
for name, path in required_data.items():
    frame = pl.read_parquet(path)
    print(f"  {name}: shape={frame.shape}, columns={frame.columns}")

print("CHECK_ONLY: 如果 embeddings 已存在，则检查 shape；不存在则仅提示，非失败。")
for variant in variants:
    warm_path = embedding_root / variant / "item_embeddings" / "embeddings_warm.npy"
    cold_path = embedding_root / variant / "item_embeddings" / "embeddings_cold.npy"
    if not warm_path.is_file() or not cold_path.is_file():
        print(f"  {variant}: embeddings 尚未生成。")
        continue
    warm = np.load(warm_path)
    cold = np.load(cold_path)
    print(f"  {variant}: warm_shape={warm.shape}, cold_shape={cold.shape}")
    if warm.shape != (42647, 768):
        raise ValueError(f"{variant} warm embedding shape 应为 (42647, 768)，实际 {warm.shape}")
    if cold.shape != (1402, 768):
        raise ValueError(f"{variant} cold embedding shape 应为 (1402, 768)，实际 {cold.shape}")
    if warm.shape[1] != cold.shape[1]:
        raise ValueError(f"{variant} warm/cold embedding 维度不一致。")

print("CHECK_ONLY: degraded-view training pilot 入口检查通过。")
PY
}

print_download_checklist() {
  echo
  echo "===== DOWNLOAD CHECKLIST ====="
  echo
  echo "必须下载："
  echo "$TRAINING_LOG"
  echo "$EMBEDDING_LOG"
  echo "$VARIANT_EMBEDDING_ROOT/run_manifest.json"
  echo "$VARIANT_EMBEDDING_ROOT/variant_embedding_summary.csv"
  echo "$VARIANT_EMBEDDING_ROOT/degraded_view_training_profile_summary.csv"
  echo "$VARIANT_EMBEDDING_ROOT/degraded_view_training_item_profile.csv"
  echo "$RUN_MANIFEST"
  echo "/root/.clearml/cache/offline/offline-*.zip"
  echo
  echo "可选下载："
  echo "$CHECKPOINT_DIR"
  echo
  echo "不建议现在下载："
  echo "$VARIANT_EMBEDDING_ROOT/*/item_embeddings/*.npy"
  echo
  echo "最近 ClearML offline zip 候选："
  ls -t /root/.clearml/cache/offline/offline-*.zip 2>/dev/null | head -n 20 || true
}

IFS=',' read -ra RAW_VARIANT_LIST <<< "$VARIANTS"
VARIANT_LIST=()
for item in "${RAW_VARIANT_LIST[@]}"; do
  item=$(echo "$item" | xargs)
  if [[ -n "$item" ]]; then
    VARIANT_LIST+=("$item")
  fi
done

IFS=',' read -ra RAW_SEED_LIST <<< "$SEEDS"
SEED_LIST=()
for item in "${RAW_SEED_LIST[@]}"; do
  item=$(echo "$item" | xargs)
  if [[ -n "$item" ]]; then
    SEED_LIST+=("$item")
  fi
done

if [[ ${#VARIANT_LIST[@]} -eq 0 ]]; then
  echo "VARIANTS 不能为空。" >&2
  exit 1
fi
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "SEEDS 不能为空。" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$RUN_OUTPUT_DIR" "$VARIANT_EMBEDDING_ROOT" "$CHECKPOINT_DIR"
write_server_manifest

if [[ "$CHECK_ONLY" == "1" ]]; then
  check_only
  print_download_checklist
  exit 0
fi

echo "Amazon-M2 degraded-view training pilot"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DATE_TAG=$DATE_TAG"
echo "RUN_TS=$RUN_TS"
echo "DATA_ROOT=$DATA_ROOT"
echo "PRODUCTS_PATH=$PRODUCTS_PATH"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_OUTPUT_DIR=$RUN_OUTPUT_DIR"
echo "VARIANT_EMBEDDING_ROOT=$VARIANT_EMBEDDING_ROOT"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "TRAINER_DEVICES=$TRAINER_DEVICES"
echo "VARIANTS=$VARIANTS"
echo "SEEDS=$SEEDS"
echo "DRY_RUN=$DRY_RUN"
echo "SKIP_EMBEDDING=$SKIP_EMBEDDING"
echo "EMBEDDING_LOG=$EMBEDDING_LOG"
echo "TRAINING_LOG=$TRAINING_LOG"

if [[ "$SKIP_EMBEDDING" != "1" ]]; then
  {
    echo "Amazon-M2 degraded-view embedding generation"
    echo "created_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "VARIANT_EMBEDDING_ROOT=$VARIANT_EMBEDDING_ROOT"
    run_cmd "$PYTHON_BIN" "$PROJECT_ROOT/validata/build_amazon_m2_degraded_view_training_embeddings.py" \
      --data-root "$DATA_ROOT" \
      --products-path "$PRODUCTS_PATH" \
      --output-dir "$VARIANT_EMBEDDING_ROOT" \
      --variants "$VARIANTS" \
      --sentence-checkpoint "$SENTENCE_CHECKPOINT" \
      --encode-batch-size "$ENCODE_BATCH_SIZE"
  } 2>&1 | tee "$EMBEDDING_LOG"
else
  echo "SKIP_EMBEDDING=1: 跳过 embedding 生成，使用已有目录 $VARIANT_EMBEDDING_ROOT"
fi

{
  echo "Amazon-M2 degraded-view training matrix"
  echo "created_at=$(date '+%Y-%m-%d %H:%M:%S')"
  cd "$SCRIPT_DIR"

  i=0
  total=$(( ${#VARIANT_LIST[@]} * ${#SEED_LIST[@]} ))
  for variant in "${VARIANT_LIST[@]}"; do
    warm_embedding="$VARIANT_EMBEDDING_ROOT/$variant/item_embeddings/embeddings_warm.npy"
    cold_embedding="$VARIANT_EMBEDDING_ROOT/$variant/item_embeddings/embeddings_cold.npy"
    if [[ "$DRY_RUN" != "1" ]]; then
      if [[ ! -f "$warm_embedding" ]]; then
        echo "$variant warm embedding 不存在：$warm_embedding" >&2
        exit 1
      fi
      if [[ ! -f "$cold_embedding" ]]; then
        echo "$variant cold embedding 不存在：$cold_embedding" >&2
        exit 1
      fi
    fi

    for seed in "${SEED_LIST[@]}"; do
      i=$((i + 1))
      echo
      echo "===== [${i}/${total}] variant=${variant} seed=${seed} degraded-view training START ====="
      run_cmd "$PYTHON_BIN" run.py \
        seed="$seed" \
        dataset=amazon_m2 \
        "$(hydra_quoted_arg project_name "$PROJECT_NAME/$variant/seed_$seed")" \
        "$(hydra_quoted_arg checkpoint_dir "$CHECKPOINT_DIR/$variant/seed_$seed")" \
        "$(hydra_quoted_arg dataset.train_filepath "$DATA_ROOT/processed/train_interactions.parquet")" \
        "$(hydra_quoted_arg dataset.val_filepath "$DATA_ROOT/processed/val_interactions.parquet")" \
        "$(hydra_quoted_arg dataset.test_filepath "$DATA_ROOT/processed/test_interactions.parquet")" \
        "$(hydra_quoted_arg dataset.gt_filepath "$DATA_ROOT/processed/ground_truth.parquet")" \
        "$(hydra_quoted_arg dataset.item_embeddings.warm "$warm_embedding")" \
        "$(hydra_quoted_arg dataset.item_embeddings.cold "$cold_embedding")" \
        trainer.max_epochs="$MAX_EPOCHS" \
        "trainer.devices=$TRAINER_DEVICES" \
        recommender.learning_rate="$LEARNING_RATE" \
        use_pretrained_item_embeddings=True \
        train_delta=True \
        quality_aware_delta=False \
        max_delta_norm="$MAX_DELTA_NORM"
      echo "===== [${i}/${total}] variant=${variant} seed=${seed} degraded-view training DONE ====="
    done
  done
} 2>&1 | tee "$TRAINING_LOG"

print_download_checklist
