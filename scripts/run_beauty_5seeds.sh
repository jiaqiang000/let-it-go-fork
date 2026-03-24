#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 激活已经配好的虚拟环境
source /root/letitgo-runtime/.venv/bin/activate

# ClearML 使用离线模式
export CLEARML_OFFLINE_MODE=1

# 统一的 Beauty 数据路径和输出路径
COMMON_ARGS="
dataset=beauty
project_name=letitgo
checkpoint_dir=/hy-tmp/letitgo_ckpt
dataset.train_filepath=/root/letitgo-data/data/beauty/processed/train_interactions.parquet
dataset.val_filepath=/root/letitgo-data/data/beauty/processed/val_interactions.parquet
dataset.test_filepath=/root/letitgo-data/data/beauty/processed/test_interactions.parquet
dataset.gt_filepath=/root/letitgo-data/data/beauty/processed/ground_truth.parquet
dataset.item_embeddings.warm=/root/letitgo-data/data/beauty/item_embeddings/embeddings_warm.npy
dataset.item_embeddings.cold=/root/letitgo-data/data/beauty/item_embeddings/embeddings_cold.npy
"

QUALITY_WARM=$PROJECT_ROOT/quality_score_output/beauty/warm_quality.npy
QUALITY_COLD=$PROJECT_ROOT/quality_score_output/beauty/cold_quality.npy

# 确保 checkpoint 目录存在
mkdir -p /hy-tmp/letitgo_ckpt

# 1) baseline：普通 SASRec
python run.py -m seed=42,221,451,934,1984 $COMMON_ARGS

# 2) content initialization：内容向量初始化
python run.py -m seed=42,221,451,934,1984 $COMMON_ARGS use_pretrained_item_embeddings=True

# 3) trainable delta：论文核心方法
python run.py -m seed=42,221,451,934,1984 $COMMON_ARGS use_pretrained_item_embeddings=True train_delta=True

# 4) quality-aware delta：如果已经生成了 quality score v1，就追加这一组
if [[ -f "$QUALITY_WARM" && -f "$QUALITY_COLD" ]]; then
python run.py -m seed=42,221,451,934,1984 $COMMON_ARGS \
  use_pretrained_item_embeddings=True \
  train_delta=True \
  quality_aware_delta=True \
  quality_score.warm_filepath=$QUALITY_WARM \
  quality_score.cold_filepath=$QUALITY_COLD
fi
