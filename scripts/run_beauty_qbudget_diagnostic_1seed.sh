#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

# 本脚本是 q-budget diagnostic 的隔离运行脚本。
# 目的：不改旧 run_beauty_1seed.sh / run_beauty_5seeds.sh，也不改训练主代码。

if [[ -z "${PYTHON_BIN:-}" && -f /root/letitgo-runtime/.venv/bin/activate ]]; then
  # 服务器默认兼容：如果用户没有显式指定 PYTHON_BIN，就沿用旧脚本的 venv。
  # 本地 smoke test 通常会显式传 PYTHON_BIN，因此不会进入这里。
  source /root/letitgo-runtime/.venv/bin/activate
fi

PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-42}
DATA_ROOT=${DATA_ROOT:-/root/letitgo-data/data/beauty}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/hy-tmp/letitgo_ckpt/qbudget_diagnostic_20260604}
PROJECT_NAME=${PROJECT_NAME:-letitgo_qbudget_diagnostic}
CONTROL_DIR=${CONTROL_DIR:-$PROJECT_ROOT/outputs/qbudget_controls/beauty}
MAX_EPOCHS=${MAX_EPOCHS:-100}
RUN_GROUPS=${RUN_GROUPS:-A0,A1,A2,A3,A4,A5,A6}
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
  dataset=beauty
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

require_file() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo "缺少必要文件：$path" >&2
    return 1
  fi
}

require_q_variant() {
  local variant=$1
  require_file "$CONTROL_DIR/$variant/warm_quality.npy"
  require_file "$CONTROL_DIR/$variant/cold_quality.npy"
}

mkdir -p "$CHECKPOINT_DIR"

echo "q-budget diagnostic single-seed run"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SEED=$SEED"
echo "DATA_ROOT=$DATA_ROOT"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "CONTROL_DIR=$CONTROL_DIR"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "RUN_GROUPS=$RUN_GROUPS"
echo "DRY_RUN=$DRY_RUN"
echo "CHECK_ONLY=$CHECK_ONLY"

if [[ "$CHECK_ONLY" == "1" ]]; then
  # 本地轻量 smoke test：只验证 q 文件加载、budget 计算、q-aware 模型构建。
  # 目的：避免在 Mac 本地用全量 Beauty 数据跑完整 epoch，正式训练仍交给服务器。
  PROJECT_ROOT="$PROJECT_ROOT" CONTROL_DIR="$CONTROL_DIR" RUN_GROUPS="$RUN_GROUPS" "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


project_root = Path(os.environ["PROJECT_ROOT"])
control_dir = Path(os.environ["CONTROL_DIR"])
groups = {group.strip() for group in os.environ["RUN_GROUPS"].split(",") if group.strip()}

os.chdir(project_root / "scripts")
sys.path.insert(0, str(project_root / "scripts"))

from run import build_delta_budget, get_model, load_quality_scores  # noqa: E402


variant_by_group = {
    "A4": "current",
    "A5": "shuffle_seed42",
    "A6": "reverse_rank",
}

checked = False
for group, variant in variant_by_group.items():
    if group not in groups:
        continue

    config = OmegaConf.create(
        {
            "use_pretrained_item_embeddings": True,
            "train_delta": True,
            "quality_aware_delta": True,
            "quality_score": {
                "warm_filepath": str(control_dir / variant / "warm_quality.npy"),
                "cold_filepath": str(control_dir / variant / "cold_quality.npy"),
                "min_budget": 0.3,
                "max_budget": 0.6,
                "reverse_budget": False,
            },
            "model": {
                "num_items": 11165,
                "embedding_dim": 128,
                "num_blocks": 2,
                "num_heads": 1,
                "p": 0.3,
                "max_length": 64,
            },
        }
    )

    warm_q, cold_q = load_quality_scores(config)
    budget = build_delta_budget(config, warm_q)
    model = get_model(config, warm_delta_budget=budget)
    checked = True

    print(f"{group}/{variant}:")
    print(f"  warm_q_shape={tuple(warm_q.shape)} cold_q_shape={tuple(cold_q.shape)}")
    print(
        "  budget_min={:.6f} budget_mean={:.6f} budget_max={:.6f}".format(
            float(budget.min()),
            float(budget.mean()),
            float(budget.max()),
        )
    )
    print(f"  model={model.__class__.__name__}")

if not checked:
    print("CHECK_ONLY: RUN_GROUPS 中没有 A4/A5/A6，跳过 q-aware 加载检查。")
PY
  exit 0
fi

if group_enabled A0; then
  # A0：普通 SASRec baseline。
  # 目的：确认没有 content initialization 和 delta 时的基础参照。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}"
fi

if group_enabled A1; then
  # A1：content initialization。
  # 目的：确认作者给好的 content embeddings 进入 SASRec 后的参照。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True
fi

if group_enabled A2; then
  # A2：Let It Go 原版全局 cap。
  # 目的：使用论文默认 max_delta_norm=0.5，作为官方 trainable delta 参照。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    max_delta_norm=0.5
fi

if group_enabled A3; then
  # A3：全局 cap 容量匹配对照。
  # 这里把 max_delta_norm 设为 q-aware budget 的均值 0.363，
  # 目的是区分“平均修正容量变小”与“item-specific 分配是否有用”。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    max_delta_norm=0.363
fi

if group_enabled A4; then
  require_q_variant current
  # A4：q-aware current。
  # 目的：使用原始 q_metadata_all 分配 warm item 的 delta budget。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    quality_aware_delta=True \
    "$(hydra_quoted_arg quality_score.warm_filepath "$CONTROL_DIR/current/warm_quality.npy")" \
    "$(hydra_quoted_arg quality_score.cold_filepath "$CONTROL_DIR/current/cold_quality.npy")" \
    quality_score.min_budget=0.3 \
    quality_score.max_budget=0.6 \
    quality_score.reverse_budget=False
fi

if group_enabled A5; then
  require_q_variant shuffle_seed42
  # A5：q-aware shuffle-budget。
  # 目的：保持 q/budget 分布不变，只随机破坏 q 和 warm item 的对应关系。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    quality_aware_delta=True \
    "$(hydra_quoted_arg quality_score.warm_filepath "$CONTROL_DIR/shuffle_seed42/warm_quality.npy")" \
    "$(hydra_quoted_arg quality_score.cold_filepath "$CONTROL_DIR/shuffle_seed42/cold_quality.npy")" \
    quality_score.min_budget=0.3 \
    quality_score.max_budget=0.6 \
    quality_score.reverse_budget=False
fi

if group_enabled A6; then
  require_q_variant reverse_rank
  # A6：q-aware reverse-rank-budget。
  # 目的：保持 q/budget 分布不变，只反转 q 和 warm item 的 rank 对应关系。
  # 注意：这里必须保持 quality_score.reverse_budget=False，因为 reverse 已经体现在 q 文件里。
  run_cmd "$PYTHON_BIN" run.py \
    seed="$SEED" \
    "${COMMON_ARGS[@]}" \
    use_pretrained_item_embeddings=True \
    train_delta=True \
    quality_aware_delta=True \
    "$(hydra_quoted_arg quality_score.warm_filepath "$CONTROL_DIR/reverse_rank/warm_quality.npy")" \
    "$(hydra_quoted_arg quality_score.cold_filepath "$CONTROL_DIR/reverse_rank/cold_quality.npy")" \
    quality_score.min_budget=0.3 \
    quality_score.max_budget=0.6 \
    quality_score.reverse_budget=False
fi
