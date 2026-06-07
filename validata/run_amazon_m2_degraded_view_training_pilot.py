"""生成或执行 Amazon-M2 degraded-view 训练 pilot 的服务器 wrapper。

真正的服务器主入口是 scripts/run_amazon_m2_degraded_view_training_pilot_2seeds.sh。
本脚本只生成一个带路径/环境变量的 wrapper，方便从任意目录启动，并保留
server_training_manifest.json。训练命令本身不在这里拼接，避免和 scripts/run.py
的工作目录要求脱节。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent
LOCAL_DATA_ROOT = RESEARCH_ROOT / "letitgo-data" / "data" / "amazon_m2_fr"
SERVER_DATA_ROOT = Path("/root/letitgo-data/data/amazon_m2_fr")
DEFAULT_DATA_ROOT = SERVER_DATA_ROOT if SERVER_DATA_ROOT.exists() else LOCAL_DATA_ROOT
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "row_data" / "amazon_m2_raw" / "products_train.csv"
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "temp_202606_实验文件记录"
    / f"temp_{datetime.now().strftime('%Y%m%d')}"
    / "degraded-view-training-pilot-preflight"
)
DATE_TAG = datetime.now().strftime("%Y%m%d")
DEFAULT_RUN_OUTPUT_DIR = Path(f"/hy-tmp/letitgo_outputs/amazon_m2_degraded_view_training_pilot_{DATE_TAG}")
DEFAULT_EMBEDDING_OUTPUT_DIR = Path(f"/hy-tmp/letitgo_outputs/amazon_m2_degraded_view_training_embeddings_{DATE_TAG}")
DEFAULT_CHECKPOINT_DIR = Path(f"/hy-tmp/letitgo_ckpt/amazon_m2_degraded_view_training_pilot_{DATE_TAG}")
DEFAULT_VARIANTS = "control_full,title_trunc_8,random_title_dropout_p30,no_title"
DEFAULT_SEEDS = "42,43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write or execute degraded-view training pilot commands.")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_RUN_OUTPUT_DIR)
    parser.add_argument("--embedding-output-dir", type=Path, default=DEFAULT_EMBEDDING_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=Path("/hy-tmp/letitgo_logs"))
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--python-bin", default="/opt/anaconda3/envs/let-it-go-py3.11/bin/python")
    parser.add_argument("--project-name", default="amazon_m2_degraded_view_training_pilot")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--devices", default="[0]")
    parser.add_argument("--learning-rate", default="1e-3")
    parser.add_argument("--max-delta-norm", default="0.5")
    parser.add_argument("--sentence-checkpoint", default="intfloat/multilingual-e5-base")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行生成的 shell 脚本；默认只写脚本和 manifest。",
    )
    return parser.parse_args()


def parse_csv_list(value: str, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} 不能为空。")
    return items


def parse_int_csv(value: str, name: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} 必须是逗号分隔整数：{value}") from exc


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_shell_script(args: argparse.Namespace, variants: list[str], seeds: list[int]) -> str:
    runner = Path(args.repo_root) / "scripts" / "run_amazon_m2_degraded_view_training_pilot_2seeds.sh"
    env_rows = [
        ("PYTHON_BIN", args.python_bin),
        ("DATA_ROOT", args.data_root),
        ("PRODUCTS_PATH", args.products_path),
        ("LOG_DIR", args.log_dir),
        ("RUN_OUTPUT_DIR", args.run_output_dir),
        ("VARIANT_EMBEDDING_ROOT", args.embedding_output_dir),
        ("CHECKPOINT_DIR", args.checkpoint_dir),
        ("PROJECT_NAME", args.project_name),
        ("MAX_EPOCHS", args.max_epochs),
        ("TRAINER_DEVICES", args.devices),
        ("LEARNING_RATE", args.learning_rate),
        ("MAX_DELTA_NORM", args.max_delta_norm),
        ("SENTENCE_CHECKPOINT", args.sentence_checkpoint),
        ("ENCODE_BATCH_SIZE", args.encode_batch_size),
        ("VARIANTS", ",".join(variants)),
        ("SEEDS", ",".join(str(seed) for seed in seeds)),
        ("CHECK_ONLY", int(args.check_only)),
        ("DRY_RUN", int(args.dry_run)),
        ("SKIP_EMBEDDING", int(args.skip_embedding)),
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated wrapper. The audited training logic lives in scripts/run_amazon_m2_degraded_view_training_pilot_2seeds.sh.",
    ]
    for key, value in env_rows:
        lines.append(f"export {key}={quote(value)}")
    lines.extend(["", f"bash {quote(runner)}", ""])
    return "\n".join(lines)


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "server_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants, "variants")
    seeds = parse_int_csv(args.seeds, "seeds")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shell_script = build_shell_script(args, variants, seeds)
    script_path = output_dir / "server_degraded_view_training_pilot.sh"
    script_path.write_text(shell_script, encoding="utf-8")
    script_path.chmod(0o755)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script_role": "write/execute server training command matrix for degraded-view pilot",
        "repo_root": str(args.repo_root),
        "data_root": str(args.data_root),
        "products_path": str(args.products_path),
        "output_dir": str(output_dir),
        "run_output_dir": str(args.run_output_dir),
        "log_dir": str(args.log_dir),
        "embedding_output_dir": str(args.embedding_output_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "variants": variants,
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "devices": args.devices,
        "python_bin": args.python_bin,
        "check_only": args.check_only,
        "dry_run": args.dry_run,
        "skip_embedding": args.skip_embedding,
        "execute": args.execute,
        "server_script": str(script_path),
        "audited_runner": str(args.repo_root / "scripts" / "run_amazon_m2_degraded_view_training_pilot_2seeds.sh"),
    }
    write_manifest(output_dir, manifest)

    print(f"wrote server script: {script_path}")
    print(f"wrote manifest: {output_dir / 'server_training_manifest.json'}")
    if args.execute:
        subprocess.run(["bash", str(script_path)], check=True)


if __name__ == "__main__":
    main()
