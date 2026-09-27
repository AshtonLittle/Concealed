"""Snowflake Container Runtime (GPU) & Snowpark Container Services (SPCS) Training Entrypoint.

Can be executed directly inside:
  1. A Snowflake Notebook running on Container Runtime for ML (GPU Compute Pool: GPU_NV_S / GPU_NV_M)
  2. A Snowpark Container Services (SPCS) Job with Stage Volume Mounts

Usage inside a Snowflake GPU Notebook or SPCS Job:
    python scripts/snowflake_train.py \
        --image-stage "@CONCEALED_DB.PUBLIC.IMAGE_STAGE" \
        --model-stage "@CONCEALED_DB.PUBLIC.MODEL_STAGE/exp1" \
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from concealed.pipeline.export import export_to_onnx
from concealed.train import train


def sync_from_snowflake_stage(stage_path: str, local_dir: Path) -> None:
    """Download training images from an internal Snowflake Stage using the active Snowpark session."""
    from snowflake.snowpark.context import get_active_session

    session = get_active_session()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from Snowflake stage {stage_path} -> {local_dir} ...")
    session.file.get(stage_path, str(local_dir))


def upload_to_snowflake_stage(local_dir: Path, stage_path: str) -> None:
    """Upload trained checkpoints, ONNX models, and logs back to a Snowflake Stage."""
    from snowflake.snowpark.context import get_active_session

    session = get_active_session()
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            rel_parent = file_path.parent.relative_to(local_dir).as_posix()
            target_stage = stage_path.rstrip("/")
            if rel_parent != ".":
                target_stage = f"{target_stage}/{rel_parent}"
            print(f"Uploading {file_path.name} -> {target_stage} ...")
            session.file.put(
                str(file_path),
                target_stage,
                auto_compress=False,
                overwrite=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Concealed Generator on Snowflake GPU Compute Pool")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--image-stage",
        type=str,
        default=None,
        help="Snowflake stage path for input images (e.g., @CONCEALED_DB.PUBLIC.IMAGE_STAGE)",
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default=None,
        help="Snowflake stage path to upload trained models (e.g., @CONCEALED_DB.PUBLIC.MODEL_STAGE/exp1)",
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default="/tmp/concealed_data",
        help="Local path (or mounted SPCS stage volume path like /mnt/images)",
    )
    parser.add_argument(
        "--local-output-dir",
        type=str,
        default="/tmp/concealed_runs/exp1",
        help="Local output directory (or mounted SPCS stage volume path like /mnt/models/exp1)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    data_dir = Path(args.local_data_dir)
    out_dir = Path(args.local_output_dir)

    if args.image_stage:
        sync_from_snowflake_stage(args.image_stage, data_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    # 1. Train generator on Snowflake CUDA GPU
    trained_gen, val_metrics = train(
        config=config,
        data_dir=data_dir,
        output_dir=out_dir,
        device_str="cuda",
    )
    print(f"Training complete. Final validation metrics: {val_metrics}")

    # 2. Export trained generator to dynamic-axis ONNX and INT8 ONNX for real-time pipelines
    onnx_path = out_dir / "generator.onnx"
    export_info = export_to_onnx(
        generator=trained_gen,
        output_path=onnx_path,
        verify=True,
        quantize_int8=True,
    )
    print(f"ONNX export complete: {export_info}")

    # 3. Upload artifacts back to Snowflake Stage if specified
    if args.model_stage:
        upload_to_snowflake_stage(out_dir, args.model_stage)


if __name__ == "__main__":
    main()
