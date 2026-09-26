"""Single-Command CLI to Train on Snowflake GPU Compute Pools from Your Local Terminal.

Eliminates the need for Snowflake Notebooks or manual SQL Worksheets.
From your local terminal, this CLI:
  1. Connects to your Snowflake account
  2. Auto-provisions the Database, Stages, HuggingFace Network Rule, and GPU Compute Pool
  3. Uploads your local image folder to Snowflake Stage
  4. Launches the PyTorch GPU training + ONNX export job on Snowflake's NVIDIA GPU
  5. Downloads the trained `best_generator.pt` and `generator.onnx` to your local machine
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml

from concealed.data.dataset import discover_images


SETUP_SQL_STATEMENTS = [
    "CREATE DATABASE IF NOT EXISTS CONCEALED_DB",
    "USE DATABASE CONCEALED_DB",
    "CREATE SCHEMA IF NOT EXISTS PUBLIC",
    "CREATE STAGE IF NOT EXISTS IMAGE_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE)",
    "CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE)",
    """CREATE COMPUTE POOL IF NOT EXISTS CONCEALED_GPU_POOL
       MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = {instance_family}
       AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 600""",
    """CREATE OR REPLACE NETWORK RULE HF_PYPI_NETWORK_RULE
       MODE = EGRESS TYPE = HOST_PORT
       VALUE_LIST = (
         'huggingface.co:443', 'cdn-lfs.huggingface.co:443',
         'cdn-lfs-us-1.huggingface.co:443', 'cas-bridge.xethub.hf.co:443',
         'pypi.org:443', 'files.pythonhosted.org:443'
       )""",
    """CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CONCEALED_HF_ACCESS
       ALLOWED_NETWORK_RULES = (HF_PYPI_NETWORK_RULE) ENABLED = TRUE""",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command CLI to train Concealed on Snowflake GPU and download the trained ONNX model"
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Local directory of training images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/snowflake_exp",
        help="Local directory where trained best_generator.pt and generator.onnx will be downloaded",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--connection-name",
        type=str,
        default=os.environ.get("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "default"),
        help="Snowflake CLI connection name in ~/.snowflake/connections.toml (or set SNOWFLAKE_ACCOUNT/USER/PASSWORD)",
    )
    parser.add_argument(
        "--gpu-family",
        type=str,
        default="GPU_NV_S",
        choices=["GPU_NV_S", "GPU_NV_M", "GPU_NV_L"],
        help="Snowflake GPU instance family (default: GPU_NV_S = 1x NVIDIA A10G 24GB VRAM)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--surrogates", type=str, nargs="+", default=None, help="Override ViT surrogate model names")
    args = parser.parse_args()

    try:
        from snowflake.snowpark import Session
    except ImportError as exc:
        raise SystemExit(
            "Snowflake CLI training requires `snowflake-snowpark-python` and `snowflake-ml-python`.\n"
            "Install them once via: pip install snowflake-snowpark-python snowflake-ml-python"
        ) from exc

    # 1. Connect to Snowflake via ~/.snowflake/connections.toml or environment variables
    if os.environ.get("SNOWFLAKE_ACCOUNT") and os.environ.get("SNOWFLAKE_USER"):
        conn_params = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ.get("SNOWFLAKE_PASSWORD", ""),
            "role": os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        }
        session = Session.builder.configs(conn_params).create()
    else:
        session = Session.builder.config("connection_name", args.connection_name).create()

    print("[1/4] Provisioning Snowflake Database, Stages, and GPU Compute Pool...")
    for stmt in SETUP_SQL_STATEMENTS:
        session.sql(stmt.format(instance_family=args.gpu_family)).collect()

    # 2. Upload local dataset images to @CONCEALED_DB.PUBLIC.IMAGE_STAGE
    images = discover_images(args.data_dir)
    print(f"[2/4] Uploading {len(images)} images from {args.data_dir} to @CONCEALED_DB.PUBLIC.IMAGE_STAGE ...")
    session.file.put(
        f"{Path(args.data_dir).resolve().as_posix()}/*",
        "@CONCEALED_DB.PUBLIC.IMAGE_STAGE",
        auto_compress=False,
        overwrite=True,
        parallel=16,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.surrogates is not None:
        config.setdefault("surrogates", {})["train_models"] = [{"name": s, "weight": 1.0} for s in args.surrogates]

    # 3. Dispatch remote GPU job via snowflake.ml.jobs
    from snowflake.ml.jobs import remote

    repo_root = Path(__file__).resolve().parent.parent

    @remote(
        "CONCEALED_GPU_POOL",
        stage_name="MODEL_STAGE",
        external_access_integrations=["CONCEALED_HF_ACCESS"],
        pip_requirements=["torch", "torchvision", "transformers", "timm", "pyyaml", "pillow", "tqdm", "onnx", "onnxruntime", "opencv-python-headless"],
        imports=[str(repo_root / "concealed")],
        session=session,
    )
    def _run_remote_gpu_training(cfg: dict) -> dict:
        from pathlib import Path
        from snowflake.snowpark.context import get_active_session
        from concealed.train import train

        sp_session = get_active_session()
        local_imgs = Path("/tmp/images")
        local_out = Path("/tmp/output")
        local_imgs.mkdir(parents=True, exist_ok=True)
        sp_session.file.get("@CONCEALED_DB.PUBLIC.IMAGE_STAGE", str(local_imgs))

        _, metrics = train(config=cfg, data_dir=local_imgs, output_dir=local_out, device_str="cuda")

        for fpath in local_out.glob("*.pt"):
            sp_session.file.put(str(fpath), "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest", auto_compress=False, overwrite=True)
        for fpath in local_out.glob("*.onnx"):
            sp_session.file.put(str(fpath), "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest", auto_compress=False, overwrite=True)
        return metrics

    print(f"[3/4] Running GPU training on CONCEALED_GPU_POOL ({args.gpu_family})...")
    job = _run_remote_gpu_training(config)
    metrics = job.result()
    print(f"Remote GPU training finished! Validation metrics: {metrics}")

    # 4. Download trained checkpoints and ONNX model back to local output_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Downloading trained models to {out_dir} ...")
    session.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/", str(out_dir))
    print(f"Done! Your real-time model is ready at: {out_dir / 'generator.onnx'}")


if __name__ == "__main__":
    main()
