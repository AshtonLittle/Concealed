"""Single-Command CLI to Train on Snowflake GPU Compute Pools from Your Local Terminal.

Supports BOTH standard Snowflake accounts and **Snowflake Trial Accounts** (where outbound
internet access / External Access Integrations are blocked inside containers).

On Trial Accounts, this CLI automatically:
  1. Provisions `CONCEALED_DB`, `@IMAGE_STAGE`, `@MODEL_STAGE`, and `CONCEALED_GPU_POOL`
  2. Packs your 5,000 unique local images into `images_bundle.tar` and uploads to `@IMAGE_STAGE`
  3. Caches the pretrained ViT surrogate weights locally, packs them into `hf_models_bundle.tar`,
     and uploads them to `@MODEL_STAGE` so the Snowflake GPU container can load them 100% offline
  4. Runs GPU training on `CONCEALED_GPU_POOL`, downloads `best_generator.pt`, and exports
     `generator.onnx` locally!
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile
import tempfile
from tqdm import tqdm
import yaml

from concealed.data.dataset import discover_images
from concealed.models.surrogates import VisionTransformerSurrogate
from concealed.pipeline.export import export_to_onnx
from concealed.train import load_generator_checkpoint


CORE_SQL_STATEMENTS = [
    "CREATE DATABASE IF NOT EXISTS CONCEALED_DB",
    "USE DATABASE CONCEALED_DB",
    "CREATE SCHEMA IF NOT EXISTS PUBLIC",
    "CREATE STAGE IF NOT EXISTS IMAGE_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE)",
    "CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE)",
    """CREATE COMPUTE POOL IF NOT EXISTS CONCEALED_GPU_POOL
       MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = {instance_family}
       AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 600""",
]

EGRESS_SQL_STATEMENTS = [
    """CREATE OR REPLACE NETWORK RULE HF_PYPI_NETWORK_RULE
       MODE = EGRESS TYPE = HOST_PORT
       VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80')""",
    """CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CONCEALED_HF_ACCESS
       ALLOWED_NETWORK_RULES = (HF_PYPI_NETWORK_RULE) ENABLED = TRUE""",
]


def _stage_has_file(session, stage_file_path: str) -> bool:
    try:
        rows = session.sql(f"LIST {stage_file_path}").collect()
        return len(rows) > 0
    except Exception:
        return False


def _create_and_upload_image_bundle(session, images: list[Path], force_upload: bool = False) -> None:
    stage_target = "@CONCEALED_DB.PUBLIC.IMAGE_STAGE/images_bundle.tar"
    if not force_upload and _stage_has_file(session, stage_target):
        print(
            f"[2/4] Found existing {stage_target} on Snowflake Stage "
            "(skipping image upload; pass --force-upload to overwrite)."
        )
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "images_bundle.tar"
        print(f"[2/4] Packing {len(images)} unique images into {tar_path.name} for fast single-file upload...")
        with tarfile.open(tar_path, "w") as tar:
            for img_path in tqdm(images, desc="Packing images", unit="img"):
                tar.add(str(img_path), arcname=img_path.name)

        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {tar_path.name} ({size_mb:.1f} MB) to @CONCEALED_DB.PUBLIC.IMAGE_STAGE ...")
        session.file.put(
            tar_path.resolve().as_posix(),
            "@CONCEALED_DB.PUBLIC.IMAGE_STAGE",
            auto_compress=False,
            overwrite=True,
            parallel=16,
        )


def _stage_surrogate_weights_for_trial_account(
    session,
    config: dict,
    profile_key: str,
    force_upload: bool = False,
) -> str:
    """Download surrogate vision towers locally, pack into a tar archive, and upload to @MODEL_STAGE."""
    bundle_name = f"hf_cache_{profile_key}.tar"
    stage_target = f"@CONCEALED_DB.PUBLIC.MODEL_STAGE/{bundle_name}"
    if not force_upload and _stage_has_file(session, stage_target):
        print(f"[2.5/4] Found cached surrogate weights {stage_target} on Snowflake Stage.")
        return bundle_name

    cache_dir = Path(".snowflake_hf_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_hf_home = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(cache_dir)

    try:
        model_specs = config.get("surrogates", {}).get("train_models", [])
        print(
            f"[2.5/4] Snowflake Trial Account detected (container internet disabled). "
            f"Caching {len(model_specs)} surrogate vision towers locally to stage into Snowflake..."
        )
        for spec in model_specs:
            mname = str(spec["name"])
            print(f"  -> Caching vision tower: {mname}")
            _ = VisionTransformerSurrogate(model_name=mname, weight=1.0, pretrained=True)
    finally:
        if old_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = old_hf_home

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / bundle_name
        print(f"Packing cached surrogate weights into {bundle_name} ...")
        with tarfile.open(tar_path, "w") as tar:
            for item in cache_dir.rglob("*"):
                if item.is_file() and not item.name.endswith(".lock"):
                    rel = item.relative_to(cache_dir).as_posix()
                    tar.add(str(item), arcname=rel)

        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {bundle_name} ({size_mb:.1f} MB) to @CONCEALED_DB.PUBLIC.MODEL_STAGE ...")
        session.file.put(
            tar_path.resolve().as_posix(),
            "@CONCEALED_DB.PUBLIC.MODEL_STAGE",
            auto_compress=False,
            overwrite=True,
            parallel=16,
        )
    return bundle_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command CLI to train Concealed on Snowflake GPU and download the trained ONNX model"
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Local directory of training images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_model",
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
    parser.add_argument(
        "--profile",
        type=str,
        choices=["default", "fast", "heavy", "ocr"],
        default="fast",
        help="Select built-in surrogate profile ('fast' ~1.1GB, 'heavy' ~7GB, 'ocr', or 'default')",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--surrogates", type=str, nargs="+", default=None, help="Override ViT surrogate model names")
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Force re-uploading images_bundle.tar and model cache even if already on Snowflake Stage",
    )
    args = parser.parse_args()

    try:
        from snowflake.snowpark import Session
    except ImportError as exc:
        raise SystemExit(
            "Snowflake CLI training requires `snowflake-snowpark-python` and `snowflake-ml-python`.\n"
            "Install them once via: pip install snowflake-snowpark-python snowflake-ml-python"
        ) from exc

    # 1. Connect to Snowflake via environment variables or ~/.snowflake/connections.toml
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
    for stmt in CORE_SQL_STATEMENTS:
        session.sql(stmt.format(instance_family=args.gpu_family)).collect()

    has_external_access = True
    try:
        for stmt in EGRESS_SQL_STATEMENTS:
            session.sql(stmt).collect()
    except Exception as exc:
        if "trial account" in str(exc).lower() or "509009" in str(exc):
            has_external_access = False
            print("  -> Note: Snowflake Trial Account detected. Switching to Stage-Bridged Offline Mode.")
        else:
            raise

    # 2. Pack unique local images into a single tar archive and upload to @CONCEALED_DB.PUBLIC.IMAGE_STAGE
    images = discover_images(args.data_dir, deduplicate=True)
    if not images:
        raise SystemExit(f"No supported images found in {args.data_dir}")
    _create_and_upload_image_bundle(session, images, force_upload=args.force_upload)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.profile and args.profile != "default":
        prof = config.get("surrogates", {}).get("profiles", {}).get(args.profile)
        if prof:
            config["surrogates"]["train_models"] = prof["train_models"]
            if "cpu_offload" in prof:
                config["surrogates"]["cpu_offload"] = prof["cpu_offload"]
        if args.profile == "ocr":
            config.setdefault("generator", {})["hybrid_global_weight"] = 0.35
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.surrogates is not None:
        config.setdefault("surrogates", {})["train_models"] = [{"name": s, "weight": 1.0} for s in args.surrogates]

    # Disable inside-container ONNX export; we export locally after downloading best_generator.pt
    config.setdefault("training", {})["auto_export_onnx"] = False

    hf_bundle_name: str | None = None
    if not has_external_access:
        profile_tag = args.profile or "custom"
        hf_bundle_name = _stage_surrogate_weights_for_trial_account(
            session, config, profile_key=profile_tag, force_upload=args.force_upload
        )

    # 3. Dispatch remote GPU job via snowflake.ml.jobs
    from snowflake.ml.jobs import remote

    repo_root = Path(__file__).resolve().parent.parent
    import timm

    timm_pkg_dir = Path(timm.__file__).resolve().parent
    job_imports = [
        (str(repo_root / "concealed"), "concealed"),
        (str(timm_pkg_dir), "timm"),
    ]

    remote_kwargs = {
        "stage_name": "MODEL_STAGE",
        "imports": job_imports,
        "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "session": session,
    }
    if has_external_access:
        remote_kwargs["external_access_integrations"] = ["CONCEALED_HF_ACCESS"]
        remote_kwargs["pip_requirements"] = ["timm", "onnx", "onnxruntime"]

    @remote("CONCEALED_GPU_POOL", **remote_kwargs)
    def run_concealed_gpu_training(cfg: dict, cached_hf_bundle: str | None) -> dict:
        import os
        from pathlib import Path
        import tarfile
        from snowflake.snowpark.context import get_active_session
        from concealed.train import train

        sp_session = get_active_session()
        tmp_root = Path("/tmp/concealed_job")
        local_imgs = tmp_root / "images"
        local_out = tmp_root / "output"
        local_imgs.mkdir(parents=True, exist_ok=True)
        local_out.mkdir(parents=True, exist_ok=True)

        if cached_hf_bundle:
            hf_cache_dir = tmp_root / "hf_cache"
            hf_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading staged surrogate weights {cached_hf_bundle} from @MODEL_STAGE ...")
            sp_session.file.get(f"@CONCEALED_DB.PUBLIC.MODEL_STAGE/{cached_hf_bundle}", str(tmp_root))
            matches = list(tmp_root.glob(f"{cached_hf_bundle}*"))
            if matches:
                with tarfile.open(matches[0], "r:*") as tar:
                    tar.extractall(path=hf_cache_dir)
                matches[0].unlink()
            os.environ["HF_HOME"] = str(hf_cache_dir)
            os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
            os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "hub")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print("Downloading images_bundle.tar from @CONCEALED_DB.PUBLIC.IMAGE_STAGE ...")
        sp_session.file.get("@CONCEALED_DB.PUBLIC.IMAGE_STAGE/images_bundle.tar", str(tmp_root))
        img_matches = list(tmp_root.glob("images_bundle.tar*"))
        if img_matches:
            with tarfile.open(img_matches[0], "r:*") as tar:
                tar.extractall(path=local_imgs)
            img_matches[0].unlink()
        else:
            sp_session.file.get("@CONCEALED_DB.PUBLIC.IMAGE_STAGE", str(local_imgs))

        _, metrics = train(config=cfg, data_dir=local_imgs, output_dir=local_out, device_str="cuda")

        for fpath in local_out.glob("*.pt"):
            sp_session.file.put(
                str(fpath), "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest", auto_compress=False, overwrite=True
            )
        for fpath in (local_out / "samples").glob("*.png"):
            sp_session.file.put(
                str(fpath), "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/samples", auto_compress=False, overwrite=True
            )
        return metrics

    print(f"[3/4] Submitting GPU training job to CONCEALED_GPU_POOL ({args.gpu_family})...")
    job = run_concealed_gpu_training(config, hf_bundle_name)
    print(f"Job dispatched (ID: {job.id}). Waiting for GPU container startup and training completion...")
    try:
        job.wait()
    except Exception:
        print("\n--- Remote Job Logs ---")
        job.show_logs()
        raise
    job.show_logs()
    metrics = job.result()
    print(f"Remote GPU training finished! Validation metrics: {metrics}")

    # 4. Download trained checkpoints back to local output_dir and export ONNX locally
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Downloading trained checkpoint to {out_dir} and exporting generator.onnx ...")
    session.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/", str(out_dir))

    best_pt = out_dir / "best_generator.pt"
    if best_pt.exists():
        gen, _ = load_generator_checkpoint(best_pt, device="cpu")
        onnx_info = export_to_onnx(gen, out_dir / "generator.onnx", verify=True)
        print(f"Exported verified real-time ONNX model -> {onnx_info['onnx_path']} ({onnx_info['size_mb']} MB)")


if __name__ == "__main__":
    main()
