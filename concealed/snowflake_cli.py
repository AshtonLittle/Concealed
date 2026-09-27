"""Single-Command & Multi-Account Fleet CLI to Train on Snowflake GPU Compute Pools.

Supports BOTH single-account execution and **Multi-Account Fleet Training** (e.g., 4 separate
Snowflake accounts, each running 1x NVIDIA A10G GPU in parallel) configured via a single `.env` file.

Multi-Account Fleet Features:
  1. Reads `SNOWFLAKE_ACCOUNT_1..N`, `SNOWFLAKE_USER_1..N`, `SNOWFLAKE_PASSWORD_1..N` from `.env`
     (backward-compatible with unnumbered `SNOWFLAKE_ACCOUNT`).
  2. Partitions the 5,000-image dataset into disjoint training shards with a globally shared
     validation gallery so every account's analytics report is 100% comparable.
  3. Warm-starts all accounts from `trained_model/best_generator.pt` (unless `--from-scratch`)
     and assigns specialized ViT surrogate emphasis across workers (`SigLIP`, `CLIP`, `DINOv2`, `Anchor`).
  4. Streams a unified real-time multi-account analytics dashboard and merges all worker checkpoints
     via Validation-Weighted Model Soup & Task-Vector Arithmetic into `trained_model/best_generator.pt`
     and `trained_model/generator.onnx`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tarfile
import tempfile
import threading
import time
from typing import Dict, List, Optional

from tqdm import tqdm
import yaml

from concealed.data.dataset import discover_images, split_shared_val_and_train_shard
from concealed.models.surrogates import VisionTransformerSurrogate
from concealed.pipeline.export import export_to_onnx
from concealed.train import load_generator_checkpoint, merge_generator_checkpoints


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

SPECIALIST_ROLES = [
    ("SigLIP/Gemini Specialist", "siglip"),
    ("CLIP/OpenAI Specialist", "clip"),
    ("DINOv2/Dense Specialist", "dinov2"),
    ("Balanced Multi-Scale Anchor", "anchor"),
]

_PRINT_LOCK = threading.Lock()


@dataclass
class SnowflakeAccountSpec:
    """Connection credentials and slot metadata for a single Snowflake account."""

    slot: int
    account: str
    user: str
    password: str
    role: str = "ACCOUNTADMIN"
    warehouse: str = "COMPUTE_WH"

    @property
    def label(self) -> str:
        short_acct = self.account.split(".")[0][:18]
        return f"Acct-{self.slot}({short_acct})"


def load_env_file(env_path: str | Path = ".env") -> None:
    """Load key=value pairs from a single `.env` file into `os.environ`."""
    p = Path(env_path)
    if not p.exists():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            key = k.strip().replace("SNOWFLAE_", "SNOWFLAKE_")
            val = v.strip().strip('"').strip("'")
            os.environ[key] = val


def _is_placeholder(val: str) -> bool:
    v = val.strip()
    if not v:
        return True
    upper = v.upper()
    return any(tok in upper for tok in ("YOUR_", "<", "REPLACE_ME", "EXAMPLE", "TODO", "XXXX"))


def discover_snowflake_accounts(
    env_path: str | Path = ".env",
    requested_slots: Optional[List[int]] = None,
) -> List[SnowflakeAccountSpec]:
    """Discover all configured Snowflake accounts from environment variables / `.env`.

    Supports both:
      - Unnumbered `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`
      - Numbered `SNOWFLAKE_ACCOUNT_1..16`, `SNOWFLAKE_USER_1..16`, `SNOWFLAKE_PASSWORD_1..16`
    """
    load_env_file(env_path)

    default_role = os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
    default_wh = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

    accounts: List[SnowflakeAccountSpec] = []
    seen_keys: set[tuple[str, str]] = set()

    def _try_add(slot_num: int, acct: str, user: str, pwd: str, role: str, wh: str) -> None:
        if _is_placeholder(acct) or _is_placeholder(user) or _is_placeholder(pwd):
            return
        dedup_key = (acct.strip().lower(), user.strip().lower())
        if dedup_key in seen_keys:
            return
        seen_keys.add(dedup_key)
        accounts.append(
            SnowflakeAccountSpec(
                slot=slot_num,
                account=acct.strip(),
                user=user.strip(),
                password=pwd.strip(),
                role=(role or default_role).strip(),
                warehouse=(wh or default_wh).strip(),
            )
        )

    # 1. Check numbered slots 1..16
    for slot in range(1, 17):
        acct = os.environ.get(f"SNOWFLAKE_ACCOUNT_{slot}", "")
        user = os.environ.get(f"SNOWFLAKE_USER_{slot}", "") or os.environ.get(f"SNOWFLAE_USER_{slot}", "")
        pwd = os.environ.get(f"SNOWFLAKE_PASSWORD_{slot}", "")
        role = os.environ.get(f"SNOWFLAKE_ROLE_{slot}", default_role)
        wh = os.environ.get(f"SNOWFLAKE_WAREHOUSE_{slot}", default_wh)
        _try_add(slot, acct, user, pwd, role, wh)

    # 2. Also check unnumbered SNOWFLAKE_ACCOUNT/USER/PASSWORD (assigning slot 1 if free, else next slot)
    un_acct = os.environ.get("SNOWFLAKE_ACCOUNT", "")
    un_user = os.environ.get("SNOWFLAKE_USER", "") or os.environ.get("SNOWFLAE_USER", "")
    un_pwd = os.environ.get("SNOWFLAKE_PASSWORD", "")
    used_slots = {a.slot for a in accounts}
    un_slot = 1 if 1 not in used_slots else (max(used_slots) + 1 if used_slots else 1)
    _try_add(un_slot, un_acct, un_user, un_pwd, default_role, default_wh)

    accounts.sort(key=lambda a: a.slot)

    if requested_slots:
        slot_set = set(requested_slots)
        accounts = [a for a in accounts if a.slot in slot_set]

    return accounts


def build_worker_config(
    base_config: dict,
    worker_idx: int,
    num_workers: int,
    fleet_strategy: str = "hybrid_specialist",
    pre_sharded_bundle: bool = False,
) -> tuple[dict, str]:
    """Create a worker-specific training configuration for a multi-account fleet node."""
    cfg = copy.deepcopy(base_config)
    train_cfg = cfg.setdefault("training", {})

    if pre_sharded_bundle or num_workers <= 1:
        # Image tar bundle for this account was already pre-sliced to (shared_val + shard_i)
        train_cfg["num_shards"] = 1
        train_cfg["shard_id"] = 0
    else:
        # Full 5,000-image tar bundle exists on stage; slice shard_i inside the container
        train_cfg["num_shards"] = num_workers
        train_cfg["shard_id"] = worker_idx

    # Offset augmentation seed per worker while keeping dataset split seed=42 identical
    base_seed = int(train_cfg.get("seed", 42))
    train_cfg["seed"] = base_seed

    if num_workers <= 1 or fleet_strategy == "sharded":
        role_desc = f"Data Shard {worker_idx + 1}/{num_workers}"
        return cfg, role_desc

    models = cfg.get("surrogates", {}).get("train_models", [])
    if len(models) == 1:
        only_name = str(models[0].get("name", "ViT")).split("/")[-1]
        base_lr = float(train_cfg.get("lr", 6.0e-4))
        lr_mults = [1.0, 1.08, 0.94, 1.04]
        train_cfg["lr"] = round(base_lr * lr_mults[worker_idx % len(lr_mults)], 7)
        role_desc = f"{only_name} Specialist (Shard {worker_idx + 1}/{num_workers})"
        return cfg, role_desc

    role_title, focus_token = SPECIALIST_ROLES[worker_idx % len(SPECIALIST_ROLES)]
    for m in models:
        mname = str(m.get("name", "")).lower()
        base_w = float(m.get("weight", 1.0))
        if focus_token == "anchor":
            m["weight"] = round(base_w * 1.25, 3)
        elif focus_token in mname:
            m["weight"] = round(base_w * 2.25, 3)
        else:
            m["weight"] = round(base_w * 1.0, 3)

    # Slight learning rate & salience diversity across workers for richer Model Soup basin
    base_lr = float(train_cfg.get("lr", 6.0e-4))
    lr_mults = [1.0, 1.08, 0.94, 1.04]
    train_cfg["lr"] = round(base_lr * lr_mults[worker_idx % len(lr_mults)], 7)

    if focus_token == "anchor":
        loss_cfg = cfg.setdefault("loss", {})
        loss_cfg["patch_cosine_weight"] = round(float(loss_cfg.get("patch_cosine_weight", 3.5)) * 1.15, 3)

    role_desc = f"{role_title} (Shard {worker_idx + 1}/{num_workers})"
    return cfg, role_desc


def _stage_has_file(session, stage_file_path: str) -> bool:
    try:
        rows = session.sql(f"LIST {stage_file_path}").collect()
        return len(rows) > 0
    except Exception:
        return False


def _prepare_local_surrogate_tar(config: dict, profile_key: str) -> Path:
    """Cache surrogate vision towers once locally and pack ONLY the active models into a `.tar` archive."""
    cache_dir = Path(".snowflake_hf_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_specs = config.get("surrogates", {}).get("train_models", [])
    model_names = [str(spec["name"]) for spec in model_specs]

    # Build a unique bundle filename based on the active surrogate model set
    short_slug = "_".join(sorted(m.split("/")[-1][:14] for m in model_names)) or profile_key
    bundle_path = cache_dir / f"hf_cache_{short_slug}.tar"

    if bundle_path.exists() and bundle_path.stat().st_size > 1024 * 1024:
        return bundle_path

    old_hf_home = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(cache_dir)
    try:
        print(f"[Prep] Ensuring {len(model_names)} surrogate vision tower(s) are cached in {cache_dir.name} ...")
        for mname in model_names:
            print(f"  -> Verifying vision tower: {mname}")
            _ = VisionTransformerSurrogate(model_name=mname, weight=1.0, pretrained=True)
    finally:
        if old_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = old_hf_home

    # Only include HF hub directories matching the active surrogates so single-model runs stay compact
    allowed_dir_tokens = [m.replace("/", "--").lower() for m in model_names] + [
        m.split("/")[-1].lower() for m in model_names
    ]

    print(f"[Prep] Packing surrogate weights archive {bundle_path.name} ...")
    with tarfile.open(bundle_path, "w") as tar:
        for item in cache_dir.rglob("*"):
            if item.is_file() and not item.name.endswith(".lock") and not item.name.endswith(".tar"):
                rel = item.relative_to(cache_dir).as_posix()
                rel_lower = rel.lower()
                if "models--" in rel_lower and not any(tok in rel_lower for tok in allowed_dir_tokens):
                    continue
                tar.add(str(item), arcname=rel)
    return bundle_path


def _upload_image_bundle_for_worker(
    session,
    acct_label: str,
    all_images: list[Path],
    worker_idx: int,
    num_workers: int,
    max_images: Optional[int],
    force_upload: bool = False,
) -> bool:
    """Ensure `@CONCEALED_DB.PUBLIC.IMAGE_STAGE/images_bundle.tar` exists on the account.

    Returns ``True`` if a pre-sharded bundle was uploaded, or ``False`` if an existing full
    `images_bundle.tar` is being reused with in-container sharding.
    """
    stage_target = "@CONCEALED_DB.PUBLIC.IMAGE_STAGE/images_bundle.tar"
    if not force_upload and _stage_has_file(session, stage_target):
        with _PRINT_LOCK:
            print(f"[{acct_label}] Found existing {stage_target} on stage (reusing with in-container sharding).")
        return False

    # Build a compact shard-specific bundle (shared val set + worker's disjoint training shard)
    if num_workers > 1:
        train_files, val_files = split_shared_val_and_train_shard(
            all_files=all_images,
            val_split=0.1,
            seed=42,
            max_images=max_images,
            num_shards=num_workers,
            shard_id=worker_idx,
        )
        images_to_pack = val_files + train_files
        pre_sharded = True
    else:
        images_to_pack = all_images
        pre_sharded = False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "images_bundle.tar"
        with _PRINT_LOCK:
            print(
                f"[{acct_label}] Packing {len(images_to_pack)} images "
                f"({'shard ' + str(worker_idx + 1) + '/' + str(num_workers) if pre_sharded else 'full pool'}) -> {tar_path.name} ..."
            )
        with tarfile.open(tar_path, "w") as tar:
            for img_path in images_to_pack:
                tar.add(str(img_path), arcname=img_path.name)

        size_mb = tar_path.stat().st_size / (1024 * 1024)
        with _PRINT_LOCK:
            print(f"[{acct_label}] Uploading {tar_path.name} ({size_mb:.1f} MB) to @CONCEALED_DB.PUBLIC.IMAGE_STAGE ...")
        session.file.put(
            tar_path.resolve().as_posix(),
            "@CONCEALED_DB.PUBLIC.IMAGE_STAGE",
            auto_compress=False,
            overwrite=True,
            parallel=16,
        )
    return pre_sharded


def _create_session_for_account(acct: SnowflakeAccountSpec):
    from snowflake.snowpark import Session

    conn_params = {
        "account": acct.account,
        "user": acct.user,
        "password": acct.password,
        "role": acct.role,
        "warehouse": acct.warehouse,
    }
    return Session.builder.configs(conn_params).create()


def _cancel_account_jobs(session, acct_label: str, specific_id: str | None = None) -> None:
    from snowflake.ml.jobs import get_job, list_jobs

    try:
        session.sql("USE DATABASE CONCEALED_DB").collect()
        session.sql("USE SCHEMA PUBLIC").collect()
    except Exception:
        return

    if specific_id and specific_id != "ALL":
        try:
            j = get_job(specific_id, session=session)
            with _PRINT_LOCK:
                print(f"[{acct_label}] Canceling job {j.id} (status: {j.status})...")
            j.cancel()
        except Exception as exc:
            with _PRINT_LOCK:
                print(f"[{acct_label}] Could not cancel {specific_id}: {exc}")
        return

    try:
        df = list_jobs(session=session)
    except Exception:
        return

    for _, row in df.iterrows():
        st = str(row.get("status", "")).upper()
        if st in {"RUNNING", "PENDING", "STARTING", "QUEUED"}:
            jid = f"CONCEALED_DB.PUBLIC.{row['name']}"
            try:
                j = get_job(jid, session=session)
                with _PRINT_LOCK:
                    print(f"[{acct_label}] Canceling active GPU job {jid} ({st}) to free compute pool...")
                j.cancel()
            except Exception:
                pass


def _print_stage_epoch_reports(
    session,
    acct_label: str,
    out_dir_path: Path,
    printed_epochs: set[int],
) -> int:
    out_dir_path.mkdir(parents=True, exist_ok=True)
    try:
        session.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/training_log.json", str(out_dir_path))
    except Exception:
        return 0
    log_file = out_dir_path / "training_log.json"
    if not log_file.exists():
        return 0
    try:
        history = json.loads(log_file.read_text(encoding="utf-8"))
    except Exception:
        return 0

    max_ep = 0
    for entry in history:
        ep = int(entry.get("epoch", 0))
        max_ep = max(max_ep, ep)
        if ep in printed_epochs:
            continue
        printed_epochs.add(ep)
        val_s = entry.get("val", {})
        el = float(entry.get("elapsed_sec", 0.0))
        with _PRINT_LOCK:
            print(
                f"\n+---------------------------------------------------------------------------------------+\n"
                f"| [{acct_label}] EPOCH {ep:02d} ANALYTICS REPORT ({el:.1f}s) [Synced from @MODEL_STAGE]\n"
                f"+---------------------------------------------------------------------------------------+\n"
                f"| Feature Disruption : Patch Cos = {val_s.get('patch_cos_sim', 0.0):.4f} | Salient Foreground Cos = {val_s.get('salient_patch_cos', 0.0):.4f} | Global Cos = {val_s.get('global_cos_sim', 0.0):.4f}\n"
                f"| Identification Stat: Feature ID Evasion = {val_s.get('feature_reid_evasion_pct', 0.0):5.1f}% | Semantic Category Flip = {val_s.get('semantic_neighbor_flip_pct', 0.0):5.1f}%\n"
                f"| Spatial Masking    : Patches <0.70 Sim  = {val_s.get('concealed_patches_70_pct', 0.0):5.1f}% | Patches <0.50 Sim      = {val_s.get('concealed_patches_50_pct', 0.0):5.1f}%\n"
                f"| Visual Stealth     : PSNR = {val_s.get('psnr_db', 0.0):.2f} dB | Chroma Shift = {val_s.get('chroma_rms_255', 0.0):.2f}/255 | L_inf = {val_s.get('linf_255', 0.0):.2f}/255 | UAP Ratio = {val_s.get('uap_collapse_ratio', 0.0):.3f}\n"
                f"| Per-Transformer Breakdown:",
                flush=True,
            )
            for k, v in val_s.items():
                if k.startswith("patch_cos/"):
                    s_name = k.split("/", 1)[1]
                    s_c = val_s.get(f"salient_cos/{s_name}", 0.0)
                    g_c = val_s.get(f"global_cos/{s_name}", 0.0)
                    ev_p = val_s.get(f"reid_evasion_pct/{s_name}", 0.0)
                    fl_p = val_s.get(f"semantic_flip_pct/{s_name}", 0.0)
                    print(
                        f"|   * {s_name:28s} -> PatchCos: {v:.4f} | SalientCos: {s_c:.4f} | GlobalCos: {g_c:.4f} | ID Evasion: {ev_p:5.1f}% | SemFlip: {fl_p:5.1f}%",
                        flush=True,
                    )
            print("+---------------------------------------------------------------------------------------+", flush=True)
    return max_ep


def _sync_and_merge_fleet_checkpoints(
    out_dir: Path,
    worker_dirs: List[Path],
    base_checkpoint: Optional[Path] = None,
) -> Optional[Path]:
    """Merge all available worker `best_generator.pt` checkpoints into `out_dir / best_generator.pt` + ONNX."""
    ckpt_paths = [d / "best_generator.pt" for d in worker_dirs if (d / "best_generator.pt").exists()]
    if not ckpt_paths:
        return None

    merged_pt = out_dir / "best_generator.pt"
    base_ckpt = base_checkpoint if (base_checkpoint and base_checkpoint.exists()) else None

    gen, merge_info = merge_generator_checkpoints(
        checkpoint_paths=ckpt_paths,
        output_path=merged_pt,
        base_checkpoint=base_ckpt,
        task_vector_scaling=1.15 if len(ckpt_paths) > 1 else 1.0,
        device="cpu",
    )

    m = merge_info.get("metrics", {})
    print(
        f"\n+=======================================================================================+\n"
        f"| MULTI-ACCOUNT FLEET MODEL SOUP / TASK-VECTOR MERGE COMPLETE ({merge_info['num_merged']} GPU Nodes)\n"
        f"+=======================================================================================+\n"
        f"| Worker Weights     : {merge_info['weights']}\n"
        f"| Merged Val Metrics : PatchCos = {m.get('patch_cos_sim', 0.0):.4f} | SalientCos = {m.get('salient_patch_cos', 0.0):.4f} | SemanticFlip = {m.get('semantic_neighbor_flip_pct', 0.0):.1f}%\n"
        f"| Saved Checkpoint   : {merged_pt}\n"
        f"+=======================================================================================+",
        flush=True,
    )

    try:
        onnx_info = export_to_onnx(gen, out_dir / "generator.onnx", verify=True)
        print(f"Exported unified ONNX model -> {onnx_info['onnx_path']} ({onnx_info['size_mb']} MB)")
    except Exception as exc:
        print(f"  (ONNX export warning: {exc})")

    return merged_pt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single & Multi-Account Fleet CLI to train Concealed on Snowflake GPU Compute Pools"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Local directory of training images")
    parser.add_argument(
        "--status",
        nargs="?",
        const="LATEST",
        default=None,
        metavar="JOB_ID",
        help="Check status and live analytics across all configured Snowflake accounts",
    )
    parser.add_argument(
        "--cancel",
        nargs="?",
        const="ALL",
        default=None,
        metavar="JOB_ID",
        help="Cancel running Snowflake GPU jobs across all configured accounts",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all staged files (@IMAGE_STAGE, @MODEL_STAGE), cancel active jobs on all Snowflake accounts, and clear local trained_model checkpoints",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge already-downloaded account checkpoints in trained_model/account_*/best_generator.pt",
    )
    parser.add_argument(
        "--accounts",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of account slot numbers from .env to use (e.g., --accounts 1 2 3 4)",
    )
    parser.add_argument(
        "--fleet-strategy",
        type=str,
        choices=["hybrid_specialist", "sharded"],
        default="hybrid_specialist",
        help="Multi-account strategy: 'hybrid_specialist' (data shards + ViT family focus) or 'sharded' (pure data shards)",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to warm-start checkpoint (defaults to trained_model/best_generator.pt if present)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from random initialization instead of warm-starting from existing best_generator.pt",
    )
    parser.add_argument(
        "--poc",
        action="store_true",
        help="Fast Proof-of-Concept mode (~12-15 mins on GPU: 15 epochs, batch_size=12, eps=10/255)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Total images to use across the fleet (in --poc mode defaults to 1,600 per GPU up to 5,000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_model",
        help="Local directory where worker checkpoints and merged best_generator.pt / generator.onnx are saved",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--connection-name",
        type=str,
        default=os.environ.get("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "default"),
        help="Fallback Snowflake CLI connection name if .env is not used",
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

    out_dir = Path(args.output_dir)

    if args.merge:
        worker_dirs = sorted([d for d in out_dir.glob("account_*") if d.is_dir()])
        base_init = out_dir / "init_base_generator.pt"
        _sync_and_merge_fleet_checkpoints(out_dir, worker_dirs, base_checkpoint=base_init)
        return

    try:
        from snowflake.snowpark import Session
    except ImportError as exc:
        raise SystemExit(
            "Snowflake CLI training requires `snowflake-snowpark-python` and `snowflake-ml-python`.\n"
            "Install them once via: pip install snowflake-snowpark-python snowflake-ml-python"
        ) from exc

    accounts = discover_snowflake_accounts(".env", requested_slots=args.accounts)
    sessions_and_specs: List[tuple[SnowflakeAccountSpec, object]] = []

    if accounts:
        print(f"Discovered {len(accounts)} Snowflake account(s) in .env: {', '.join(a.label for a in accounts)}")
        for acct in accounts:
            try:
                sess = _create_session_for_account(acct)
                sessions_and_specs.append((acct, sess))
            except Exception as exc:
                print(f"  [!] Failed to connect to {acct.label} ({acct.account}): {exc}")
        if not sessions_and_specs:
            raise SystemExit("Could not connect to any configured Snowflake accounts. Check your .env credentials.")
    else:
        try:
            sess = Session.builder.config("connection_name", args.connection_name).create()
            fallback_spec = SnowflakeAccountSpec(
                slot=1, account=args.connection_name, user="default", password=""
            )
            sessions_and_specs.append((fallback_spec, sess))
        except Exception as exc:
            raise SystemExit(
                "No Snowflake credentials found in .env or ~/.snowflake/connections.toml.\n"
                "Add SNOWFLAKE_ACCOUNT_1..4, SNOWFLAKE_USER_1..4, SNOWFLAKE_PASSWORD_1..4 to .env."
            ) from exc

    num_workers = len(sessions_and_specs)

    # Handle --cancel across all accounts
    if args.cancel is not None:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(_cancel_account_jobs, sess, acct.label, args.cancel)
                for acct, sess in sessions_and_specs
            ]
            for f in as_completed(futs):
                f.result()
        print("Done canceling jobs across all accounts.")
        return

    # Handle --clean across all accounts
    if args.clean:
        import shutil

        def _clean_account(acct: SnowflakeAccountSpec, sess) -> None:
            _cancel_account_jobs(sess, acct.label, "ALL")
            try:
                sess.sql("CREATE DATABASE IF NOT EXISTS CONCEALED_DB").collect()
                sess.sql("USE DATABASE CONCEALED_DB").collect()
                sess.sql("CREATE SCHEMA IF NOT EXISTS PUBLIC").collect()
                sess.sql("USE SCHEMA PUBLIC").collect()
                # Check how many files currently exist before wiping
                img_rows = []
                mod_rows = []
                try:
                    img_rows = sess.sql("LIST @CONCEALED_DB.PUBLIC.IMAGE_STAGE").collect()
                except Exception:
                    pass
                try:
                    mod_rows = sess.sql("LIST @CONCEALED_DB.PUBLIC.MODEL_STAGE").collect()
                except Exception:
                    pass
                sess.sql(
                    "CREATE OR REPLACE STAGE CONCEALED_DB.PUBLIC.IMAGE_STAGE "
                    "ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE)"
                ).collect()
                sess.sql(
                    "CREATE OR REPLACE STAGE CONCEALED_DB.PUBLIC.MODEL_STAGE "
                    "ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE)"
                ).collect()
                with _PRINT_LOCK:
                    print(
                        f"[{acct.label}] Purged @IMAGE_STAGE ({len(img_rows)} files) and "
                        f"@MODEL_STAGE ({len(mod_rows)} files) -> Stages are now 100% empty."
                    )
            except Exception as exc:
                with _PRINT_LOCK:
                    print(f"[{acct.label}] Clean note: {exc}")

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = [pool.submit(_clean_account, acct, sess) for acct, sess in sessions_and_specs]
            for f in as_completed(futs):
                f.result()

        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[Local] Cleared local checkpoint directory: {out_dir}")
        print("All Snowflake stages and local checkpoints have been wiped clean. Ready to train from scratch!")
        return

    # Handle --status across all accounts
    if args.status is not None:
        from snowflake.ml.jobs import get_job, list_jobs

        worker_dirs: List[Path] = []
        for acct, sess in sessions_and_specs:
            w_dir = out_dir if num_workers == 1 else (out_dir / f"account_{acct.slot}")
            worker_dirs.append(w_dir)
            print(f"\n==================== {acct.label} ====================")
            try:
                sess.sql("USE DATABASE CONCEALED_DB").collect()
                sess.sql("USE SCHEMA PUBLIC").collect()
                df = list_jobs(session=sess)
                if "created_on" in df.columns:
                    df = df.sort_values("created_on", ascending=False)
            except Exception as exc:
                print(f"[{acct.label}] Database CONCEALED_DB not provisioned yet ({exc}).")
                continue

            target_id = args.status
            if target_id == "LATEST":
                if len(df) == 0:
                    print(f"[{acct.label}] No Snowflake ML jobs found.")
                    continue
                active_rows = df[df["status"].astype(str).str.upper().isin(["RUNNING", "PENDING", "STARTING", "QUEUED"])]
                chosen_row = active_rows.iloc[0] if len(active_rows) > 0 else df.iloc[0]
                target_id = f"CONCEALED_DB.PUBLIC.{chosen_row['name']}"

            try:
                job = get_job(target_id, session=sess)
                print(f"[{acct.label}] Job ID : {job.id}")
                print(f"[{acct.label}] Status : {job.status}")
            except Exception as exc:
                print(f"[{acct.label}] Could not fetch job {target_id}: {exc}")
                continue

            printed: set[int] = set()
            latest_ep = _print_stage_epoch_reports(sess, acct.label, w_dir, printed)
            if latest_ep > 0:
                print(f"[{acct.label}] Syncing latest checkpoint (Epoch {latest_ep}) -> {w_dir} ...")
                try:
                    sess.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/", str(w_dir))
                    (w_dir / "samples").mkdir(parents=True, exist_ok=True)
                    sess.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/samples/", str(w_dir / "samples"))
                except Exception as exc:
                    print(f"[{acct.label}] Checkpoint sync note: {exc}")
            else:
                print(f"[{acct.label}] --- Latest Container Logs (Tail) ---")
                try:
                    lines = job.get_logs(as_list=True)
                    for ln in lines[-25:]:
                        print(f"  {ln}")
                except Exception:
                    pass

        if num_workers > 1:
            base_init = out_dir / "init_base_generator.pt"
            _sync_and_merge_fleet_checkpoints(out_dir, worker_dirs, base_checkpoint=base_init)
        elif (out_dir / "best_generator.pt").exists():
            gen, _ = load_generator_checkpoint(out_dir / "best_generator.pt", device="cpu")
            onnx_info = export_to_onnx(gen, out_dir / "generator.onnx", verify=True)
            print(f"Updated local ONNX model -> {onnx_info['onnx_path']} ({onnx_info['size_mb']} MB)")
        return

    if not args.data_dir:
        parser.error("--data-dir is required when launching a training job (or pass --status / --cancel / --merge).")

    all_images = discover_images(args.data_dir, deduplicate=True)
    if not all_images:
        raise SystemExit(f"No supported images found in {args.data_dir}")

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    if args.profile and args.profile != "default":
        prof = base_config.get("surrogates", {}).get("profiles", {}).get(args.profile)
        if prof:
            base_config["surrogates"]["train_models"] = prof["train_models"]
            if "cpu_offload" in prof:
                base_config["surrogates"]["cpu_offload"] = prof["cpu_offload"]
        if args.profile == "ocr":
            base_config.setdefault("generator", {})["hybrid_global_weight"] = 0.35

    if args.poc:
        # Scale total dataset pool with number of GPUs so 4 GPUs cover all 5,000 images in the same time!
        total_poc_images = min(len(all_images), 1600 * num_workers)
        print(
            f"  -> Multi-GPU Proof-of-Concept (--poc) across {num_workers} Snowflake Account(s): "
            f"{total_poc_images} total images ({total_poc_images // num_workers}/GPU), "
            f"15 epochs, batch_size=12, eps=10/255, strategy={args.fleet_strategy}"
        )
        base_config.setdefault("generator", {})["mode"] = "native"
        base_config.setdefault("generator", {})["canonical_size"] = 256
        base_config.setdefault("generator", {})["epsilon_255"] = 10.0
        base_config.setdefault("surrogates", {})["sequential_grad_accum"] = False
        base_config.setdefault("loss", {})["patch_cosine_weight"] = 3.5
        base_config.setdefault("loss", {})["global_cosine_weight"] = 2.0
        base_config.setdefault("loss", {})["patch_dispersion_weight"] = 0.8
        base_config.setdefault("training", {})["train_resolution"] = 256
        base_config.setdefault("training", {})["batch_size"] = 12
        base_config.setdefault("training", {})["grad_accumulation_steps"] = 1
        base_config.setdefault("training", {})["epochs"] = 15
        base_config.setdefault("training", {})["warmup_epochs"] = 1
        base_config.setdefault("training", {})["lr"] = 6.0e-4
        base_config.setdefault("training", {})["max_images"] = total_poc_images

    if args.max_images is not None:
        base_config.setdefault("training", {})["max_images"] = args.max_images
    if args.epochs is not None:
        base_config.setdefault("training", {})["epochs"] = args.epochs
    if args.surrogates is not None:
        base_config.setdefault("surrogates", {})["train_models"] = [{"name": s, "weight": 1.0} for s in args.surrogates]

    base_config.setdefault("training", {})["auto_export_onnx"] = False

    # Check warm-start checkpoint so all fleet workers start from the same weight basin
    out_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt_path: Optional[Path] = None
    if not args.from_scratch:
        candidate = Path(args.init_checkpoint) if args.init_checkpoint else (out_dir / "best_generator.pt")
        if candidate.exists():
            init_ckpt_path = out_dir / "init_base_generator.pt"
            if candidate.resolve() != init_ckpt_path.resolve():
                import shutil

                shutil.copy2(candidate, init_ckpt_path)
            print(f"[Prep] Warm-starting all {num_workers} account(s) from {candidate} (saved base snapshot -> {init_ckpt_path.name})")

    # Pre-build the shared surrogate tar archive once locally before parallel account provisioning
    profile_tag = args.profile or "custom"
    local_hf_tar = _prepare_local_surrogate_tar(base_config, profile_key=profile_tag)

    from snowflake.ml.jobs import remote

    repo_root = Path(__file__).resolve().parent.parent
    import timm

    timm_pkg_dir = Path(timm.__file__).resolve().parent
    job_imports = [
        (str(repo_root / "concealed"), "concealed"),
        (str(timm_pkg_dir), "timm"),
    ]

    def _provision_and_launch_worker(worker_idx: int, acct: SnowflakeAccountSpec, sess):
        with _PRINT_LOCK:
            print(f"[{acct.label}] [1/4] Provisioning Database, Stages, and GPU Compute Pool ({args.gpu_family})...")
        for stmt in CORE_SQL_STATEMENTS:
            sess.sql(stmt.format(instance_family=args.gpu_family)).collect()

        _cancel_account_jobs(sess, acct.label, "ALL")

        has_external_access = True
        try:
            for stmt in EGRESS_SQL_STATEMENTS:
                sess.sql(stmt).collect()
        except Exception as exc:
            if "trial account" in str(exc).lower() or "509009" in str(exc):
                has_external_access = False
                with _PRINT_LOCK:
                    print(f"[{acct.label}] Trial Account detected -> using Stage-Bridged Offline Mode.")
            else:
                raise

        max_imgs = base_config.get("training", {}).get("max_images")
        pre_sharded = _upload_image_bundle_for_worker(
            session=sess,
            acct_label=acct.label,
            all_images=all_images,
            worker_idx=worker_idx,
            num_workers=num_workers,
            max_images=int(max_imgs) if max_imgs is not None else None,
            force_upload=args.force_upload,
        )

        worker_cfg, role_desc = build_worker_config(
            base_config=base_config,
            worker_idx=worker_idx,
            num_workers=num_workers,
            fleet_strategy=args.fleet_strategy,
            pre_sharded_bundle=pre_sharded,
        )

        hf_bundle_name: str | None = None
        if not has_external_access:
            hf_bundle_name = local_hf_tar.name
            stage_target = f"@CONCEALED_DB.PUBLIC.MODEL_STAGE/{hf_bundle_name}"
            if args.force_upload or not _stage_has_file(sess, stage_target):
                size_mb = local_hf_tar.stat().st_size / (1024 * 1024)
                with _PRINT_LOCK:
                    print(f"[{acct.label}] Uploading {hf_bundle_name} ({size_mb:.1f} MB) to @MODEL_STAGE ...")
                sess.file.put(
                    local_hf_tar.resolve().as_posix(),
                    "@CONCEALED_DB.PUBLIC.MODEL_STAGE",
                    auto_compress=False,
                    overwrite=True,
                    parallel=16,
                )
            else:
                with _PRINT_LOCK:
                    print(f"[{acct.label}] Found cached {stage_target} on @MODEL_STAGE.")

        has_init_ckpt = False
        if init_ckpt_path is not None and init_ckpt_path.exists():
            with _PRINT_LOCK:
                print(f"[{acct.label}] Uploading warm-start checkpoint -> @MODEL_STAGE/init_generator.pt ...")
            sess.file.put(
                init_ckpt_path.resolve().as_posix(),
                "@CONCEALED_DB.PUBLIC.MODEL_STAGE/init",
                auto_compress=False,
                overwrite=True,
            )
            has_init_ckpt = True

        remote_kwargs = {
            "stage_name": "MODEL_STAGE",
            "imports": job_imports,
            "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "session": sess,
        }
        if has_external_access:
            remote_kwargs["external_access_integrations"] = ["CONCEALED_HF_ACCESS"]
            remote_kwargs["pip_requirements"] = ["timm", "onnx", "onnxruntime"]

        @remote("CONCEALED_GPU_POOL", **remote_kwargs)
        def run_concealed_gpu_training(
            cfg: dict,
            cached_hf_bundle: str | None,
            warm_start_ckpt_name: str | None,
        ) -> dict:
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
                print(f"Downloading staged surrogate weights {cached_hf_bundle} from @MODEL_STAGE ...", flush=True)
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

            local_init_ckpt: Path | None = None
            if warm_start_ckpt_name:
                try:
                    sp_session.file.get(f"@CONCEALED_DB.PUBLIC.MODEL_STAGE/init/{warm_start_ckpt_name}", str(tmp_root))
                    cand = tmp_root / warm_start_ckpt_name
                    if cand.exists():
                        local_init_ckpt = cand
                except Exception as exc:
                    print(f"Warm-start download note: {exc}", flush=True)

            print("Downloading images_bundle.tar from @CONCEALED_DB.PUBLIC.IMAGE_STAGE ...", flush=True)
            sp_session.file.get("@CONCEALED_DB.PUBLIC.IMAGE_STAGE/images_bundle.tar", str(tmp_root))
            img_matches = list(tmp_root.glob("images_bundle.tar*"))
            if img_matches:
                with tarfile.open(img_matches[0], "r:*") as tar:
                    tar.extractall(path=local_imgs)
                img_matches[0].unlink()
            else:
                sp_session.file.get("@CONCEALED_DB.PUBLIC.IMAGE_STAGE", str(local_imgs))

            def _sync_epoch_artifacts(ep: int, out_dir_path: Path, _val_metrics: dict) -> None:
                for fname in ("best_generator.pt", "latest_generator.pt", "training_log.json"):
                    p = out_dir_path / fname
                    if p.exists():
                        sp_session.file.put(
                            str(p), "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest", auto_compress=False, overwrite=True
                        )
                sample_p = out_dir_path / "samples" / f"epoch_{ep:03d}.png"
                if sample_p.exists():
                    sp_session.file.put(
                        str(sample_p),
                        "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/samples",
                        auto_compress=False,
                        overwrite=True,
                    )

            _, metrics = train(
                config=cfg,
                data_dir=local_imgs,
                output_dir=local_out,
                device_str="cuda",
                on_epoch_end=_sync_epoch_artifacts,
                init_checkpoint=local_init_ckpt,
            )

            for fpath in (local_out / "samples").glob("*.png"):
                sp_session.file.put(
                    str(fpath), "@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/samples", auto_compress=False, overwrite=True
                )
            return metrics

        with _PRINT_LOCK:
            print(f"[{acct.label}] [3/4] Dispatching GPU job ({role_desc}) to CONCEALED_GPU_POOL...")
            job = run_concealed_gpu_training(
                worker_cfg,
                hf_bundle_name,
                init_ckpt_path.name if (has_init_ckpt and init_ckpt_path) else None,
            )
            print(f"[{acct.label}] Dispatched! Job ID: {job.id} | Role: {role_desc}")
        return acct, sess, job, role_desc

    # Launch all accounts in parallel
    active_workers = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_provision_and_launch_worker, idx, acct, sess)
            for idx, (acct, sess) in enumerate(sessions_and_specs)
        ]
        for fut in as_completed(futures):
            active_workers.append(fut.result())

    active_workers.sort(key=lambda x: x[0].slot)
    print(
        f"\nAll {len(active_workers)} Snowflake GPU node(s) are running! "
        f"Streaming unified fleet analytics (press Ctrl+C after any epoch to sync & merge)..."
    )

    seen_lines_per_slot: Dict[int, set[str]] = {w[0].slot: set() for w in active_workers}
    printed_epochs_per_slot: Dict[int, set[int]] = {w[0].slot: set() for w in active_workers}
    worker_dirs: Dict[int, Path] = {
        w[0].slot: (out_dir if num_workers == 1 else (out_dir / f"account_{w[0].slot}"))
        for w in active_workers
    }

    noise_tokens = (
        "logger=",
        "otelcol",
        "prometheus",
        "tsdb",
        "Metric sent (UDP",
        "UNEXPECTED |",
        "text_model.",
        "logit_scale",
        "logit_bias",
        "visual_projection",
        "-------------------------------------------------------------+",
        "%|",
    )

    try:
        while True:
            all_finished = True
            for acct, sess, job, _role in active_workers:
                st = str(job.status).upper()
                if st not in {"DONE", "FAILED", "CANCELLED", "INTERNAL_ERROR", "DELETED"}:
                    all_finished = False

                try:
                    lines = job.get_logs(as_list=True)
                    if isinstance(lines, list):
                        seen = seen_lines_per_slot[acct.slot]
                        for ln in lines:
                            s_ln = ln.strip()
                            if s_ln and s_ln not in seen and not any(tok in ln for tok in noise_tokens):
                                seen.add(s_ln)
                                prefix = f"[{acct.label}] " if num_workers > 1 else ""
                                with _PRINT_LOCK:
                                    print(f"{prefix}{ln}", flush=True)
                except Exception:
                    pass

                _print_stage_epoch_reports(
                    sess,
                    acct.label,
                    worker_dirs[acct.slot],
                    printed_epochs_per_slot[acct.slot],
                )

            if all_finished:
                break
            time.sleep(4.0)
    except KeyboardInterrupt:
        print("\n[Interrupted] Syncing latest checkpoints from all active Snowflake accounts...")

    # Download each worker's latest checkpoint and samples
    for acct, sess, job, role_desc in active_workers:
        w_dir = worker_dirs[acct.slot]
        w_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{acct.label}] [4/4] Downloading trained artifacts ({role_desc}) -> {w_dir} ...")
        try:
            sess.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/", str(w_dir))
            (w_dir / "samples").mkdir(parents=True, exist_ok=True)
            sess.file.get("@CONCEALED_DB.PUBLIC.MODEL_STAGE/latest/samples/", str(w_dir / "samples"))
        except Exception as exc:
            print(f"  [{acct.label}] Download note: {exc}")

    if num_workers > 1:
        _sync_and_merge_fleet_checkpoints(
            out_dir=out_dir,
            worker_dirs=[worker_dirs[w[0].slot] for w in active_workers],
            base_checkpoint=init_ckpt_path,
        )
    else:
        best_pt = out_dir / "best_generator.pt"
        if best_pt.exists():
            gen, _ = load_generator_checkpoint(best_pt, device="cpu")
            onnx_info = export_to_onnx(gen, out_dir / "generator.onnx", verify=True)
            print(f"Exported verified real-time ONNX model -> {onnx_info['onnx_path']} ({onnx_info['size_mb']} MB)")


if __name__ == "__main__":
    main()
