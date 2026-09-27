"""Unit tests for Multi-Account Snowflake Fleet Sharding, Discovery, and Model Soup Merging."""

from __future__ import annotations

from pathlib import Path
import torch

from concealed.data.dataset import split_shared_val_and_train_shard
from concealed.models.generator import AmortizedObfuscationGenerator
from concealed.snowflake_cli import build_worker_config, discover_snowflake_accounts
from concealed.train import merge_generator_checkpoints, save_checkpoint


def test_split_shared_val_and_disjoint_train_shards() -> None:
    """All 4 accounts must receive the exact same validation gallery and disjoint training shards."""
    fake_files = [Path(f"img_{i:04d}.jpg") for i in range(200)]

    shards_train = []
    shards_val = []
    for s_id in range(4):
        tr, vl = split_shared_val_and_train_shard(
            all_files=fake_files,
            val_split=0.1,
            seed=42,
            num_shards=4,
            shard_id=s_id,
        )
        shards_train.append(set(tr))
        shards_val.append(vl)

    # 1. Every shard must have the exact same ordered validation set (20 images)
    assert len(shards_val[0]) == 20
    for s_id in range(1, 4):
        assert shards_val[s_id] == shards_val[0]

    # 2. Training shards must be pairwise disjoint and union to the remaining 180 images
    for i in range(4):
        assert len(shards_train[i]) == 45
        assert shards_train[i].isdisjoint(set(shards_val[0]))
        for j in range(i + 1, 4):
            assert shards_train[i].isdisjoint(shards_train[j])

    union_train = set().union(*shards_train)
    assert len(union_train) == 180


def test_discover_snowflake_accounts_from_single_env(tmp_path: Path, monkeypatch) -> None:
    """Single .env file should discover both unnumbered slot 1 and numbered slots 1..4."""
    for k in list(monkeypatch._setitem):  # clean any existing SNOWFLAKE_* env vars via monkeypatch
        pass
    for idx in range(1, 6):
        monkeypatch.delenv(f"SNOWFLAKE_ACCOUNT_{idx}", raising=False)
        monkeypatch.delenv(f"SNOWFLAKE_USER_{idx}", raising=False)
        monkeypatch.delenv(f"SNOWFLAKE_PASSWORD_{idx}", raising=False)
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_USER", raising=False)
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "SNOWFLAKE_ACCOUNT=acct1_org\n"
        "SNOWFLAKE_USER=user1\n"
        "SNOWFLAKE_PASSWORD=pass1\n"
        "SNOWFLAKE_ACCOUNT_2=acct2_org\n"
        "SNOWFLAKE_USER_2=user2\n"
        "SNOWFLAKE_PASSWORD_2=pass2\n"
        "SNOWFLAKE_ACCOUNT_3=acct3_org\n"
        "SNOWFLAKE_USER_3=user3\n"
        "SNOWFLAKE_PASSWORD_3=pass3\n"
        "SNOWFLAKE_ACCOUNT_4=YOUR_ACCOUNT_4\n"
        "SNOWFLAKE_USER_4=\n"
        "SNOWFLAKE_PASSWORD_4=\n",
        encoding="utf-8",
    )

    accounts = discover_snowflake_accounts(env_file)
    assert len(accounts) == 3
    assert [a.slot for a in accounts] == [1, 2, 3]
    assert [a.account for a in accounts] == ["acct1_org", "acct2_org", "acct3_org"]

    # Filtering by --accounts 1 3
    subset = discover_snowflake_accounts(env_file, requested_slots=[1, 3])
    assert [a.slot for a in subset] == [1, 3]


def test_build_worker_config_specialist_roles() -> None:
    base_cfg = {
        "training": {"lr": 6.0e-4, "seed": 42},
        "loss": {"patch_cosine_weight": 3.5},
        "surrogates": {
            "train_models": [
                {"name": "google/siglip-base-patch16-224", "weight": 1.0},
                {"name": "openai/clip-vit-base-patch16", "weight": 1.0},
                {"name": "facebook/dinov2-base", "weight": 1.0},
            ]
        },
    }
    cfg0, role0 = build_worker_config(base_cfg, worker_idx=0, num_workers=4, fleet_strategy="hybrid_specialist")
    cfg1, role1 = build_worker_config(base_cfg, worker_idx=1, num_workers=4, fleet_strategy="hybrid_specialist")
    assert "SigLIP" in role0
    assert "CLIP" in role1
    assert cfg0["surrogates"]["train_models"][0]["weight"] > cfg0["surrogates"]["train_models"][1]["weight"]
    assert cfg1["surrogates"]["train_models"][1]["weight"] > cfg1["surrogates"]["train_models"][0]["weight"]


def test_merge_generator_checkpoints_task_vector_soup(tmp_path: Path) -> None:
    """Merging 4 worker checkpoints warm-started from a base checkpoint should produce a valid unified model."""
    cfg = {"generator": {"variant": "tiny", "epsilon_255": 10.0, "mode": "native", "canonical_size": 64}}
    base_gen = AmortizedObfuscationGenerator(variant="tiny", epsilon_255=10.0, mode="native", canonical_size=64)
    base_pt = tmp_path / "base.pt"
    save_checkpoint(base_pt, base_gen, None, cfg, epoch=5, metrics={"patch_cos_sim": 0.82})

    worker_pts = []
    for i, p_cos in enumerate([0.71, 0.74, 0.68, 0.76]):
        w_gen = AmortizedObfuscationGenerator(variant="tiny", epsilon_255=10.0, mode="native", canonical_size=64)
        with torch.no_grad():
            for p in w_gen.parameters():
                p.add_(0.01 * (i + 1))
        w_pt = tmp_path / f"worker_{i}.pt"
        save_checkpoint(
            w_pt,
            w_gen,
            None,
            cfg,
            epoch=15,
            metrics={"patch_cos_sim": p_cos, "salient_patch_cos": p_cos + 0.02, "semantic_neighbor_flip_pct": 35.0},
        )
        worker_pts.append(w_pt)

    merged_pt = tmp_path / "merged.pt"
    merged_gen, info = merge_generator_checkpoints(
        checkpoint_paths=worker_pts,
        output_path=merged_pt,
        base_checkpoint=base_pt,
        task_vector_scaling=1.15,
    )

    assert merged_pt.exists()
    assert info["num_merged"] == 4
    # Worker 2 had the lowest PatchCos (0.68), so it must receive the highest weight in the soup
    assert info["weights"][2] == max(info["weights"])

    x = torch.rand(1, 3, 64, 64)
    out = merged_gen(x)
    assert out.shape == (1, 3, 64, 64)
