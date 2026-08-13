from __future__ import annotations

import mjlab  # noqa: F401  # import-order contract: mjlab before orcs

from orcs.tasks.uolm import env_cfg


def test_smpl_training_uses_dataset_filter_without_virtual_object_force(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        env_cfg,
        "_resolve_reconstructed_smpl_seeds",
        lambda _motion_sets, _interaction_names: ("dummy/smpl_motion.npz", 1, 10),
    )

    cfg = env_cfg.uolm_smpl_env_cfg(
        motion_sets=("small-cube-table",),
        interaction_names=("throw",),
    )

    assert "virtual_object_force" not in cfg.events
    assert "policy_update_counter" in cfg.events
    assert cfg.commands["motion"].interaction_names == ("throw",)
