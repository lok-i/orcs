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
    assert cfg.commands["motion"].table_motion_set_names == ("small-cube-table",)
    assert "table" in cfg.scene.entities


def test_base_and_cube_scenes_instantiate_only_selected_objects(monkeypatch) -> None:
    monkeypatch.setattr(
        env_cfg,
        "_resolve_reconstructed_smpl_seeds",
        lambda _motion_sets, _interaction_names: ("dummy/smpl_motion.npz", 1, 10),
    )

    base = env_cfg.uolm_smpl_env_cfg()
    variants = base.scene.entities["object"].variants
    assert tuple(variants) == ("woodchair2-floor", "tire-floor")
    assert "table" not in base.scene.entities
    assert base.commands["motion"].support_entity_name is None
    assert base.commands["motion"].table_motion_set_names == ()

    small = env_cfg.uolm_smpl_env_cfg(motion_sets=("small-cube-table",))
    assert not hasattr(small.scene.entities["object"], "variants")
    assert "table" in small.scene.entities
    assert small.commands["motion"].support_entity_name == "table"

    big = env_cfg.uolm_smpl_env_cfg(motion_sets=("big-cube-floor",))
    assert not hasattr(big.scene.entities["object"], "variants")
    assert "table" not in big.scene.entities
    assert big.commands["motion"].support_entity_name is None


def test_redundant_small_cube_specializations_are_not_registered() -> None:
    from orcs.tasks.uolm import _TASKS

    task_ids = {task_id for task_id, _, _ in _TASKS}
    assert "Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl" in task_ids
    assert "Orcs-Uolm-BigCubeFloor-AdaptSonic-Smpl" in task_ids
    assert "Orcs-Uolm-SmallCubeTable-PickPlace-AdaptSonic-Smpl" not in task_ids
    assert "Orcs-Uolm-SmallCubeTable-Throw-AdaptSonic-Smpl" not in task_ids
