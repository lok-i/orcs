"""Preserve the compiled robot contract used by existing SONIC adapters."""

import hashlib
import json
from pathlib import Path

import mjlab  # noqa: F401 -- complete task discovery before importing ORCS
import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.scene import Scene, SceneCfg
from mocke.sonic import profile

from orcs.assets.g1 import (
    _preserve_legacy_ankle_limits,
    flat_hand_spec,
    get_g1_flat_hand_cfg,
)

FIELDS = (
    "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_limited",
    "jnt_range", "jnt_pos", "jnt_axis", "qpos0", "dof_armature",
    "dof_damping", "dof_frictionloss", "body_parentid", "body_pos", "body_quat",
    "body_ipos", "body_iquat", "body_mass", "body_inertia",
    "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_gainprm",
    "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
    "geom_bodyid", "geom_type", "geom_size", "geom_pos", "geom_quat",
    "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction",
    "geom_solref", "geom_solimp", "exclude_signature", "tendon_range",
)


def robot_contract():
    cfg = profile.robot_cfg(base=get_g1_flat_hand_cfg())
    model = Scene(SceneCfg(entities={"robot": cfg}), device="cpu").compile()
    return {
        "sizes": {k: int(getattr(model, k)) for k in ("nq", "nv", "nu", "ntendon")},
        "joint_names": [model.joint(i).name for i in range(model.njnt)],
        "actuator_names": [model.actuator(i).name for i in range(model.nu)],
        "fields": {
            k: {"shape": list(getattr(model, k).shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(getattr(model, k)).tobytes()).hexdigest()}
            for k in FIELDS
        },
    }


def test_compiled_robot_matches_pre_upgrade_contract():
    expected = json.loads((Path(__file__).parent / "fixtures/g1-contract.json").read_text())
    actual = robot_contract()
    for key in ("sizes", "joint_names", "actuator_names"):
        assert actual[key] == expected[key], key
    for name in FIELDS:
        assert actual["fields"][name] == expected["fields"][name], name


def test_known_ankle_stops_removed_without_removing_other_tendons():
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    assert len(list(spec.tendons)) == 8
    other = spec.add_tendon(name="independent_knee", limited=True, range=[-1, 1])
    other.wrap_joint("left_knee_joint", 1.0)
    _preserve_legacy_ankle_limits(spec)
    assert [t.name for t in spec.tendons] == ["independent_knee"]
    assert len(list(flat_hand_spec().tendons)) == 0


def test_unrecognized_ankle_mechanics_require_review():
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    next(iter(spec.tendons)).range[1] += 0.01
    with pytest.raises(ValueError, match="Unrecognized G1 ankle tendon"):
        _preserve_legacy_ankle_limits(spec)
    assert len(list(spec.tendons)) == 8  # validation precedes mutation
