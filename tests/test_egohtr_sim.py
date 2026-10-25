"""EgoHTR CCD allocations must not change other tasks or explicit budgets."""

from copy import deepcopy
from dataclasses import asdict

import mujoco
import mujoco_warp as mjwarp
import pytest
from mjlab.sim import MujocoCfg

from orcs.tasks.perloco.egohtr_sim import EgoHtrMujocoCfg


def test_ccd_bridge_only_overrides_models_with_egohtr_configuration(monkeypatch):
    monkeypatch.delenv("ORCS_NCCDMAX", raising=False)
    monkeypatch.setattr(mjwarp, "put_data", lambda m, **kwargs: kwargs)
    ego = mujoco.MjModel.from_xml_string("<mujoco/>")
    other = mujoco.MjModel.from_xml_string("<mujoco/>")
    cfg = deepcopy(EgoHtrMujocoCfg(nccdmax=128))
    assert asdict(cfg)["nccdmax"] == 128
    cfg.apply(ego)
    cfg.apply(ego)  # repeated model configuration does not stack the bridge
    MujocoCfg().apply(other)
    assert mjwarp.put_data(ego, nconmax=256)["nccdmax"] == 128
    assert mjwarp.put_data(other, nconmax=256) == {"nconmax": 256}
    assert mjwarp.put_data(ego, nconmax=64)["nccdmax"] == 64
    assert mjwarp.put_data(ego)["nccdmax"] == 128

    monkeypatch.setenv("ORCS_NCCDMAX", "192")
    assert mjwarp.put_data(ego, nconmax=256)["nccdmax"] == 192
    # Non-EgoHTR calls still arrive at the original allocator unchanged.
    assert mjwarp.put_data(other, nconmax=256) == {"nconmax": 256}
    assert mjwarp.put_data(ego, nconmax=256, nccdmax=32)["nccdmax"] == 32
    assert "nccdmax" not in mjwarp.put_data(ego, nconmax=256, naccdmax=320)

    monkeypatch.setenv("ORCS_NCCDMAX", "0")
    with pytest.raises(ValueError, match="positive"):
        mjwarp.put_data(ego, nconmax=256)
    assert mjwarp.put_data(other, nconmax=256) == {"nconmax": 256}
