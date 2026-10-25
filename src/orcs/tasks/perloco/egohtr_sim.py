"""EgoHTR-only CCD allocation bridge for the pinned mjlab simulator."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import mujoco
import mujoco_warp as mjwarp
from mjlab.sim import MujocoCfg

# Weak keys keep the setting bound to a compiled EgoHTR model's lifetime.
# Other models pass through to the existing ORCS allocator unchanged.
_CCD_CAPACITY: WeakKeyDictionary[mujoco.MjModel, int] = WeakKeyDictionary()


def _install_ccd_bridge() -> None:
    if getattr(mjwarp.put_data, "_orcs_egohtr_ccd", False):
        return
    original = mjwarp.put_data

    @functools.wraps(original)
    def put_data(m, *args, **kwargs):
        capacity = _CCD_CAPACITY.get(m)
        if (capacity is not None
                and kwargs.get("nccdmax") is None
                and kwargs.get("naccdmax") is None):
            capacity = int(os.environ.get("ORCS_NCCDMAX", capacity))
            if capacity <= 0:
                raise ValueError("EgoHTR CCD capacity must be positive")
            nconmax = kwargs.get("nconmax")
            kwargs["nccdmax"] = (
                capacity if nconmax is None else min(capacity, nconmax)
            )
        return original(m, *args, **kwargs)

    put_data._orcs_egohtr_ccd = True  # type: ignore[attr-defined]
    mjwarp.put_data = put_data


@dataclass
class EgoHtrMujocoCfg(MujocoCfg):
    nccdmax: int = 128
    """Convex collision workspace rows per world; ORCS_NCCDMAX overrides it.

    mjlab 1.4 lacks an allocation setting for this. Register the capacity when
    mjlab applies this configuration to the compiled model, before put_data.
    This is a workspace budget, not a MuJoCo solver option.
    """

    def apply(self, model: mujoco.MjModel) -> None:
        if self.nccdmax <= 0:
            raise ValueError("EgoHTR CCD capacity must be positive")
        super().apply(model)
        _CCD_CAPACITY[model] = self.nccdmax
        _install_ccd_bridge()
