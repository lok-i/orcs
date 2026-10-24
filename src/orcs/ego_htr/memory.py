"""Optional synthetic mjlab memory diagnostic; excludes policy/PPO and real scans."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orcs-egohtr-mpl")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/orcs-egohtr-warp")
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg


def probe(worlds=128, terrain="hfield", ccd=50, nccd=64):
    from types import SimpleNamespace

    from orcs.assets.g1 import flat_hand_spec

    args = SimpleNamespace(worlds=worlds, terrain=terrain, ccd=ccd, nccd=nccd)
    wp.init()
    device = wp.get_device("cuda:0")
    spec = flat_hand_spec()
    for geom in list(spec.geoms):
        if geom.contype == 0 and geom.conaffinity == 0:
            spec.delete(geom)
    terrain = spec.worldbody.add_body(name="terrain")
    if args.terrain == "hfield":
        n = 151
        x = np.linspace(-3, 3, n)
        xx, yy = np.meshgrid(x, x)
        heights = 0.12 * np.sin(2 * xx) * np.cos(2 * yy) + 0.05 * np.sin(5 * xx + yy)
        heights[np.abs(xx) < 0.3] = 0
        zmin, zmax = float(heights.min()), float(heights.max())
        spec.add_hfield(
            name="crop",
            nrow=n,
            ncol=n,
            size=[3, 3, zmax - zmin, 0.15],
            userdata=((heights - zmin) / (zmax - zmin)).ravel(),
        )
        terrain.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="crop", pos=[0, 0, zmin]
        )
    else:
        terrain.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
        if args.terrain == "convex":
            # 32 distinct low-complexity convex rocks; no dense scan mesh.
            rng = np.random.default_rng(10)
            for k in range(32):
                v = rng.normal(size=(16, 3))
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                v *= np.array([0.15, 0.14, 0.12])
                v[:, 2] += 0.12
                name = f"rock_{k}"
                spec.add_mesh(name=name, uservert=v.ravel(), maxhullvert=16)
                x, y = rng.uniform(-2.7, 2.7, 2)
                if abs(x) < 0.3:
                    x += 0.7
                terrain.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=[x, y, 0]
                )
    model = spec.compile()
    # Put the nominal feet into shallow terrain contact. Stop before large
    # penetration; exercise collision work rather than a suspended robot.
    terrain_id = model.body("terrain").id
    native_data = mujoco.MjData(model)
    lowered = 0.0
    while lowered <= 0.12:
        native_data.qpos[:] = model.qpos0
        native_data.qpos[2] -= lowered
        mujoco.mj_forward(model, native_data)
        ground_contacts = [
            c for c in native_data.contact if terrain_id in model.geom_bodyid[c.geom]
        ]
        if ground_contacts and min(c.dist for c in ground_contacts) < -0.001:
            break
        lowered += 0.002
    if lowered > 0.12:
        raise RuntimeError("Failed to establish nominal terrain contact")
    model.qpos0[:] = native_data.qpos
    # mjlab does not expose nccdmax in this installed version. Match ORCS's
    # existing wrapper explicitly and record the actual resulting allocation.
    original = mjw.put_data

    def capped(*a, **kw):
        kw.setdefault("nccdmax", args.nccd)
        return original(*a, **kw)

    mjw.put_data = capped
    try:
        before = device.free_memory
        started = time.monotonic()
        cfg = SimulationCfg(
            nconmax=256,
            njmax=768,
            mujoco=MujocoCfg(
                timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=args.ccd
            ),
        )
        sim = Simulation(num_envs=args.worlds, cfg=cfg, model=model, device="cuda:0")
        wp.synchronize()
        construction = time.monotonic() - started
        for _ in range(10):
            sim.forward()
        wp.synchronize()
        elapsed = time.monotonic()
        # Repeated forward evaluates contacts/constraints at the reset pose. Avoid
        # conflating an unactuated fall with a terrain tracking experiment.
        for _ in range(40):
            sim.forward()
        wp.synchronize()
        forward_ms = (time.monotonic() - elapsed) / 40 * 1000
        contact_count = int(sim.wp_data.nacon.numpy()[0])
        max_constraints = int(sim.wp_data.nefc.numpy().max())
        # Exercise the captured step graph as well. This short passive settling
        # run checks allocation/finite state only, not tracking or stable control.
        for _ in range(100):
            sim.step()
        wp.synchronize()
        r = {
            "terrain": args.terrain,
            "worlds": args.worlds,
            "ngeom": model.ngeom,
            "nmesh": model.nmesh,
            "nhfield": model.nhfield,
            "ccd_iterations": args.ccd,
            "nconmax_per_world": 256,
            "nccdmax_per_world": args.nccd,
            "njmax": 768,
            "actual_naccdmax": sim.wp_data.naccdmax,
            "actual_naconmax": sim.wp_data.naconmax,
            "warp_pool_current_MiB": wp.get_mempool_used_mem_current(device) / 2**20,
            "warp_pool_peak_MiB": wp.get_mempool_used_mem_high(device) / 2**20,
            "device_memory_delta_MiB": (before - device.free_memory) / 2**20,
            "construction_seconds": construction,
            "forward_ms": forward_ms,
            "initial_root_lowering_m": lowered,
            "initial_contacts_all_worlds": contact_count,
            "initial_max_constraints": max_constraints,
            "after_100_steps_contacts_all_worlds": int(sim.wp_data.nacon.numpy()[0]),
            "after_100_steps_max_constraints": int(sim.wp_data.nefc.numpy().max()),
            "finite_qpos": bool(np.isfinite(sim.wp_data.qpos.numpy()).all()),
            "scope": "mjlab Simulation construction, CUDA graph capture, 50 forwards in shallow foot contact, 100 passive steps; synthetic terrain. Excludes policy/PPO, sensor pipeline, DR, and motion tracking.",
        }
        return r
    finally:
        mjw.put_data = original
