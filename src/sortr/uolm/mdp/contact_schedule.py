"""Cached, name-queryable narrowphase contact schedule for demo motions.

Single source of contact truth. One MuJoCo narrowphase `contact_matrix.npz`
per motion (T, N, N) int8 symmetric, ragged per-motion list (no padding — the
dataset guarantees motion/object/contact T-consistency). The deprecated scalar
`object_motion.npz["contact"]` flag == `object_robot_contact_any()`.

Indexing legend (verified byte-identical across every object in the dataset):
    col 0        = world  (floor geom)
    col 1..N-2   = robot bodies (MuJoCo body order)
    col N-1      = object (name varies per scene; ALWAYS last)

Query by NAME via the legend, never by positional alignment with motion.npz.

API (all vectorized over T, return device tensors):
    pair(midx, a, b)            -> (T,) int8   contact[:, i, j]
    row(midx, name)             -> (T, N) int8 one body vs all
    object_robot_contact_any()  -> (T,) float  robot<->object OR (excl. floor)
    body_object_contacts(m, ns) -> (T, K) float per-body <-> object (graph nodes)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

WORLD_NAME = "world"
_OBJECT_ALIASES = {"object", "Object"}
_FLOOR_ALIASES = {"floor", "world"}


class ContactSchedule:
    def __init__(
        self,
        motion_files: Sequence[str],
        motion_lengths: Sequence[int],
        device: str = "cpu",
    ):
        self.device = device
        self._n: int | None = None
        self._name_to_idx: dict[str, int] = {}
        self._canon_block: tuple[str, ...] = ()

        # pass 1: load present matrices, lock canonical legend
        raw: list[torch.Tensor | None] = []
        for mf in motion_files:
            cm_path = Path(mf).with_name("contact_matrix.npz")
            if cm_path.exists():
                d = np.load(cm_path, allow_pickle=True)
                names = [str(x) for x in np.asarray(d["body_names"]).tolist()]
                mat = torch.as_tensor(d["matrix"], dtype=torch.int8, device=device)
                self._register_legend(names, mat.shape[-1])
                raw.append(mat)
            else:
                raw.append(None)

        if self._n is None:
            raise RuntimeError(
                "ContactSchedule: no contact_matrix.npz found for ANY motion — "
                "cannot establish the contact legend."
            )

        # pass 2: zeros-fallback for missing matrices (T from motion length)
        self._matrix_list: list[torch.Tensor] = []
        n_missing = 0
        for mat, length in zip(raw, motion_lengths, strict=True):
            if mat is None:
                mat = torch.zeros((int(length), self._n, self._n),
                                  dtype=torch.int8, device=device)
                n_missing += 1
            self._matrix_list.append(mat)
        if n_missing:
            print(f"[ContactSchedule] {n_missing}/{len(motion_files)} motions "
                  f"missing contact_matrix.npz -> zeros fallback")

    # ── legend ──

    def _register_legend(self, names: list[str], n: int):
        # object is the last col; the robot+world block must be invariant
        block = tuple(names[:-1])
        if self._n is None:
            self._n = n
            self._canon_block = block
            self._name_to_idx = {nm: i for i, nm in enumerate(block)}
        elif n != self._n or block != self._canon_block:
            raise RuntimeError(
                "ContactSchedule: contact-matrix legend drift across motions — "
                "body order is not consistent.\n"
                f"  expected robot+world block: {self._canon_block}\n"
                f"  got:                        {block}"
            )

    @property
    def object_idx(self) -> int:
        return self._n - 1  # type: ignore

    def _idx(self, name: str) -> int:
        if name in _OBJECT_ALIASES:
            return self.object_idx
        if name in _FLOOR_ALIASES:
            name = WORLD_NAME
        try:
            return self._name_to_idx[name]
        except KeyError as err:
            legend = list(self._name_to_idx) + ["<object>"]
            raise KeyError(f"body '{name}' not in contact legend {legend}") from err

    # ── queries ──

    def pair(self, midx: int, a: str, b: str) -> torch.Tensor:
        """(T,) int8 contact track between bodies `a` and `b`."""
        return self._matrix_list[midx][:, self._idx(a), self._idx(b)]

    def row(self, midx: int, name: str) -> torch.Tensor:
        """(T, N) int8 — body `name` vs every body."""
        return self._matrix_list[midx][:, self._idx(name), :]

    def object_robot_contact_any(self, midx: int) -> torch.Tensor:
        """(T,) float {0,1}: ANY robot body in contact with the object.

        OR of the object row, excluding the world/floor col and self — i.e. the
        robot has control-authority over the object. Subsumes the deprecated
        scalar `object_motion.npz["contact"]` flag.
        """
        r = self._matrix_list[midx][:, self.object_idx, :].clone()
        r[:, 0] = 0                # world/floor
        r[:, self.object_idx] = 0  # self
        return (r != 0).any(dim=-1).float()

    def body_object_contacts(self, midx: int, names: Sequence[str]) -> torch.Tensor:
        """(T, K) float {0,1}: per-body contact with the object — the contact-graph
        node vector for the named robot bodies (order preserved). OR over K with
        `.amax(-1)` if you want any-of."""
        obj_row = self._matrix_list[midx][:, self.object_idx, :]
        cols = torch.tensor([self._idx(n) for n in names], device=self.device)
        return (obj_row[:, cols] != 0).float()


if __name__ == "__main__":
    # smoke test against a real sample (run from repo root)
    import glob
    import os

    cms = sorted(glob.glob(
        "data/retargeted_motions/data/**/contact_matrix.npz", recursive=True))
    assert cms, "no contact_matrix.npz found under data/"
    motion_files = [os.path.dirname(p) + "/motion.npz" for p in cms[:8]]
    lengths = [np.load(m)["body_pos_w"].shape[0] for m in motion_files]

    cs = ContactSchedule(motion_files, lengths, device="cpu")
    print(f"N={cs._n}  object_idx={cs.object_idx}  legend[:4]={list(cs._name_to_idx)[:4]}")

    midx = 0
    d = np.load(Path(motion_files[0]).with_name("contact_matrix.npz"), allow_pickle=True)
    M, names = d["matrix"], [str(x) for x in d["body_names"].tolist()]
    obj = len(names) - 1
    # manual robot<->object OR (excl world + self)
    man = M[:, obj, :].copy()
    man[:, 0] = 0
    man[:, obj] = 0
    man = (man != 0).any(1).astype("float32")
    got = cs.object_robot_contact_any(midx).numpy()
    assert (man == got).all(), "object_robot_contact_any mismatch"
    # pair() == manual lookup (object via the motion-agnostic "object" alias)
    assert (cs.pair(midx, names[1], "object").numpy() == M[:, 1, obj]).all()
    print(f"  object_robot_contact_any: {int(got.sum())}/{len(got)} contact frames  OK")
    print("PASSED")
