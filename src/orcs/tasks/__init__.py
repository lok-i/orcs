"""Tasks — one self-registering sub-package per task.

  uolm      Uni-Object Loco-Manipulation — the adapter reads OBJECT kinematics
  perloco   Perceptive Locomotion — the adapter reads a TERRAIN height scan

A task owns its env cfg, its mdp term library and its obs groups. It does NOT
own an agent (`orcs.core.rl`) or a path (`orcs.core.paths`), and it never
imports a sibling task — the two differ in exactly one obs group, so a term
duplicated across them is a bug. See docs/ethos.md.
"""
