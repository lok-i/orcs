"""Tasks — one self-registering sub-package per task.

  uolm      Uni-Object Loco-Manipulation — the adapter reads OBJECT kinematics
  perloco   Perceptive Locomotion — the adapter reads a TERRAIN height scan
  dodge     Whole-body evasion — the adapter reads BALL kinematics

A task owns its env cfg, its mdp term library and its obs groups. It does NOT
own an agent (`orcs.core.rl`) or a path (`orcs.core.paths`), and it never
imports a sibling task. Shared robot/control terms belong in core; task-state
semantics remain in the owning task. See docs/ethos.md.
"""
