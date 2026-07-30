"""Core — robot- and task-agnostic infrastructure. Functional only, no semantics.

  paths           filesystem roots (repo/data/deps/assets), env-overridable
  _mjlab_compat   import-time mjlab patches (multi-clip cmd, --agent initial, VRAM)

Nothing here may import from :mod:`orcs.tasks` or :mod:`orcs.assets`.
"""
