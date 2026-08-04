#!/usr/bin/env python
"""Wrapper — the code lives in `orcs.cli.rollout_smpl` so it ships in a wheel.
Equivalent console entry point: see [project.scripts] in pyproject.toml."""

from orcs.cli.rollout_smpl import main

if __name__ == "__main__":
    main()
