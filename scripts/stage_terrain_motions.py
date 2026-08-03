#!/usr/bin/env python
"""Wrapper — the code lives in `orcs.cli.stage_terrain_motions` so it ships in a wheel.
Equivalent console entry point: see [project.scripts] in pyproject.toml."""

from orcs.cli.stage_terrain_motions import main

if __name__ == "__main__":
    main()
