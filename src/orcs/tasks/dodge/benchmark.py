"""Dodge success means surviving the throw episode, not tracking a fixed pose."""

from orcs.core.motion_benchmark.contract import BenchmarkSpec
from orcs.core.robustness import strip_domain


def prepare(cfg, nominal: bool) -> None:
    cfg.events["throw_ball"].params["stand_fraction"] = 0.0
    if nominal:
        strip_domain(cfg)
        for group in cfg.observations.values():
            group.enable_corruption = False


SPEC = BenchmarkSpec(prepare=prepare, horizon="episode",
                     completion_term="time_out", tracking_quality=False)
