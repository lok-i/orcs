"""UOLM's scene and domain choices for the shared repertoire benchmark."""

from orcs.core.motion_benchmark.contract import BenchmarkSpec
from orcs.core.robustness import strip_domain


def prepare(cfg, nominal: bool) -> None:
    cfg.events.pop("virtual_object_force", None)
    cfg.events.pop("policy_update_counter", None)
    if nominal:
        strip_domain(cfg)
        cfg.commands["motion"].object_init_pose_range = {}
        cfg.commands["motion"].object_in_contact_velocity_range = {}


def configure_batch(cfg, clips: list[dict]) -> None:
    entity = cfg.scene.entities[cfg.commands["motion"].object_entity_name]
    if hasattr(entity, "assignment"):
        contexts = tuple(clip["context"] for clip in clips)
        def assignment(num_envs):
            if num_envs != len(contexts):
                raise ValueError("Benchmark assignment does not match batch size")
            return list(contexts)
        entity.assignment = assignment


SPEC = BenchmarkSpec(prepare=prepare, configure_batch=configure_batch)
