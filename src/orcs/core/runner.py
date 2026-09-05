"""Runner shims orcs owns.

``DistillRunner`` — mjlab's on-policy runner plus the two things a teacher-only load
needs: the guard that a distillation run has a teacher at all, and a reset of the env's
curriculum clock, which mjlab otherwise inherits from the TEACHER's run.

It lives in core, not under a task, for the same reason the agent zoo does: every orcs
task that grows a `-Bcd` row needs exactly this runner, and a consumer (vibe) needs it
too. A second spelling is a second place for the guard to go missing.
"""

from __future__ import annotations

from mjlab.rl.runner import MjlabOnPolicyRunner


class DistillRunner(MjlabOnPolicyRunner):
    """Distillation runner: mjlab's runner + rsl_rl's teacher-loaded guard.

    The guard is not a nicety. A distillation run whose teacher was never loaded still
    trains — against the freshly constructed teacher, which for an adapter agent is the
    frozen SONIC base with zero-init LoRA. That is a policy, it produces plausible
    actions, and the loss falls. You would be distilling the base into the student and
    reading it as success.

    Load one with mjlab's own flags — no orcs plumbing involved:

        train <task> --agent.resume True --wandb-run-path <teacher run>

    `Distillation.load` sees an RL checkpoint's `actor_state_dict`, routes it into the
    TEACHER only, and leaves the iteration counter alone; a distillation checkpoint
    (which carries `student_state_dict`) resumes everything instead.

    Subclasses mjlab's runner rather than rsl_rl's `DistillationRunner` to keep the
    env-state save/load and the `upload_model` gating.
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Refuse to train against an unloaded teacher."""
        if not getattr(self.alg, "teacher_loaded", False):
            raise ValueError(
                "Teacher parameters not loaded — training would distill the frozen base "
                "into the student and look like it was working. Pass "
                "`--agent.resume True --wandb-run-path <teacher run>` (or --agent.load-run)."
            )
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load, then drop the teacher's curriculum clock on a fresh student.

        mjlab's runner restores ``env.common_step_counter`` from the checkpoint
        UNCONDITIONALLY — it has no way to know that ``Distillation.load`` took only the
        teacher's weights out of an RL checkpoint and left the student untouched. So a
        fresh student inherits the converged teacher's step count, and every orcs anneal
        reads its clock off exactly that (``PolicyUpdateCounter`` →
        ``env.policy_update_count``).

        Measured, two terms consume it, and which ones bite is per task:

        - ``VirtualObjectForceCurriculum`` (uolm only) — the VOF wrench decays over
          ``decay_by_policy_iterations`` and snaps to *exactly* zero past it, so an
          inherited clock trains the student with no object assist from its first step
          while the PPO row it is measured against ran the whole ramp.
        - clip-phase annealing (``init_phase_anneal_iterations``, any multi-clip task) —
          inert at its dataclass default of 0, live whenever a run passes the flag.

        A genuine distillation resume must keep its clock — that run's curriculum is its
        own. `Distillation.load` reports that case by loading the iteration counter,
        which is what separates the two here. Note the counter is NOT restored on the
        play path, so this fires there too; harmless, because play pops both
        ``policy_update_counter`` and the curricula that read it.
        """
        infos = super().load(path, load_cfg, strict, map_location)
        if self.current_learning_iteration == 0:
            env = self.env.unwrapped  # type: ignore[attr-defined]
            inherited = getattr(env, "common_step_counter", 0)
            if inherited:
                env.common_step_counter = 0
                print(
                    f"[DistillRunner] fresh student: dropped the teacher's env clock "
                    f"(common_step_counter {inherited} -> 0), so the curriculum starts at "
                    f"iteration 0 rather than terminal."
                )
        return infos
