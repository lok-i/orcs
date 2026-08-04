"""Core — robot-generic, task-blind infrastructure.

  paths           filesystem roots (repo/data/deps/assets), env-overridable
  deps            shared-dependency drift check
  _mjlab_compat   import-time mjlab patches (multi-clip cmd, --agent initial, VRAM)
  data            clip discovery (`scan`) + the concatenated timeline (`loader`)
  mdp             mjlab's stock terms + every orcs term that is task-blind,
                  including `MultiClipMotionCommand`
  obs             observation-group plumbing + the robot-only term bundles
  rl              the PPO runner spine + the actor builders
  sensors         robot<->terrain contact + the kill-body vocabulary

**The contract.** Core may know what a joint, a body, a clip and a reference
are. It must **not** know what an *object* or a *terrain* is — the moment a
name here mentions one, it belongs to the task that has one. Nothing here may
import from :mod:`orcs.tasks` or :mod:`orcs.assets`.

That line is grep-testable, which the older "functional only, no semantics"
was not: core carried no semantics only for as long as it carried no terms,
and the second task (perloco) made term sharing unavoidable.
"""
