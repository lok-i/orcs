# sortr — mjlab task package
#
# mjlab imports this file at startup (via the entry-point in pyproject.toml).
# Call register_mjlab_task() once per task you want to expose.
#
# ── single-task project ────────────────────────────────────────────────────────
# import mjlab.tasks
# import mjlab.rl
# from sortr.env_cfg import train_cfg, play_cfg
#
# mjlab.tasks.register_mjlab_task(
#     task_id="Mjlab-MyTask",           # string used with train / play CLI
#     env_cfg=train_cfg(),              # ManagerBasedRlEnvCfg for training
#     play_env_cfg=play_cfg(),          # same but DR off, fewer envs
#     rl_cfg=mjlab.rl.RslRlOnPolicyRunnerCfg(
#         experiment_name="sortr",   # groups runs in wandb / logs/
#     ),
#     runner_cls=None,                  # None -> default MjlabOnPolicyRunner
# )
#
# ── multi-task project (e.g. mjlab_playground style) ──────────────────────────
# import mjlab.tasks
# import mjlab.rl
# from sortr.task_a.env_cfg import train_cfg as a_train, play_cfg as a_play
# from sortr.task_b.env_cfg import train_cfg as b_train, play_cfg as b_play
#
# for task_id, train_fn, play_fn in [
#     ("Mjlab-TaskA", a_train, a_play),
#     ("Mjlab-TaskB", b_train, b_play),
# ]:
#     mjlab.tasks.register_mjlab_task(
#         task_id=task_id,
#         env_cfg=train_fn(),
#         play_env_cfg=play_fn(),
#         rl_cfg=mjlab.rl.RslRlOnPolicyRunnerCfg(experiment_name=task_id),
#         runner_cls=None,
#     )
