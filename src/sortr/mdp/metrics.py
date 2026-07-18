"""Repose metrics — object tracking metrics surfaced via OmniObjectMotionCommand."""

# All oracle-specific metrics (guard_d, guard_theta, base_track_error,
# ee_track_error) have been removed. Object tracking metrics (error_object_pos,
# error_object_ori, at_goal) are computed inside OmniObjectMotionCommand._update_metrics.

__all__: list[str] = []
