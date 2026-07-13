# Dynamic Obstacles — Revalidation Status

Dynamic obstacle collision-resolution fix is implemented and verified
functionally via forced-collision test (commit 2bde77a) — see
docs/session_logs or commit message for details.

Full paper-style scenario validation (static/slow/equal_speed/fast/mixed
obstacles x 10/20 UAVs) was attempted but deferred due to runtime cost
(~1hr+ for full batch); a reduced script
(experiments/scripts/run_obstacle_scenarios.py) exists and is ready to
run when time permits, but has not yet been executed to completion.

This is a known gap — Member 2 subsystem is functionally fixed but not
yet quantitatively validated across obstacle-speed categories.
