"""
Research-style colour palette for Pygame visualization layers.
"""

from __future__ import annotations

# Background and static geometry
COLOR_BACKGROUND = (18, 22, 28)
COLOR_OBSTACLE = (90, 90, 95)

# Exploration grid (subtle, low-contrast)
COLOR_GRID_UNEXPLORED = (28, 36, 52)
COLOR_GRID_EXPLORED = (32, 58, 62)

# Frontiers (Paper 1 Sec. 4–5)
COLOR_FRONTIER_CELL = (210, 180, 60)
COLOR_FRONTIER_CENTROID = (255, 214, 90)

# Agent roles
COLOR_UAV = (66, 133, 244)
COLOR_UGV = (76, 175, 80)
COLOR_MASTER = (255, 193, 7)
COLOR_AGENT_ID = (200, 210, 220)

# Motion and BSA decision overlays
COLOR_TRAIL_OLD = (40, 70, 110)
COLOR_TRAIL_NEW = (120, 170, 230)
COLOR_VELOCITY = (100, 220, 180)
COLOR_SENSOR_RING = (80, 140, 200)
COLOR_TARGET_MARKER = (255, 140, 60)
COLOR_TARGET_LINE = (255, 140, 60)

# Dashboard chrome
COLOR_DASHBOARD_BG = (24, 28, 36)
COLOR_DASHBOARD_BORDER = (55, 62, 74)
COLOR_DASHBOARD_TITLE = (180, 190, 200)
COLOR_DASHBOARD_TEXT = (220, 225, 230)
COLOR_DASHBOARD_MUTED = (140, 148, 158)

TRAIL_MAX_LENGTH = 150
VELOCITY_ARROW_SCALE = 2.0
DASHBOARD_WIDTH_PX = 220

# ---------------------------------------------------------------------------
# Dynamic obstacle colors (SDS §37)
# Classified by speed relative to UAV max speed (1.5 m/s):
#   Slow  : speed < 0.8 × v_uav   → Green
#   Equal : 0.8 ≤ speed ≤ 1.2 × v_uav → Orange
#   Fast  : speed > 1.2 × v_uav   → Red
# ---------------------------------------------------------------------------
COLOR_DYN_SLOW = (80, 200, 120)       # green
COLOR_DYN_EQUAL = (255, 160, 60)      # orange
COLOR_DYN_FAST = (230, 80, 70)        # red
COLOR_DYN_VELOCITY = (200, 230, 255)  # light blue — distinct from UAV velocity arrows
COLOR_DYN_SAFETY = (100, 100, 60)     # muted yellow — safety radius ring
COLOR_DYN_PREDICTION = (160, 100, 200)  # purple — predicted trajectory

# Fraction of UAV max speed used to classify obstacle speed category
DYN_SPEED_SLOW_RATIO = 0.8    # below this → Slow
DYN_SPEED_FAST_RATIO = 1.2    # above this → Fast

# Prediction horizon [s] for optional trajectory overlay
DYN_PREDICTION_HORIZON_S = 1.5
