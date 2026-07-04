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

# ── Search extension target colours ──────────────────────────────
# Static target: white diamond
COLOR_TARGET_STATIC = (230, 230, 230)
# Dynamic target (moving): amber
COLOR_TARGET_DYNAMIC = (255, 180, 40)
# Time-varying target: violet
COLOR_TARGET_TIME_VARYING = (190, 120, 255)
# Tracked target: bright green outline
COLOR_TARGET_TRACKED = (50, 220, 100)
# Completed target: muted teal
COLOR_TARGET_COMPLETED = (60, 160, 140)
# Lost target: red
COLOR_TARGET_LOST = (220, 60, 60)
# Detection ring overlay
COLOR_DETECTION_RING = (100, 200, 100)
# Tracking path history
COLOR_TRACKING_PATH_OLD = (30, 80, 50)
COLOR_TRACKING_PATH_NEW = (50, 220, 100)
# Search spiral / expanding search waypoints
COLOR_SEARCH_WAYPOINT = (200, 160, 80)
# Assignment line (UAV → assigned target)
COLOR_ASSIGNMENT_LINE = (180, 100, 220)
# ─────────────────────────────────────────────────────────────────

# Dashboard chrome
COLOR_DASHBOARD_BG = (24, 28, 36)
COLOR_DASHBOARD_BORDER = (55, 62, 74)
COLOR_DASHBOARD_TITLE = (180, 190, 200)
COLOR_DASHBOARD_TEXT = (220, 225, 230)
COLOR_DASHBOARD_MUTED = (140, 148, 158)

TRAIL_MAX_LENGTH = 150
VELOCITY_ARROW_SCALE = 2.0
DASHBOARD_WIDTH_PX = 220
TRACKING_PATH_MAX_LENGTH = 80
