"""
Render-only formation specifications for Paper 2.

Stores desired relative offsets M_i without implementing consensus control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FormationSpec:
    """
    Desired group formation for visualization (Paper 2 Eqs. 27–29).

    ``offsets`` maps member agent_id → 2D offset from ``leader_id``.
    """

    group_id: int
    leader_id: int
    member_ids: tuple[int, ...]
    offsets: dict[int, NDArray[np.float64]] = field(default_factory=dict)

    def desired_position(
        self,
        leader_position: NDArray[np.float64],
        member_id: int,
    ) -> NDArray[np.float64] | None:
        """Compute desired world position for a group member."""
        if member_id not in self.offsets:
            return None
        return leader_position + self.offsets[member_id]
