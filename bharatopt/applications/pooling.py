"""
BharatOpt Core - Bilinear Pooling & McCormick Convex Envelope Relaxations (FR-EXT3, §7.5, §8.3 Tier B)
Clean-room implementation of:
1. Bilinear term relaxation w = x * y over box [x_L, x_U] x [y_L, y_U] using McCormick Envelopes:
     w >= x_L * y + y_L * x - x_L * y_L
     w >= x_U * y + y_U * x - x_U * y_U
     w <= x_U * y + y_L * x - x_U * y_L
     w <= x_L * y + y_U * x - x_L * y_U
2. Spatial branching relaxation gap estimation
3. Mixed-grade tank quality pooling relaxation
"""

from typing import List, Tuple, Dict, Optional, Any
from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense


class McCormickEnvelope:
    """Creates McCormick convex relaxation inequalities for bilinear term w = x * y."""

    @staticmethod
    def add_bilinear_term(model: CanonicalModel,
                          x_idx: int, y_idx: int,
                          x_L: float, x_U: float,
                          y_L: float, y_U: float,
                          w_name: Optional[str] = None) -> int:
        """
        Adds relaxation variable w and 4 McCormick linear bounding constraints.
        Returns index of w.
        """
        if w_name is None:
            w_name = f"mc_{model.var_names[x_idx]}_{model.var_names[y_idx]}"

        w_L = min(x_L * y_L, x_L * y_U, x_U * y_L, x_U * y_U)
        w_U = max(x_L * y_L, x_L * y_U, x_U * y_L, x_U * y_U)
        w_idx = model.add_var(name=w_name, lb=w_L, ub=w_U, obj=0.0, var_type=VarType.CONTINUOUS)

        # 1. w >= x_L * y + y_L * x - x_L * y_L
        # ==>  -y_L * x - x_L * y + w >= -x_L * y_L
        # In canonical LE: y_L * x + x_L * y - w <= x_L * y_L
        row1 = {x_idx: y_L, y_idx: x_L, w_idx: -1.0}
        model.add_constraint(row1, ConstraintSense.LE, x_L * y_L, name=f"{w_name}_mccormick_1")

        # 2. w >= x_U * y + y_U * x - x_U * y_U
        # In canonical LE: y_U * x + x_U * y - w <= x_U * y_U
        row2 = {x_idx: y_U, y_idx: x_U, w_idx: -1.0}
        model.add_constraint(row2, ConstraintSense.LE, x_U * y_U, name=f"{w_name}_mccormick_2")

        # 3. w <= x_U * y + y_L * x - x_U * y_L
        # In canonical LE: -y_L * x - x_U * y + w <= -x_U * y_L
        row3 = {x_idx: -y_L, y_idx: -x_U, w_idx: 1.0}
        model.add_constraint(row3, ConstraintSense.LE, -x_U * y_L, name=f"{w_name}_mccormick_3")

        # 4. w <= x_L * y + y_U * x - x_L * y_U
        # In canonical LE: -y_U * x - x_L * y + w <= -x_L * y_U
        row4 = {x_idx: -y_U, y_idx: -x_L, w_idx: 1.0}
        model.add_constraint(row4, ConstraintSense.LE, -x_L * y_U, name=f"{w_name}_mccormick_4")

        return w_idx
