# /home/kostya/workspace/mpc/MPC/__init__.py

"""
Модуль MPC (Model Predictive Control).

Содержит линейный и нелинейный MPC контроллеры.
"""

from .unicycle_mpc_active_set_tracking import UnicycleMPC_ActiveSet_Tracking
from .unicycle_mpc_ipopt import UnicycleMPC
from .unicycle_mpc_osqp_tracking import UnicycleMPC_OSQP_Tracking

__all__ = [
    "UnicycleMPC",
    "UnicycleMPC_OSQP_Tracking",
    "UnicycleMPC_ActiveSet_Tracking",
]