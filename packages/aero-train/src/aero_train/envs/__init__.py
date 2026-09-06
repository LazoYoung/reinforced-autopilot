"""Environment construction — the only module in aero_train that may import aero_gym.

Every other module in this package talks to the Gymnasium API alone, which is
what lets the whole learning stack be tested against CartPole.
"""

from .factory import make_env_fn

__all__ = ["make_env_fn"]
