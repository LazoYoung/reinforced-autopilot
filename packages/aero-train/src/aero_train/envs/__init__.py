"""Environment construction: the only module in aero_train that may import aero_gym.

Every other module in this package talks to the Gymnasium API alone, which is
what lets the whole learning stack be tested against CartPole.
"""

from .factory import make_env_fn

__all__ = ["make_env_fn", "observation_saturation_report"]


def observation_saturation_report() -> str:
    """What the run's observations did outside their declared bounds, or "".

    Lives here because it reaches into aero_gym, and the import stays inside
    the function so a Gymnasium-only run never pays for it. Empty for those
    runs, which have no aero_gym features to exceed anything.
    """
    from aero_gym.saturation import format_saturation_report, saturation_report

    return format_saturation_report(saturation_report())
