"""The training sandbox: every piece of task knowledge this project defines.

Scenarios live under `sandbox.scenarios`, which `pyproject.toml` declares as
an `aero_gym.scenarios` entry point so `aero_gym.registry` finds them with no
configuration. Nothing in `aero_gym` or `aero_train` may import this package.
"""
