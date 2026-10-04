"""Settings for every test directory, applied before any test module loads."""

import os

# onnxruntime uploads telemetry from a thread that can abort the process at
# exit (libc++ "recursive_mutex lock failed"), and tests open no sockets.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
