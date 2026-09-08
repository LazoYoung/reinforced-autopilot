"""The `autopilot` command: run, resume, eval."""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
from pathlib import Path

import gymnasium as gym

from .algo import build_algorithm
from .checkpoint import CheckpointManager, load_checkpoint
from .config import TrainConfig, config_from_dict, load_config
from .envs import make_env_fn
from .rollout import build_engine
from .tracking import build_tracker
from .trainer import Trainer, evaluate_policy


def _spaces(env_fn) -> tuple[gym.Space, gym.Space]:
    """Probe one throwaway env for its spaces before building the algorithm."""
    probe = env_fn()
    try:
        return probe.observation_space, probe.action_space
    finally:
        probe.close()


def _train(cfg: TrainConfig, resume_from: dict | None = None) -> int:
    env_fn = make_env_fn(cfg.env)
    obs_space, act_space = _spaces(env_fn)
    algo = build_algorithm(cfg, obs_space, act_space)
    frames = 0
    best_return = float("-inf")
    if resume_from is not None:
        algo.load_state_dict(resume_from["algo"])
        frames = int(resume_from["frames"])
        # .get(...), not [...]: a checkpoint written before this field existed
        # should still resume, just back at -inf like today's behavior, rather
        # than KeyError on an old file
        best_return = float(resume_from.get("best_return", float("-inf")))
        if resume_from.get("buffer") is not None and hasattr(algo, "buffer"):
            algo.buffer.load_state_dict(resume_from["buffer"])

    engine = build_engine(cfg.rollout, env_fn, algo.make_agent(), cfg.seed)
    tracker = build_tracker(cfg)
    checkpoints = CheckpointManager(cfg.checkpoint.dir, cfg.checkpoint, best_return=best_return)
    trainer = Trainer(cfg, algo, engine, tracker, checkpoints, eval_env_fn=env_fn)
    trainer.frames = frames
    trainer.run()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return _train(load_config(args.config, args.override))


def cmd_resume(args: argparse.Namespace) -> int:
    payload = load_checkpoint(Path(args.checkpoint))
    data = dict(payload["config"])
    if args.total_steps is not None:
        # extend a run past its original budget without editing the checkpoint
        data["total_steps"] = args.total_steps
    return _train(config_from_dict(data), resume_from=payload)


def cmd_eval(args: argparse.Namespace) -> int:
    payload = load_checkpoint(Path(args.checkpoint))
    cfg = config_from_dict(payload["config"])
    env_cfg = cfg.env
    if args.flightgear:
        if env_cfg.scenario is None:
            raise ValueError(
                "--flightgear needs an aero_gym scenario env; this checkpoint was "
                f"trained on Gymnasium env {env_cfg.id!r}, which has no viewer"
            )
        # eval is the only path that touches FlightGear, and it is single-env and
        # local by construction -- one viewer, one aircraft.
        env_cfg = dataclasses.replace(env_cfg, render_mode="flightgear")
    env_fn = make_env_fn(env_cfg)
    obs_space, act_space = _spaces(env_fn)
    algo = build_algorithm(cfg, obs_space, act_space)
    algo.load_state_dict(payload["algo"])
    mean_return = evaluate_policy(algo, env_fn, args.episodes, seed=cfg.seed)
    print(f"{args.episodes} episodes, mean return {mean_return:.2f}")
    return 0


def build_mlflow_server_argv(store: str, host: str, port: int) -> list[str]:
    """Build the argv for `uvx --from mlflow mlflow server ...`.

    Pure -- no I/O, no env mutation -- so it can be tested without spawning
    anything. A `store` containing "://" is already a URI (a sqlite file, a
    real tracking server) and is passed through untouched; otherwise it is a
    filesystem path and needs the `file:` scheme MLflow expects.
    """
    uri = store if "://" in store else f"file:{store}"
    return [
        "uvx",
        "--from",
        "mlflow",
        "mlflow",
        "server",
        "--backend-store-uri",
        uri,
        "--host",
        host,
        "--port",
        str(port),
    ]


def cmd_mlflow(args: argparse.Namespace) -> int:
    # setdefault, not assignment: an explicit value the caller already set
    # (e.g. a shared-host MLFLOW_ALLOW_FILE_STORE policy) must win over this
    # default. MLflowTracker makes the identical tradeoff, for the same
    # reason -- see its constructor.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    argv = build_mlflow_server_argv(args.store, args.host, args.port)

    uvx = shutil.which("uvx")
    if uvx is None:
        manual = " ".join(argv)
        print(
            "error: `uvx` not found on PATH -- it ships with uv "
            "(https://docs.astral.sh/uv/). Run the server manually instead:\n"
            f"  MLFLOW_ALLOW_FILE_STORE=true {manual}",
            file=sys.stderr,
        )
        return 1

    # exec, not subprocess: this is a foreground server the user Ctrl-C's out
    # of, and exec gives that the obvious behavior with no wrapper process in
    # the middle. It never returns on success -- the `return 0` below only
    # runs under a test's monkeypatched os.execvp.
    os.execvp(uvx, argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autopilot")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train from a YAML run config")
    run.add_argument("config")
    run.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="dotted override, e.g. rollout.n_workers=4; applied before validation",
    )
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="continue training from a checkpoint")
    resume.add_argument("checkpoint")
    resume.add_argument("--total-steps", type=int, default=None)
    resume.set_defaults(func=cmd_resume)

    ev = sub.add_parser("eval", help="run a checkpointed policy greedily")
    ev.add_argument("checkpoint")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--flightgear", action="store_true")
    ev.set_defaults(func=cmd_eval)

    ml = sub.add_parser("mlflow", help="launch `mlflow server` against the local run store")
    ml.add_argument("--port", type=int, default=5000)
    ml.add_argument("--host", default="127.0.0.1")
    ml.add_argument("--store", default="./mlruns", help="path or URI for --backend-store-uri")
    ml.set_defaults(func=cmd_mlflow)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
