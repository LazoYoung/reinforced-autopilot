"""The `autopilot` command: run, resume, eval."""

from __future__ import annotations

import argparse
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
    if resume_from is not None:
        algo.load_state_dict(resume_from["algo"])
        frames = int(resume_from["frames"])
        if resume_from.get("buffer") is not None and hasattr(algo, "buffer"):
            algo.buffer.load_state_dict(resume_from["buffer"])

    engine = build_engine(cfg.rollout, env_fn, algo.make_agent(), cfg.seed)
    tracker = build_tracker(cfg)
    checkpoints = CheckpointManager(cfg.checkpoint.dir, cfg.checkpoint)
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
    if args.flightgear:
        # eval is the only path that will touch FlightGear, and it is single-env
        # and local by construction -- one viewer, one aircraft. The flag is
        # accepted now and wired up in Phase 2, when EnvConfig gains a render
        # mode and aero_gym scenarios become loadable.
        raise NotImplementedError(
            "--flightgear needs the aero_gym env branch, which arrives in Phase 2"
        )
    env_fn = make_env_fn(cfg.env)
    obs_space, act_space = _spaces(env_fn)
    algo = build_algorithm(cfg, obs_space, act_space)
    algo.load_state_dict(payload["algo"])
    mean_return = evaluate_policy(algo, env_fn, args.episodes, seed=cfg.seed)
    print(f"{args.episodes} episodes, mean return {mean_return:.2f}")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
