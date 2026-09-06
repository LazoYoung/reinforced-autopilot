from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aero_train.algo import build_algorithm
from aero_train.algo.base import Algorithm
from aero_train.algo.dqn import DQN, DQNAgent
from aero_train.config import DQNConfig, EnvConfig, TrainConfig
from aero_train.experience import to_transitions
from aero_train.primitives import Segment, Transitions

CFG = DQNConfig(
    hidden_sizes=(8,), lr=1e-3, gamma=0.5, n_step=1,
    buffer_size=200, warmup_frames=16, batch_size=8,
    target_sync_interval=2, updates_per_iter=3,
    epsilon_start=1.0, epsilon_final=0.1, epsilon_decay_frames=100,
)


def make_dqn(cfg=CFG):
    return DQN(obs_shape=(4,), n_actions=3, cfg=cfg, device="cpu", seed=0)


def segment(T=10, B=2):
    return Segment(
        obs=np.random.default_rng(0).normal(size=(T + 1, B, 4)).astype(np.float32),
        actions=np.zeros((T, B), np.int64),
        rewards=np.ones((T, B), np.float32),
        terminated=np.zeros((T, B), bool),
        truncated=np.zeros((T, B), bool),
        valid=np.ones((T, B), bool),
    )


def test_dqn_satisfies_the_algorithm_abc():
    assert isinstance(make_dqn(), Algorithm)


def test_update_returns_none_until_the_buffer_is_warm():
    algo = make_dqn()
    algo.observe(segment(T=4, B=2))          # 8 frames < warmup 16
    assert algo.update() is None
    assert algo.global_step == 0


def test_update_returns_metrics_once_warm():
    algo = make_dqn()
    algo.observe(segment(T=10, B=2))         # 20 frames
    metrics = algo.update()
    assert metrics is not None
    assert {"loss", "q_mean", "grad_norm", "epsilon"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())


def test_global_step_advances_by_updates_per_iter():
    algo = make_dqn()
    algo.observe(segment())
    algo.update()
    assert algo.global_step == CFG.updates_per_iter


def test_observe_counts_only_valid_frames():
    algo = make_dqn()
    seg = segment(T=10, B=2)
    seg.valid[0, 0] = False
    seg = Segment(seg.obs, seg.actions, seg.rewards, seg.terminated, seg.truncated, seg.valid)
    algo.observe(seg)
    assert algo.frames == 19


def test_epsilon_decays_linearly_and_floors_at_the_final_value():
    algo = make_dqn()
    assert algo.epsilon == pytest.approx(1.0)
    algo.observe(segment(T=25, B=2))         # 50 frames of 100
    assert algo.epsilon == pytest.approx(0.55)
    algo.observe(segment(T=100, B=2))        # well past the decay window
    assert algo.epsilon == pytest.approx(0.1)


def test_target_net_syncs_on_the_configured_interval(monkeypatch):
    # Counting calls to .sync() -- not just "did any target param change" --
    # is what distinguishes "syncs every target_sync_interval steps" from both
    # "syncs on every step" (which tracks the online net and stabilises
    # nothing, the canonical silent DQN failure) and "never syncs". With
    # updates_per_iter=3 and target_sync_interval=2, global_step hits 1, 2, 3
    # across one update() call, so .sync() must fire exactly once (at step 2).
    algo = make_dqn()
    algo.observe(segment())
    sync_calls = []
    original_sync = algo.target.sync
    monkeypatch.setattr(algo.target, "sync", lambda: (sync_calls.append(None), original_sync()))
    algo.update()
    assert len(sync_calls) == 1


def test_terminated_transitions_do_not_bootstrap():
    algo = make_dqn()
    with torch.no_grad():                    # make the target net output a constant 10.0
        for p in algo.target.target_model.parameters():
            p.zero_()
        list(algo.target.target_model.parameters())[-1].fill_(10.0)
    batch = Transitions(
        obs=np.zeros((2, 4), np.float32),
        actions=np.zeros(2, np.int64),
        returns=np.ones(2, np.float32),
        next_obs=np.zeros((2, 4), np.float32),
        terminated=np.array([True, False]),
        steps=np.ones(2, np.int32),
    )
    target = algo._bellman_target(batch)
    assert float(target[0]) == pytest.approx(1.0)              # terminal: no bootstrap
    assert float(target[1]) == pytest.approx(1.0 + 0.5 * 10.0)  # gamma ** 1 * 10


def test_bootstrap_discount_uses_the_actual_step_count():
    algo = make_dqn()
    with torch.no_grad():
        for p in algo.target.target_model.parameters():
            p.zero_()
        list(algo.target.target_model.parameters())[-1].fill_(10.0)
    batch = Transitions(
        obs=np.zeros((1, 4), np.float32),
        actions=np.zeros(1, np.int64),
        returns=np.ones(1, np.float32),
        next_obs=np.zeros((1, 4), np.float32),
        terminated=np.array([False]),
        steps=np.array([3], np.int32),
    )
    assert float(algo._bellman_target(batch)[0]) == pytest.approx(1.0 + 0.125 * 10.0)


def test_bootstrap_discount_over_a_real_n_step_segment():
    # test_bootstrap_discount_uses_the_actual_step_count above hand-builds a
    # batch with steps=3 while cfg.n_step=1 -- a combination to_transitions can
    # never actually emit. This one drives the real to_transitions ->
    # _bellman_target path with n_step=3: two envs, one terminates at t=2, the
    # other is truncated at t=3, and both autoreset rows are excluded, so most
    # of the resulting transitions carry steps < n_step. That is exactly where
    # a regression to a constant gamma**n_step silently mis-weights the
    # bootstrap, and exactly the case a hand-built single-transition batch
    # cannot exercise.
    T, B = 5, 2
    obs = np.zeros((T + 1, B, 4), np.float32)
    actions = np.zeros((T, B), np.int64)
    rewards = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], np.float32)
    terminated = np.zeros((T, B), bool)
    terminated[2, 0] = True                  # env 0 terminates at t=2
    truncated = np.zeros((T, B), bool)
    truncated[3, 1] = True                   # env 1 is truncated at t=3
    valid = np.ones((T, B), bool)
    valid[3, 0] = False                      # autoreset skip after env 0's terminal
    valid[4, 1] = False                      # autoreset skip after env 1's truncation
    seg = Segment(obs, actions, rewards, terminated, truncated, valid)
    tr = to_transitions(seg, gamma=0.5, n_step=3)

    # Hand-derived: env 0's valid rows are t=0,1,2,4 (t=3 is the autoreset
    # skip; the terminal at t=2 cuts rows 0 and 1 short but not row 2 itself,
    # and t=4 starts a fresh post-reset episode). Env 1's valid rows are
    # t=0,1,2,3 (t=4 is the autoreset skip; the truncation at t=3 cuts row 2
    # short but not the truncation row itself).
    assert tr.steps.tolist() == [3, 2, 1, 1, 3, 3, 2, 1]
    assert tr.terminated.tolist() == [True, True, True, False, False, False, False, False]
    assert tr.returns.tolist() == pytest.approx([2.75, 3.5, 3.0, 5.0, 2.75, 4.5, 5.0, 4.0])

    cfg = DQNConfig(
        hidden_sizes=(8,), lr=1e-3, gamma=0.5, n_step=3,
        buffer_size=200, warmup_frames=16, batch_size=8,
        target_sync_interval=2, updates_per_iter=3,
        epsilon_start=1.0, epsilon_final=0.1, epsilon_decay_frames=100,
    )
    algo = make_dqn(cfg)
    with torch.no_grad():                    # make the target net output a constant 10.0
        for p in algo.target.target_model.parameters():
            p.zero_()
        list(algo.target.target_model.parameters())[-1].fill_(10.0)
    target = algo._bellman_target(tr)
    # A constant gamma**n_step (0.125) would only coincide with gamma**steps on
    # the steps==3 rows (indices 0, 4, 5); every steps<3 row here (1, 2, 3, 6, 7)
    # would come out wrong.
    assert target.tolist() == pytest.approx([2.75, 3.5, 3.0, 10.0, 4.0, 5.75, 7.5, 9.0])


def test_bellman_target_does_not_require_grad():
    # This is a regression guard for TargetNet's parameter freeze, not itself
    # proof of "no gradient through the target": with that freeze in place,
    # this stays green whether or not _bellman_target's torch.no_grad() block
    # is present at all, since torch never builds a graph through parameters
    # that have requires_grad=False. It only goes red if a future change
    # touches both the freeze and this no_grad() block together (see the fix
    # round 1 report for the mutation that demonstrates this).
    algo = make_dqn()
    batch = Transitions(
        obs=np.zeros((2, 4), np.float32),
        actions=np.zeros(2, np.int64),
        returns=np.ones(2, np.float32),
        next_obs=np.zeros((2, 4), np.float32),
        terminated=np.array([False, False]),
        steps=np.ones(2, np.int32),
    )
    target = algo._bellman_target(batch)
    assert not target.requires_grad


def test_actor_state_dict_carries_weights_and_epsilon_only():
    # An exact key set, not a blacklist: a blacklist only catches a leaked key
    # if it happens to start with "optimizer" or "target"; a leaked
    # optimizer-internal key like "adam_exp_avg" would sail through it. Any new
    # key -- leaked or legitimate -- must fail here until this assertion is
    # consciously updated, which is the point.
    algo = make_dqn()
    sd = algo.actor_state_dict()
    assert set(sd.keys()) == {"0.weight", "0.bias", "2.weight", "2.bias", "selector.epsilon"}
    assert all(isinstance(v, np.ndarray) for v in sd.values())


def test_the_architecture_is_named_in_exactly_one_place(monkeypatch):
    # The brief's own assertions here (kept below) only check that _build_net
    # is *callable* and reproducible -- they stay green even if __init__ or
    # make_agent stop calling it and build the architecture some other way.
    # Spying on _build_net itself is what pins "exactly one place": __init__
    # and make_agent must each route through it exactly once.
    calls: list[torch.nn.Module] = []
    original = DQN._build_net

    def spy(self):
        net = original(self)
        calls.append(net)
        return net

    monkeypatch.setattr(DQN, "_build_net", spy)
    algo = make_dqn()
    assert len(calls) == 1
    agent = algo.make_agent()
    assert len(calls) == 2
    assert agent.net is calls[-1]

    # a Dueling DQN subclass overrides _build_net and nothing else
    assert algo._build_net() is not algo.net
    assert [p.shape for p in algo._build_net().parameters()] == [
        p.shape for p in algo.net.parameters()
    ]


def test_make_agent_returns_an_independent_replica():
    algo = make_dqn()
    agent = algo.make_agent()
    assert isinstance(agent, DQNAgent)
    assert agent.net is not algo.net
    obs = np.zeros((2, 4), np.float32)
    with torch.no_grad():
        assert torch.allclose(agent.net(torch.as_tensor(obs)), algo.net(torch.as_tensor(obs)))


def test_the_agent_accepts_the_learners_actor_state_dict():
    algo = make_dqn()
    agent = algo.make_agent()
    algo.observe(segment())
    algo.update()
    agent.sync_weights(algo.actor_state_dict())
    assert agent.selector.epsilon == pytest.approx(algo.epsilon)


def test_state_dict_round_trip_restores_progress_and_weights():
    algo = make_dqn()
    algo.observe(segment())
    algo.update()
    restored = make_dqn()
    restored.load_state_dict(algo.state_dict())
    assert restored.global_step == algo.global_step
    assert restored.frames == algo.frames
    for a, b in zip(restored.net.parameters(), algo.net.parameters()):
        assert torch.equal(a, b)


def test_state_dict_restores_the_replay_sampler_stream():
    # without this a resumed run samples a different replay sequence and diverges.
    # algo samples once before checkpointing so its rng stream has already moved
    # past the shared fresh-seed state both algos start from -- otherwise this
    # test would pass even with the restore deleted, since make_dqn() always
    # seeds the buffer rng with the same value.
    algo = make_dqn()
    algo.observe(segment())
    algo.buffer.sample(4)
    restored = make_dqn()
    restored.observe(segment())
    restored.load_state_dict(algo.state_dict())
    assert restored.buffer.sample(4).obs.tolist() == algo.buffer.sample(4).obs.tolist()


def test_state_dict_snapshot_does_not_move_when_training_continues():
    # Same class of bug fixed one commit earlier in ReplayBuffer.state_dict:
    # nn.Module.state_dict() detaches each tensor but does not clone it, so it
    # aliases the same storage as the live parameter. A checkpoint dict taken
    # now must not silently change value as training keeps going.
    algo = make_dqn()
    algo.observe(segment())
    algo.update()
    sd = algo.state_dict()
    net_before = {k: v.clone() for k, v in sd["net"].items()}
    target_before = {k: v.clone() for k, v in sd["target"].items()}
    algo.observe(segment())
    algo.update()
    for k in net_before:
        assert torch.equal(net_before[k], sd["net"][k])
    for k in target_before:
        assert torch.equal(target_before[k], sd["target"][k])


def test_build_algorithm_rejects_dqn_against_a_continuous_action_space():
    cfg = TrainConfig(run_name="t", env=EnvConfig(id="Pendulum-v1"), algo=CFG)
    with pytest.raises(ValueError, match="discrete"):
        build_algorithm(cfg, gym.spaces.Box(-1, 1, (3,)), gym.spaces.Box(-1, 1, (1,)))


def test_build_algorithm_builds_dqn_for_a_discrete_action_space():
    cfg = TrainConfig(run_name="t", env=EnvConfig(id="CartPole-v1"), algo=CFG)
    algo = build_algorithm(cfg, gym.spaces.Box(-1, 1, (4,)), gym.spaces.Discrete(2))
    assert isinstance(algo, DQN)
