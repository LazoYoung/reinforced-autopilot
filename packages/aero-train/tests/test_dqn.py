from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aero_train.algo import build_algorithm
from aero_train.algo.base import Algorithm
from aero_train.algo.dqn import DQN, DQNAgent
from aero_train.config import DQNConfig, EnvConfig, TrainConfig
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


def test_target_net_syncs_on_the_configured_interval():
    algo = make_dqn()
    algo.observe(segment())
    before = [p.clone() for p in algo.target.target_model.parameters()]
    algo.update()                            # 3 updates, interval 2 -> at least one sync
    after = list(algo.target.target_model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


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


def test_bellman_target_does_not_require_grad():
    # Supplements the brief's 17: neither trap-1 nor trap-3 test above pins
    # "no gradient through the target" (trap 2 in the brief's correctness-trap
    # list) on its own, since TargetNet already freezes target_model's params
    # (requires_grad=False), which alone keeps this tensor grad-free regardless
    # of whether _bellman_target's torch.no_grad() block is present. This test
    # exercises the torch.no_grad() block as the second line of defense: it is
    # not redundant if TargetNet's own freeze is ever weakened (e.g. a future
    # refactor that shares parameters instead of a frozen deep copy).
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
    algo = make_dqn()
    sd = algo.actor_state_dict()
    assert "selector.epsilon" in sd
    assert all(isinstance(v, np.ndarray) for v in sd.values())
    # no optimizer or target-net state crosses to a worker
    assert not any(k.startswith(("optimizer", "target")) for k in sd)


def test_the_architecture_is_named_in_exactly_one_place():
    # a Dueling DQN subclass overrides _build_net and nothing else
    algo = make_dqn()
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


def test_build_algorithm_rejects_dqn_against_a_continuous_action_space():
    cfg = TrainConfig(run_name="t", env=EnvConfig(id="Pendulum-v1"), algo=CFG)
    with pytest.raises(ValueError, match="discrete"):
        build_algorithm(cfg, gym.spaces.Box(-1, 1, (3,)), gym.spaces.Box(-1, 1, (1,)))


def test_build_algorithm_builds_dqn_for_a_discrete_action_space():
    cfg = TrainConfig(run_name="t", env=EnvConfig(id="CartPole-v1"), algo=CFG)
    algo = build_algorithm(cfg, gym.spaces.Box(-1, 1, (4,)), gym.spaces.Discrete(2))
    assert isinstance(algo, DQN)
