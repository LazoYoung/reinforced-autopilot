from __future__ import annotations

import numpy as np
import torch

from aero_train.algo.dqn import DQNAgent
from aero_train.nets import mlp
from aero_train.primitives import ArgmaxSelector, EpsilonGreedySelector


def make_agent(epsilon=0.0):
    torch.manual_seed(0)
    return DQNAgent(mlp(4, (8,), 3), EpsilonGreedySelector(epsilon, np.random.default_rng(0)))


def test_agent_maps_a_batch_of_obs_to_a_batch_of_actions():
    agent = make_agent()
    actions, state = agent(np.zeros((6, 4), np.float32))
    assert actions.shape == (6,)
    assert actions.dtype == np.int64
    assert state is None


def test_agent_actions_are_in_the_action_range():
    agent = make_agent(epsilon=1.0)
    actions, _ = agent(np.zeros((32, 4), np.float32))
    assert ((actions >= 0) & (actions < 3)).all()


def test_agent_threads_agent_state_through_unchanged():
    agent = make_agent()
    sentinel = object()
    _, state = agent(np.zeros((2, 4), np.float32), sentinel)
    assert state is sentinel


def test_sync_weights_replaces_the_network_weights():
    agent = make_agent()
    other = mlp(4, (8,), 3)
    with torch.no_grad():
        for p in other.parameters():
            p.fill_(0.5)
    payload = {k: v.numpy() for k, v in other.state_dict().items()}
    agent.sync_weights(payload)
    for name, param in agent.net.state_dict().items():
        assert torch.allclose(param, other.state_dict()[name])


def test_sync_weights_applies_selector_parameters_not_network_ones():
    agent = make_agent(epsilon=1.0)
    payload = {k: v.numpy() for k, v in agent.net.state_dict().items()}
    payload["selector.epsilon"] = np.float32(0.0)
    agent.sync_weights(payload)
    assert agent.selector.epsilon == 0.0


def test_sync_weights_rejects_an_unknown_selector_parameter():
    agent = make_agent()
    payload = {k: v.numpy() for k, v in agent.net.state_dict().items()}
    payload["selector.sigma"] = np.float32(0.3)
    try:
        agent.sync_weights(payload)
    except AttributeError as exc:
        assert "sigma" in str(exc)
    else:
        raise AssertionError("expected AttributeError for an unknown selector parameter")


def test_agent_does_not_build_a_graph_during_inference():
    agent = make_agent()
    actions, _ = agent(np.zeros((3, 4), np.float32))
    # would raise if inference had been run under grad and leaked a tensor
    assert isinstance(actions, np.ndarray)
    assert all(not p.grad_fn for p in agent.net.parameters() if hasattr(p, "grad_fn"))


def test_greedy_agent_is_deterministic():
    torch.manual_seed(0)
    agent = DQNAgent(mlp(4, (8,), 3), ArgmaxSelector())
    obs = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    assert agent(obs)[0].tolist() == agent(obs)[0].tolist()


def test_forward_pass_runs_with_grad_mode_disabled():
    """Reads torch.is_grad_enabled() from inside the forward pass, which does
    flip with the guard. Inspecting net.parameters() does not: they are leaf
    tensors with grad_fn=None whether or not torch.no_grad() is active, so such
    a test passes even with __call__'s no_grad guard dropped."""
    net = mlp(4, (8,), 3)
    grad_enabled_during_forward = []
    net.register_forward_hook(
        lambda module, inp, out: grad_enabled_during_forward.append(torch.is_grad_enabled())
    )
    agent = DQNAgent(net, ArgmaxSelector())
    agent(np.zeros((2, 4), np.float32))
    assert grad_enabled_during_forward == [False]
