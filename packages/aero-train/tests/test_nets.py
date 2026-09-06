from __future__ import annotations

import torch

from aero_train.nets import TargetNet, mlp


def test_mlp_maps_the_requested_dimensions():
    net = mlp(4, (8, 8), 2)
    assert net(torch.zeros(5, 4)).shape == (5, 2)


def test_mlp_has_no_activation_on_the_output():
    # Q-values and value estimates are unbounded; a final ReLU would clamp them at 0
    net = mlp(4, (8,), 2)
    assert isinstance(net[-1], torch.nn.Linear)


def test_mlp_respects_the_hidden_sizes():
    net = mlp(4, (16, 8), 2)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linears] == [16, 8, 2]


def test_target_net_starts_as_a_copy_but_is_a_separate_object():
    net = mlp(4, (8,), 2)
    tgt = TargetNet(net)
    assert tgt.target_model is not net
    x = torch.randn(3, 4)
    assert torch.equal(net(x), tgt.target_model(x))


def test_target_net_does_not_track_source_updates_until_synced():
    net = mlp(4, (8,), 2)
    tgt = TargetNet(net)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    x = torch.randn(3, 4)
    assert not torch.equal(net(x), tgt.target_model(x))
    tgt.sync()
    assert torch.equal(net(x), tgt.target_model(x))


def test_target_net_parameters_require_no_gradient():
    tgt = TargetNet(mlp(4, (8,), 2))
    assert not any(p.requires_grad for p in tgt.target_model.parameters())
