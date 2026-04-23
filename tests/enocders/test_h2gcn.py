# test_h2gcn.py

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.typing import SparseTensor

# Modify this import to match your project structure.
from pyagc.encoders.h2gcn import H2GCNConv, H2GCN


def get_toy_graph():
    x = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
    ])

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ], dtype=torch.long)

    edge_weight = torch.tensor(
        [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 1.5, 1.5],
        dtype=torch.float,
    )

    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    return x, edge_index, edge_weight, y


def test_h2gcn_conv_output_shape():
    x, edge_index, _, _ = get_toy_graph()

    conv = H2GCNConv(cached=False, add_self_loops=False, normalize=True)
    out = conv(x, edge_index)

    assert out.shape == (x.size(0), x.size(1) * 2)


def test_h2gcn_conv_with_edge_weight():
    x, edge_index, edge_weight, _ = get_toy_graph()

    conv = H2GCNConv(cached=False, add_self_loops=False, normalize=True)
    out = conv(x, edge_index, edge_weight)

    assert out.shape == (x.size(0), x.size(1) * 2)
    assert torch.isfinite(out).all()


def test_h2gcn_conv_sparse_tensor_input():
    x, edge_index, edge_weight, _ = get_toy_graph()
    num_nodes = x.size(0)

    adj_t = SparseTensor(
        row=edge_index[1],
        col=edge_index[0],
        value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    conv = H2GCNConv(cached=False, add_self_loops=False, normalize=True)
    out = conv(x, adj_t)

    assert out.shape == (x.size(0), x.size(1) * 2)
    assert torch.isfinite(out).all()


def test_h2gcn_conv_caching_tensor_edge_index():
    x, edge_index, _, _ = get_toy_graph()

    conv = H2GCNConv(cached=True, add_self_loops=False, normalize=True)

    assert conv._cached_edge_index is None
    out1 = conv(x, edge_index)
    assert conv._cached_edge_index is not None

    cached_before = conv._cached_edge_index
    out2 = conv(x, edge_index)
    cached_after = conv._cached_edge_index

    assert cached_before is cached_after
    assert torch.allclose(out1, out2, atol=1e-6)


def test_h2gcn_conv_caching_sparse_tensor():
    x, edge_index, edge_weight, _ = get_toy_graph()
    num_nodes = x.size(0)

    adj_t = SparseTensor(
        row=edge_index[1],
        col=edge_index[0],
        value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    conv = H2GCNConv(cached=True, add_self_loops=False, normalize=True)

    assert conv._cached_adj_t is None
    out1 = conv(x, adj_t)
    assert conv._cached_adj_t is not None

    cached_before = conv._cached_adj_t
    out2 = conv(x, adj_t)
    cached_after = conv._cached_adj_t

    assert cached_before is cached_after
    assert torch.allclose(out1, out2, atol=1e-6)


def test_h2gcn_model_output_shape():
    x, edge_index, _, _ = get_toy_graph()

    model = H2GCN(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=2,
        dropout=0.0,
        cached=False,
        add_self_loops=False,
        normalize=True,
    )

    out = model(x, edge_index)
    assert out.shape == (x.size(0), 2)


def test_h2gcn_model_final_classifier_input_dim():
    model = H2GCN(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=3,
        dropout=0.0,
        cached=False,
        add_self_loops=False,
        normalize=True,
    )

    expected_dim = 8 * (2 ** (3 + 1) - 1)
    assert model.classifier.in_features == expected_dim


def test_h2gcn_forward_with_edge_weight():
    x, edge_index, edge_weight, _ = get_toy_graph()

    model = H2GCN(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=2,
        dropout=0.0,
        cached=False,
        add_self_loops=False,
        normalize=True,
    )

    out = model(x, edge_index, edge_weight)
    assert out.shape == (x.size(0), 2)
    assert torch.isfinite(out).all()


def test_h2gcn_backward():
    x, edge_index, _, y = get_toy_graph()

    model = H2GCN(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=2,
        dropout=0.0,
        cached=False,
        add_self_loops=False,
        normalize=True,
    )

    out = model(x, edge_index)
    loss = F.cross_entropy(out, y)
    loss.backward()

    has_grad = False
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            has_grad = True

    assert has_grad


def test_h2gcn_reset_parameters_changes_weights():
    model = H2GCN(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=2,
        dropout=0.0,
        cached=False,
        add_self_loops=False,
        normalize=True,
    )

    old_embed_weight = model.embed.weight.detach().clone()
    old_cls_weight = model.classifier.weight.detach().clone()

    model.reset_parameters()

    new_embed_weight = model.embed.weight.detach()
    new_cls_weight = model.classifier.weight.detach()

    assert not torch.allclose(old_embed_weight, new_embed_weight)
    assert not torch.allclose(old_cls_weight, new_cls_weight)


def test_h2gcn_invalid_num_layers():
    with pytest.raises(ValueError):
        H2GCN(
            in_channels=3,
            hidden_channels=8,
            out_channels=2,
            num_layers=0,
        )


def test_h2gcn_conv_rejects_tuple_input():
    _, edge_index, _, _ = get_toy_graph()
    conv = H2GCNConv()

    with pytest.raises(ValueError):
        conv((torch.randn(4, 3), torch.randn(4, 3)), edge_index)


def test_h2gcn_eval_mode_disables_dropout_randomness():
    x, edge_index, _, _ = get_toy_graph()

    model = H2GCN(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=2,
        dropout=0.8,
        cached=False,
        add_self_loops=False,
        normalize=True,
    )

    model.eval()
    out1 = model(x, edge_index)
    out2 = model(x, edge_index)

    assert torch.allclose(out1, out2, atol=1e-6)
