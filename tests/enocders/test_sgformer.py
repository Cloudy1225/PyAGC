"""
Tests for the SGFormer model.

Covers:
  - Single graph (batch=None) node classification
  - Batched graphs node classification
  - Both aggregate modes: 'add' and 'cat'
  - Output shape and dtype
  - reset_parameters() runs without error
  - Forward is differentiable (backward pass)
"""

import torch
import pytest
from torch_geometric.data import Data, Batch
from pyagc.encoders.sgformer import SGFormer


def make_graph(num_nodes: int, num_edges: int, in_channels: int, seed: int = 0):
    """Create a random PyG Data object."""
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)


def make_model(in_channels: int, hidden: int, out: int, aggregate: str = "add") -> SGFormer:
    """Instantiate SGFormer with small dimensions for fast testing."""
    return SGFormer(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out,
        trans_num_layers=2,
        trans_num_heads=2,
        trans_dropout=0.0,  # disable dropout for deterministic tests
        gnn_num_layers=2,
        gnn_dropout=0.0,
        graph_weight=0.5,
        aggregate=aggregate,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

IN_CH, HIDDEN, OUT = 8, 16, 4


def assert_output(out: torch.Tensor, num_nodes: int):
    """Verify shape, dtype, and numerical validity of raw logits."""
    assert out.shape == (num_nodes, OUT), \
        f"Expected ({num_nodes}, {OUT}), got {out.shape}"
    assert out.dtype == torch.float32, \
        f"Expected float32, got {out.dtype}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Tests: single graph (batch=None)
# ---------------------------------------------------------------------------

class TestSingleGraph:
    """SGFormer forward with batch=None — all nodes belong to one graph."""

    def test_output_shape_add(self):
        data = make_graph(20, 40, IN_CH)
        model = make_model(IN_CH, HIDDEN, OUT, aggregate="add")
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, batch=None)
        assert_output(out, num_nodes=20)

    def test_output_shape_cat(self):
        data = make_graph(20, 40, IN_CH)
        model = make_model(IN_CH, HIDDEN, OUT, aggregate="cat")
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, batch=None)
        assert_output(out, num_nodes=20)

    def test_single_node_graph(self):
        """Edge case: graph with a single node and no edges."""
        data = make_graph(1, 0, IN_CH)
        model = make_model(IN_CH, HIDDEN, OUT)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, batch=None)
        assert_output(out, num_nodes=1)

    def test_backward(self):
        """Gradients must flow through the full model."""
        data = make_graph(15, 30, IN_CH)
        model = make_model(IN_CH, HIDDEN, OUT)
        model.train()
        out = model(data.x, data.edge_index, batch=None)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"


# ---------------------------------------------------------------------------
# Tests: batched graphs
# ---------------------------------------------------------------------------

class TestBatchedGraphs:
    """SGFormer forward with an explicit batch vector."""

    def _make_batch(self, num_graphs: int = 3):
        graphs = [make_graph(10 + i * 5, 20 + i * 5, IN_CH, seed=i)
                  for i in range(num_graphs)]
        return Batch.from_data_list(graphs)

    def test_output_shape_add(self):
        batch = self._make_batch()
        model = make_model(IN_CH, HIDDEN, OUT, aggregate="add")
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch=batch.batch)
        assert_output(out, num_nodes=batch.num_nodes)

    def test_output_shape_cat(self):
        batch = self._make_batch()
        model = make_model(IN_CH, HIDDEN, OUT, aggregate="cat")
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch=batch.batch)
        assert_output(out, num_nodes=batch.num_nodes)

    def test_node_order_preserved(self):
        """Output node order must match the input node order across two runs."""
        batch = self._make_batch(num_graphs=2)
        model = make_model(IN_CH, HIDDEN, OUT)
        model.eval()
        with torch.no_grad():
            out1 = model(batch.x, batch.edge_index, batch=batch.batch)
            out2 = model(batch.x, batch.edge_index, batch=batch.batch)
        assert torch.allclose(out1, out2), "Non-deterministic output detected"

    def test_backward(self):
        """Gradients must flow through the full model on batched input."""
        batch = self._make_batch()
        model = make_model(IN_CH, HIDDEN, OUT)
        model.train()
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"


# ---------------------------------------------------------------------------
# Tests: batch=None vs explicit zeros batch must produce the same result
# ---------------------------------------------------------------------------

class TestBatchNoneEquivalence:
    """
    batch=None should be equivalent to passing a zero-filled batch vector
    (all nodes assigned to graph 0), since SGModule handles None by delegating
    to to_dense_batch which treats it as a single graph.
    """

    def test_equivalence(self):
        data = make_graph(18, 36, IN_CH, seed=99)
        model = make_model(IN_CH, HIDDEN, OUT)
        model.eval()

        explicit_batch = torch.zeros(data.num_nodes, dtype=torch.long)

        with torch.no_grad():
            out_none = model(data.x, data.edge_index, batch=None)
            out_explicit = model(data.x, data.edge_index, batch=explicit_batch)

        assert torch.allclose(out_none, out_explicit, atol=1e-6), \
            "batch=None and batch=zeros produced different outputs"


# ---------------------------------------------------------------------------
# Tests: model API
# ---------------------------------------------------------------------------

class TestModelAPI:

    def test_invalid_aggregate_raises(self):
        with pytest.raises(ValueError, match="Invalid aggregate type"):
            make_model(IN_CH, HIDDEN, OUT, aggregate="mean")

    def test_reset_parameters(self):
        """reset_parameters must run without errors."""
        model = make_model(IN_CH, HIDDEN, OUT)
        model.reset_parameters()

    def test_out_channels_attribute(self):
        model = make_model(IN_CH, HIDDEN, OUT)
        assert model.out_channels == OUT

    def test_params1_params2_non_empty(self):
        """Both parameter groups used for separate optimizers must be non-empty."""
        model = make_model(IN_CH, HIDDEN, OUT)
        assert len(model.params1) > 0
        assert len(model.params2) > 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])