import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops

from pyagc.models.scgc import SCGC


class _IdentityEncoder(torch.nn.Module):
    def forward(self, x, edge_index, **kwargs):
        return x


class _NoOpTransform:
    def __call__(self, x, edge_index, **kwargs):
        return Data(x=x, edge_index=edge_index)


def test_scgc_loss_matches_pos_neg_means():
    num_nodes = 4
    x = torch.randn(num_nodes, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    model = SCGC(
        encoder=_IdentityEncoder(),
        transform1=_NoOpTransform(),
        transform2=_NoOpTransform(),
    )

    z1 = torch.randn(num_nodes, 3)
    z2 = torch.randn(num_nodes, 3)
    loss_sparse, loss_pos, loss_neg = model._compute_loss(z1, z2, edge_index)

    z1_n = torch.nn.functional.normalize(z1, p=2, dim=-1)
    z2_n = torch.nn.functional.normalize(z2, p=2, dim=-1)
    S = z1_n @ z2_n.T

    adj_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    a_hat = torch.zeros(num_nodes, num_nodes)
    a_hat[adj_edge_index[0], adj_edge_index[1]] = 1.0

    pos_mask = a_hat.bool()
    neg_mask = ~pos_mask
    loss_pos_dense = (S[pos_mask] - 1).pow(2).mean()
    loss_neg_dense = S[neg_mask].pow(2).mean()
    loss_dense = loss_pos_dense + loss_neg_dense

    assert torch.allclose(loss_sparse, loss_dense, atol=1e-5)
    assert torch.allclose(loss_pos, loss_pos_dense, atol=1e-5)
    assert torch.allclose(loss_neg, loss_neg_dense, atol=1e-5)
    assert torch.allclose(loss_sparse, loss_pos + loss_neg, atol=1e-5)
