import torch
from torch_geometric.data import Data, HeteroData
from pyagc.transforms import RandomDropEdge


def test_random_drop_edge_on_homogeneous_graph():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
    edge_attr = torch.ones((4, 5))  # 4 edges, 5-dimensional attr
    data = Data(edge_index=edge_index, edge_attr=edge_attr)

    transform = RandomDropEdge(p=1.0)  # Drop all edges
    out = transform(data.clone())
    assert out.edge_index.size(1) == 0
    assert out.edge_attr.size(0) == 0

    transform = RandomDropEdge(p=0.0)  # Drop none
    out = transform(data.clone())
    assert torch.equal(out.edge_index, edge_index)
    assert torch.equal(out.edge_attr, edge_attr)


def test_random_drop_edge_on_heterogeneous_graph():
    data = HeteroData()

    data['user', 'follows', 'user'].edge_index = torch.tensor([[0, 1], [1, 2]])
    data['user', 'follows', 'user'].edge_attr = torch.ones((2, 4))

    data['author', 'writes', 'paper'].edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]])
    data['author', 'writes', 'paper'].edge_attr = torch.ones((3, 4))

    transform = RandomDropEdge(p=1.0)
    out = transform(data)
    assert out['user', 'follows', 'user'].edge_index.size(1) == 0
    assert out['user', 'follows', 'user'].edge_attr.size(0) == 0
    assert out['author', 'writes', 'paper'].edge_index.size(1) == 0
    assert out['author', 'writes', 'paper'].edge_attr.size(0) == 0


def test_random_drop_edge_inplace_false():
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.ones((2, 3))
    data = Data(edge_index=edge_index, edge_attr=edge_attr)

    transform = RandomDropEdge(p=1.0, inplace=False)
    out = transform(data)

    assert data.edge_index.size(1) == 2  # Original untouched
    assert out.edge_index.size(1) == 0  # All dropped


def test_random_drop_edge_inplace_true():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.ones((3, 2))  # 3 edges, 2-dim features
    data = Data(edge_index=edge_index.clone(), edge_attr=edge_attr.clone())

    transform = RandomDropEdge(p=1.0, inplace=True)  # Drop all
    out = transform(data)

    # Check: edge_index and edge_attr have been updated
    assert out.edge_index.size(1) == 0
    assert out.edge_attr.size(0) == 0
