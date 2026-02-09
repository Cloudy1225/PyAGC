import torch
from torch_geometric.data import Data, HeteroData
from pyagc.transforms import GSSLTransform


def test_gssl_transform_on_homogeneous_graph():
    # Test on homogeneous graph (Data)
    x = torch.randn(5, 10)
    y = torch.randn(5, 4)  # Another node attribute
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_attr = torch.randn(edge_index.size(1), 3)
    edge_weight = torch.randn(edge_index.size(1))

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

    transform = GSSLTransform(
        p_feat_mask=0.5,
        p_edge_drop=0.5,
        node_attrs=["x", "y"],
        edge_attrs=["edge_attr", "edge_weight"]
    )
    out = transform(data)

    # Check node attributes
    for attr in ["x", "y"]:
        assert hasattr(out, attr), f"Missing node attribute {attr}"
        assert out[attr].size() == getattr(data, attr).size(), f"Size mismatch for node attribute {attr}"

    # Check edge index
    assert hasattr(out, 'edge_index')
    assert out.edge_index.size(0) == 2
    assert out.edge_index.size(1) <= edge_index.size(1)

    # Check edge attributes
    for attr in ["edge_attr", "edge_weight"]:
        assert hasattr(out, attr), f"Missing edge attribute {attr}"
        assert out[attr].size(0) == out.edge_index.size(1), f"Size mismatch for edge attribute {attr}"


def test_gssl_transform_on_heterogeneous_graph():
    # Test on heterogeneous graph (HeteroData)
    hetero_data = HeteroData()
    hetero_data['paper'].x = torch.randn(6, 16)
    hetero_data['paper'].y = torch.randn(6, 8)
    hetero_data['author'].x = torch.randn(3, 8)
    hetero_data['author'].y = torch.randn(3, 4)

    hetero_data[('author', 'writes', 'paper')].edge_index = torch.tensor(
        [[0, 1, 2], [2, 3, 4]]
    )
    hetero_data[('author', 'writes', 'paper')].edge_attr = torch.randn(3, 5)
    hetero_data[('author', 'writes', 'paper')].edge_weight = torch.randn(3)

    transform = GSSLTransform(
        p_feat_mask=0.5,
        p_edge_drop=0.5,
        node_attrs=["x", "y"],
        edge_attrs=["edge_attr", "edge_weight"]
    )

    out_hetero = transform(hetero_data)

    # Check node attributes for each node type
    for node_type in ['paper', 'author']:
        for attr in ["x", "y"]:
            assert attr in out_hetero[node_type], f"Missing node attribute {attr} for node type {node_type}"
            assert out_hetero[node_type][attr].size() == hetero_data[node_type][attr].size(), f"Size mismatch for node attribute {attr} in {node_type}"

    # Check edge attributes for each edge type
    for edge_type in [('author', 'writes', 'paper')]:
        assert 'edge_index' in out_hetero[edge_type]
        assert out_hetero[edge_type].edge_index.size(0) == 2
        assert out_hetero[edge_type].edge_index.size(1) <= hetero_data[edge_type].edge_index.size(1)

        for attr in ["edge_attr", "edge_weight"]:
            assert attr in out_hetero[edge_type], f"Missing edge attribute {attr} for edge type {edge_type}"
            assert out_hetero[edge_type][attr].size(0) == out_hetero[edge_type].edge_index.size(1), f"Size mismatch for edge attribute {attr} in {edge_type}"
