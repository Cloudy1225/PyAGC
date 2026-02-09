import torch
from torch_geometric.data import Data, HeteroData
from pyagc.transforms import RandomMaskFeat


def test_random_feat_mask_on_homogeneous_graph():
    torch.manual_seed(42)

    # Create dummy node and edge features with all ones
    x = torch.ones((5, 10))
    edge_attr = torch.ones((8, 6))

    data = Data(x=x, edge_attr=edge_attr)
    transform = RandomMaskFeat(p=0.5, node_attrs=["x"], edge_attrs=["edge_attr"])

    data = transform(data)

    # Check shapes preserved
    assert data.x.shape == (5, 10)
    assert data.edge_attr.shape == (8, 6)

    # Check that some values were masked
    assert (data.x == 0).any()
    assert (data.edge_attr == 0).any()


def test_random_feat_mask_on_heterogeneous_graph():
    torch.manual_seed(42)

    # Create dummy features for multiple node and edge types
    x = torch.ones((4, 6))
    edge_attr = torch.ones((3, 4))

    data = HeteroData()
    data['user'].x = x.clone()
    data['item'].x = x.clone()
    data['user', 'rates', 'item'].edge_attr = edge_attr.clone()
    data['item', 'rev_rates', 'user'].edge_attr = edge_attr.clone()

    transform = RandomMaskFeat(p=0.6, node_attrs=["x"], edge_attrs=["edge_attr"])
    data = transform(data)

    # Check shapes preserved
    assert data['user'].x.shape == (4, 6)
    assert data['item'].x.shape == (4, 6)
    assert data['user', 'rates', 'item'].edge_attr.shape == (3, 4)
    assert data['item', 'rev_rates', 'user'].edge_attr.shape == (3, 4)

    # Check that some values were masked
    assert (data['user'].x == 0).any()
    assert (data['item'].x == 0).any()
    assert (data['user', 'rates', 'item'].edge_attr == 0).any()
    assert (data['item', 'rev_rates', 'user'].edge_attr == 0).any()


def test_random_mask_feat_inplace_false():
    x = torch.ones((4, 6))
    data = Data(x=x)
    transform = RandomMaskFeat(p=1.0, node_attrs=["x"], inplace=False)

    out = transform(data)

    # Assert original tensor is unchanged
    assert (data.x == 1).all()

    # Output tensor should be fully zeroed
    assert (out.x == 0).all()


def test_random_mask_feat_inplace_true():
    x = torch.ones((4, 6))
    data = Data(x=x)
    transform = RandomMaskFeat(p=1.0, node_attrs=["x"], inplace=True)

    out = transform(data)

    # Assert original tensor is now fully zeroed (in-place modified)
    assert (data.x == 0).all()
    assert (out.x == 0).all()
