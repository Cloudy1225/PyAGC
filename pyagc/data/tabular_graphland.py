import os
import os.path as osp
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import torch
import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import subgraph


def _load_yaml(path: str) -> dict:
    import yaml  # type: ignore
    with open(path) as f:
        return yaml.safe_load(f)


class GraphLandTensorFrameDataset(InMemoryDataset):
    r"""GraphLand dataset rewritten to store node attributes in TensorFrame.

    Differences from the original implementation:
    - Graph structure is stored in `Data.edge_index`.
    - Node attributes are stored in `Data.x` (a `torch_frame.TensorFrame`).
    - Masks and targets are still stored in `Data`.

    Notes:
    - The original sklearn-based feature preprocessing is intentionally removed.
      In a torch-frame workflow, semantic types are preserved and feature
      encoding/normalization/imputation is usually handled by the model-side
      encoders.
    """

    _url = 'https://zenodo.org/records/16895532'

    GRAPHLAND_DATASETS = {
        'hm-categories': 'multiclass_classification',
        'pokec-regions': 'multiclass_classification',
        'web-topics': 'multiclass_classification',
        'tolokers-2': 'binary_classification',
        'city-reviews': 'binary_classification',
        'artnet-exp': 'binary_classification',
        'web-fraud': 'binary_classification',
        'hm-prices': 'regression',
        'avazu-ctr': 'regression',
        'city-roads-M': 'regression',
        'city-roads-L': 'regression',
        'twitch-views': 'regression',
        'artnet-views': 'regression',
        'web-traffic': 'regression',
    }

    def __init__(
        self,
        root: str,
        name: str,
        split: str,
        to_undirected: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in self.GRAPHLAND_DATASETS, f'Unsupported dataset name: {name}'
        assert split in ['RL', 'RH', 'TH', 'THI'], f'Unsupported split name: {split}'

        if split in ['TH', 'THI']:
            assert name not in [
                'city-reviews',
                'city-roads-M',
                'city-roads-L',
                'web-traffic',
            ], ('Temporal split is not available for city-reviews, '
                'city-roads-M, city-roads-L, web-traffic.')

        self.name = name
        self.split = split
        self.task = self.GRAPHLAND_DATASETS[name]
        self._to_undirected = to_undirected

        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(
            self.root,
            self.name,
            'processed',
            f'{self.split}__to_undirected_{str(self._to_undirected).lower()}__tensorframe',
        )

    @property
    def raw_file_names(self) -> str:
        return self.name

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        zip_url = f"{self._url}/files/{self.name}.zip"
        path = download_url(zip_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def _get_raw_data(self) -> dict[str, Any]:
        raw_data_dir = osp.join(self.raw_dir, self.name)
        info = _load_yaml(osp.join(raw_data_dir, 'info.yaml'))

        features_df = pd.read_csv(
            osp.join(raw_data_dir, 'features.csv'),
            index_col=0,
        )

        targets_df = pd.read_csv(
            osp.join(raw_data_dir, 'targets.csv'),
            index_col=0,
        )
        targets = targets_df[info['target_name']].values

        masks_df = pd.read_csv(
            osp.join(raw_data_dir, f'split_masks_{self.split[:2]}.csv'),
            index_col=0,
        )
        masks = {
            k: np.array(v, dtype=bool)
            for k, v in masks_df.to_dict('list').items()
        }

        edges_df = pd.read_csv(osp.join(raw_data_dir, 'edgelist.csv'))
        edges = edges_df.values

        return {
            'info': info,
            'features_df': features_df,
            'targets': targets,
            'masks': masks,
            'edges': edges,
        }

    def _build_col_to_stype(self, info: dict) -> dict[str, torch_frame.stype]:
        col_to_stype: dict[str, torch_frame.stype] = {}

        num_cols = list(info['fraction_features_names']) + list(info['numerical_features_names'])
        cat_cols = list(info['categorical_features_names'])

        for col in num_cols:
            col_to_stype[col] = stype.numerical
        for col in cat_cols:
            col_to_stype[col] = stype.categorical

        return col_to_stype

    def _default_stats_for_stype(
            self,
            st: torch_frame.stype,
    ) -> dict[StatType, Any]:
        if st == stype.numerical:
            return {
                StatType.MEAN: np.nan,
                StatType.STD: np.nan,
                StatType.QUANTILES: [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        if st == stype.categorical:
            return {
                StatType.COUNT: ([], []),
            }
        raise ValueError(f'Unsupported stype: {st}')

    def _compute_col_stats(
        self,
        df: pd.DataFrame,
        col_to_stype: dict[str, torch_frame.stype],
    ) -> dict[str, dict[StatType, Any]]:
        """Compute column statistics in a batched manner.

        Numerical columns are processed together using vectorized NumPy
        operations. Categorical columns are still processed per column, but
        grouped under the same pass to reduce overhead.
        """
        col_stats: dict[str, dict[StatType, Any]] = {}

        num_cols = [col for col, st in col_to_stype.items() if st == stype.numerical]
        cat_cols = [col for col, st in col_to_stype.items() if st == stype.categorical]

        # Batch-compute statistics for numerical columns.
        if len(num_cols) > 0:
            num_df = df[num_cols].copy()

            for col in num_cols:
                num_df[col] = num_df[col].mask(num_df[col].isin([np.inf, -np.inf]), np.nan)
                if not ptypes.is_numeric_dtype(num_df[col]):
                    raise TypeError(
                        f"Numerical series '{col}' contains invalid entries. "
                        "Please make sure it contains only numerical values or NaNs."
                    )

            arr = num_df.to_numpy(dtype=np.float64, copy=True)
            finite_mask = np.isfinite(arr)
            arr[~finite_mask] = np.nan

            all_nan_mask = np.isnan(arr).all(axis=0)

            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            quants = np.nanquantile(
                arr,
                q=[0.0, 0.25, 0.5, 0.75, 1.0],
                axis=0,
            )

            for idx, col in enumerate(num_cols):
                if all_nan_mask[idx]:
                    col_stats[col] = self._default_stats_for_stype(stype.numerical)
                else:
                    col_stats[col] = {
                        StatType.MEAN: float(means[idx]),
                        StatType.STD: float(stds[idx]),
                        StatType.QUANTILES: quants[:, idx].tolist(),
                    }

        # Compute statistics for categorical columns.
        if len(cat_cols) > 0:
            cat_df = df[cat_cols]

            for col in cat_cols:
                ser = cat_df[col]
                if ser.isnull().all():
                    col_stats[col] = self._default_stats_for_stype(stype.categorical)
                    continue

                count = ser.dropna().value_counts(ascending=False)
                col_stats[col] = {
                    StatType.COUNT: (count.index.tolist(), count.values.tolist())
                }

        return col_stats

    def _encode_categorical_column(
            self,
            ser: pd.Series,
            stats: dict[StatType, Any],
    ) -> torch.Tensor:
        """Encode a categorical column into integer indices.

        Missing values are mapped to -1 to match TensorFrame conventions.
        Unknown categories are also mapped to -1.
        """
        categories, _ = stats[StatType.COUNT]
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        values = ser.to_numpy(copy=False)
        out = np.full(len(values), -1, dtype=np.int64)

        for i, value in enumerate(values):
            if pd.isna(value):
                continue
            out[i] = cat_to_idx.get(value, -1)

        return torch.from_numpy(out)

    def _build_tensor_frame(
            self,
            df: pd.DataFrame,
            col_to_stype: dict[str, torch_frame.stype],
            col_stats: dict[str, dict[StatType, Any]],
            y: Optional[torch.Tensor] = None,
    ) -> TensorFrame:
        """Build a TensorFrame directly from a pandas DataFrame."""
        feat_dict: dict[torch_frame.stype, torch.Tensor] = {}
        col_names_dict: dict[torch_frame.stype, list[str]] = {}

        num_cols = [col for col, st in col_to_stype.items() if st == stype.numerical]
        cat_cols = [col for col, st in col_to_stype.items() if st == stype.categorical]

        if len(num_cols) > 0:
            num_arr = df[num_cols].to_numpy(dtype=np.float32, copy=True)
            num_arr[np.isinf(num_arr)] = np.nan
            feat_dict[stype.numerical] = torch.from_numpy(num_arr)
            col_names_dict[stype.numerical] = num_cols

        if len(cat_cols) > 0:
            cat_tensors = []
            for col in cat_cols:
                cat_tensors.append(
                    self._encode_categorical_column(df[col], col_stats[col]).view(-1, 1)
                )
            feat_dict[stype.categorical] = torch.cat(cat_tensors, dim=1)
            col_names_dict[stype.categorical] = cat_cols

        return TensorFrame(
            feat_dict=feat_dict,
            col_names_dict=col_names_dict,
            y=y,
            num_rows=len(df),
        )

    def _prepare_targets(
            self,
            raw_data: dict[str, Any],
    ) -> tuple[torch.Tensor, np.ndarray]:
        targets = raw_data['targets']
        labeled_mask = ~pd.isna(targets)

        if self.task == 'regression':
            y_np = np.asarray(targets, dtype=np.float32)
            y = torch.from_numpy(y_np)
        else:
            y_np = np.asarray(targets, dtype=np.float32)
            # y_np[~labeled_mask] = -1
            # y = torch.from_numpy(y_np.astype(np.int64))
            y = torch.from_numpy(y_np)

        return y, labeled_mask

    def _get_transductive_data(self) -> list[Data]:
        raw_data = self._get_raw_data()
        info = raw_data['info']
        features_df = raw_data['features_df']
        masks = raw_data['masks']

        col_to_stype = self._build_col_to_stype(info)
        y, labeled_mask = self._prepare_targets(raw_data)

        # Compute column statistics only on training nodes to avoid leakage.
        train_df = features_df.loc[masks['train']]
        train_col_stats = self._compute_col_stats(train_df, col_to_stype)

        tf = self._build_tensor_frame(
            df=features_df,
            col_to_stype=col_to_stype,
            col_stats=train_col_stats,
            y=None,
        )

        train_mask = torch.from_numpy(masks['train'] & labeled_mask).bool()
        val_mask = torch.from_numpy(masks['val'] & labeled_mask).bool()
        test_mask = torch.from_numpy(masks['test'] & labeled_mask).bool()

        edge_index = torch.from_numpy(raw_data['edges'].T).long()

        data = Data(
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            x=tf,
            tf_col_stats=train_col_stats,
        )
        return [data]

    def _get_inductive_data(self) -> list[Data]:
        raw_data = self._get_raw_data()
        info = raw_data['info']
        features_df = raw_data['features_df']
        masks = raw_data['masks']

        col_to_stype = self._build_col_to_stype(info)
        y_all, labeled_mask = self._prepare_targets(raw_data)
        edge_index = torch.from_numpy(raw_data['edges'].T).long()

        # Compute statistics only on the training snapshot.
        train_df = features_df.loc[masks['train']]
        train_col_stats = self._compute_col_stats(train_df, col_to_stype)

        # Train snapshot.
        train_graph_mask_np = masks['train']
        train_graph_mask = torch.from_numpy(train_graph_mask_np).bool()
        train_label_mask = torch.from_numpy(masks['train'] & labeled_mask).bool()

        train_tf = self._build_tensor_frame(
            df=features_df.loc[train_graph_mask_np],
            col_to_stype=col_to_stype,
            col_stats=train_col_stats,
            y=None,
        )
        train_edge_index, _ = subgraph(
            train_graph_mask,
            edge_index,
            relabel_nodes=True,
        )
        train_node_id = torch.from_numpy(np.where(train_graph_mask_np)[0]).long()

        train_data = Data(
            edge_index=train_edge_index,
            y=y_all[train_graph_mask],
            mask=train_label_mask[train_graph_mask],
            x=train_tf,
            tf_col_stats=train_col_stats,
            node_id=train_node_id,
        )

        # Validation snapshot.
        val_graph_mask_np = masks['train'] | masks['val']
        val_graph_mask = torch.from_numpy(val_graph_mask_np).bool()
        val_label_mask = torch.from_numpy(masks['val'] & labeled_mask).bool()

        val_tf = self._build_tensor_frame(
            df=features_df.loc[val_graph_mask_np],
            col_to_stype=col_to_stype,
            col_stats=train_col_stats,
            y=None,
        )
        val_edge_index, _ = subgraph(
            val_graph_mask,
            edge_index,
            relabel_nodes=True,
        )
        val_node_id = torch.from_numpy(np.where(val_graph_mask_np)[0]).long()

        val_data = Data(
            edge_index=val_edge_index,
            y=y_all[val_graph_mask],
            mask=val_label_mask[val_graph_mask],
            x=val_tf,
            tf_col_stats=train_col_stats,
            node_id=val_node_id,
        )

        # Test snapshot.
        test_graph_mask_np = masks['train'] | masks['val'] | masks['test']
        test_graph_mask = torch.from_numpy(test_graph_mask_np).bool()
        test_label_mask = torch.from_numpy(masks['test'] & labeled_mask).bool()

        test_tf = self._build_tensor_frame(
            df=features_df.loc[test_graph_mask_np],
            col_to_stype=col_to_stype,
            col_stats=train_col_stats,
            y=None,
        )
        test_edge_index, _ = subgraph(
            test_graph_mask,
            edge_index,
            relabel_nodes=True,
        )
        test_node_id = torch.from_numpy(np.where(test_graph_mask_np)[0]).long()

        test_data = Data(
            edge_index=test_edge_index,
            y=y_all[test_graph_mask],
            mask=test_label_mask[test_graph_mask],
            x=test_tf,
            tf_col_stats=train_col_stats,
            node_id=test_node_id,
        )

        return [train_data, val_data, test_data]

    def process(self) -> None:
        data_list = (
            self._get_transductive_data()
            if self.split in ['RL', 'RH', 'TH']
            else self._get_inductive_data()
        )

        if self._to_undirected:
            to_undirected = ToUndirected()
            for i, data in enumerate(data_list):
                data_list[i] = to_undirected(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, split={self.split})'


def get_tabular_graphland_dataset(name: str, root: str, split: str = 'TH'):
    r"""Load HM / Pokec / WebTopic from GraphLandTensorFrameDataset.

    This loader is intentionally separate from the generic get_dataset()
    because these datasets store node attributes as TensorFrame instead of
    dense tensor features.

    Args:
        name (str): Dataset alias. Supported: ['HM', 'Pokec', 'WebTopic']
        root (str): Dataset root directory.
        split (str): GraphLand split. Defaults to 'TH'.

    Returns:
        Data: A PyG Data object with:
            - data.x: torch_frame.TensorFrame
            - data.edge_index: edge list
            - data.y: labels
            - train/val/test masks
            - tf_col_stats: statistics computed from train nodes only
    """

    _TABULAR_GRAPHLAND_NAME_MAP = {
        'hm': 'hm-categories',
        'pokec': 'pokec-regions',
        'webtopic': 'web-topics',
    }

    key = name.lower()
    if key not in _TABULAR_GRAPHLAND_NAME_MAP:
        raise ValueError(f'Unsupported tabular GraphLand dataset: {name}')

    dataset = GraphLandTensorFrameDataset(
        root=root,
        name=_TABULAR_GRAPHLAND_NAME_MAP[key],
        split=split,
        to_undirected=True,
    )

    data = dataset[0]
    data.y = data.y.squeeze()

    return data
