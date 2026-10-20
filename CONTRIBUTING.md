# Contributing to PyAGC

[中文说明](CONTRIBUTING.zh-CN.md)

Thanks for helping improve PyAGC. The project is a library and benchmark for attributed graph clustering. Contributions that fit the Encode-Cluster-Optimize (ECO) framework are the easiest to review.

By contributing, you agree that your work is licensed under the [MIT License](LICENSE).

## Ways to contribute

- **Bug reports.** Open an [issue](https://github.com/Cloudy1225/PyAGC/issues) with a minimal example: dataset or graph size, command or code, PyTorch / PyG versions, and the traceback.
- **New methods.** Add an algorithm under the ECO framework, with a benchmark entry and tests.
- **New datasets.** Add a loader behind `pyagc.data.get_dataset`, plus a short description of domain, scale, features, and labels.
- **Paper list.** Add a published AGC paper to [`AWESOME_AGC.md`](AWESOME_AGC.md).
- **Docs and examples.** Fix the README, the [Read the Docs](https://pyagc.readthedocs.io) site under `docs/`, or a quick-start snippet.

Please open an issue before a large change, such as a new dataset family or a breaking API change.

## Development setup

```bash
git clone https://github.com/Cloudy1225/PyAGC.git
cd PyAGC
pip install -e ".[dev]"
```

Requirements: Python >= 3.10, PyTorch >= 2.6.0, PyTorch Geometric >= 2.7.0.

Run the tests from the repository root:

```bash
pytest
```

`black`, `isort`, and `flake8` are included in the `dev` extra. Format changed Python files with Black and isort before opening a pull request.

## Add a method

Place the implementation where the rest of the library already lives:

| Piece | Location |
|---|---|
| Model | `pyagc/models/<name>.py`, exported from `pyagc/models/__init__.py` |
| Cluster head | `pyagc/clusters/<name>.py`, exported from `pyagc/clusters/__init__.py` |
| Encoder | `pyagc/encoders/` |
| Benchmark | `benchmark/<Method>/main.py` and `benchmark/<Method>/train.conf.yaml` |
| Tests | `tests/models/`, `tests/clusters/`, or `tests/enocders/` |

Models inherit from `BaseModel`. Trainable models inherit from `TrainableModel` and implement `embed` plus the loss used by `train_full`. End-to-end clustering models inherit from `ClusteringModel`. Cluster heads inherit from `BaseClusterHead`.

A benchmark config should record learning rate, epochs, architecture, metrics, and whether training is full-batch or mini-batch. Follow `benchmark/DMoN/` as the layout. Keep method-specific hyperparameters in that method's YAML instead of hard-coding them in `main.py`.

A new method PR should include:

1. The model (and cluster head or encoder, if it is new).
2. An export in the matching `__init__.py`.
3. `benchmark/<Method>/main.py` and `train.conf.yaml`.
4. A unit test that runs on a tiny synthetic graph and checks the loss or cluster assignment shape. Tests should not download benchmark datasets.
5. A one-line mention in the algorithm table of `README.md` and `README.zh-CN.md` when the method is part of the public benchmark.
6. An entry in [`AWESOME_AGC.md`](AWESOME_AGC.md) when the method has a paper.

## Add a dataset

Loaders go through `get_dataset(name, root)` in `pyagc/data/datasets.py`. A dataset PR should:

- Accept a `root` directory and cache downloads there.
- Return node features, `edge_index`, and labels in the same form as existing datasets.
- Document the name aliases, domain, node and edge counts, feature type, number of clusters, and license or source.
- Avoid committing raw data files. Large graphs stay as download-on-use.

## Paper list

Entries in [`AWESOME_AGC.md`](AWESOME_AGC.md) need the title, venue and year, a link, and code when it exists. Place the paper in the existing category (survey, non-parametric, deep decoupled, or deep joint) rather than adding a new section for a single paper.

## Pull requests

- One change per pull request.
- Describe what changed and how you checked it. Paste the `pytest` result for code changes.
- Keep new public functions documented in the same style as `pyagc/models/base.py`.
- Do not commit datasets, checkpoints, logs, or local environment files.

Maintainers may ask for a smaller test or a config change before merging. Questions can go to [issues](https://github.com/Cloudy1225/PyAGC/issues).
