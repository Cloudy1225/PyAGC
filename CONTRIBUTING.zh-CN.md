# 参与 PyAGC

[English](CONTRIBUTING.md)

感谢你帮助改进 PyAGC。本项目是属性图聚类的算法库与基准。符合编码—聚类—优化框架的贡献最容易评审。

提交贡献即表示你同意以 [MIT 许可证](LICENSE) 授权你的工作。

## 可以贡献什么

- **缺陷报告。** 在 [issue](https://github.com/Cloudy1225/PyAGC/issues) 中给出最小示例：数据集或图规模、命令或代码、PyTorch / PyG 版本，以及报错栈。
- **新方法。** 在编码—聚类—优化框架下加入算法，并附上基准入口与测试。
- **新数据集。** 在 `pyagc.data.get_dataset` 后增加加载器，并简要说明领域、规模、特征与标签。
- **论文列表。** 把已发表的属性图聚类论文加入 [`AWESOME_AGC.md`](AWESOME_AGC.md)。
- **文档与示例。** 修改 README、`docs/` 下的 [Read the Docs](https://pyagc.readthedocs.io) 站点，或快速开始片段。

大型改动请先开 issue，例如新增一类数据集、修改现有 API。

## 开发环境

```bash
git clone https://github.com/Cloudy1225/PyAGC.git
cd PyAGC
pip install -e ".[dev]"
```

环境要求：Python >= 3.10，PyTorch >= 2.6.0，PyTorch Geometric >= 2.7.0。

在仓库根目录运行测试：

```bash
pytest
```

`dev` 额外依赖包含 `black`、`isort` 与 `flake8`。提交拉取请求前，请用 Black 和 isort 格式化改动过的 Python 文件。

## 添加方法

实现放在库中已有的对应位置：

| 内容 | 位置 |
|---|---|
| 模型 | `pyagc/models/<name>.py`，并在 `pyagc/models/__init__.py` 中导出 |
| 聚类头 | `pyagc/clusters/<name>.py`，并在 `pyagc/clusters/__init__.py` 中导出 |
| 编码器 | `pyagc/encoders/` |
| 基准 | `benchmark/<Method>/main.py` 与 `benchmark/<Method>/train.conf.yaml` |
| 测试 | `tests/models/`、`tests/clusters/` 或 `tests/enocders/` |

模型继承 `BaseModel`。可训练模型继承 `TrainableModel`，实现 `embed` 以及 `train_full` 所使用的损失。端到端聚类模型继承 `ClusteringModel`。聚类头继承 `BaseClusterHead`。

基准配置应写明学习率、轮数、网络结构、指标，以及训练是全图还是 mini-batch。目录布局参照 `benchmark/DMoN/`。方法特有的超参数放在该方法的 YAML 中，不要硬编码进 `main.py`。

一个新方法的拉取请求应包含：

1. 模型本身；若编码器或聚类头是新的，一并提交。
2. 在对应 `__init__.py` 中的导出。
3. `benchmark/<Method>/main.py` 与 `train.conf.yaml`。
4. 在小型合成图上运行的单元测试，检查损失或聚类分配的形状。测试不应下载基准数据集。
5. 若该方法进入公开基准，在 `README.md` 与 `README.zh-CN.md` 的算法表中各加一行。
6. 若该方法有论文，在 [`AWESOME_AGC.md`](AWESOME_AGC.md) 中增加条目。

## 添加数据集

数据加载统一经过 `pyagc/data/datasets.py` 中的 `get_dataset(name, root)`。数据集拉取请求应：

- 接受 `root` 目录，并把下载内容缓存到该目录。
- 按现有数据集的形式返回节点特征、`edge_index` 与标签。
- 说明名称别名、领域、节点数与边数、特征类型、簇数，以及许可证或来源。
- 不要提交原始数据文件。大图在使用时再下载。

## 论文列表

[`AWESOME_AGC.md`](AWESOME_AGC.md) 中的条目需要标题、发表场合与年份、链接，以及已有代码时的代码链接。把论文放进已有类别（综述、非参数、深度解耦或深度联合），不要为单篇论文新开一节。

## 拉取请求

- 一个拉取请求只做一件事。
- 说明改了什么、如何验证。代码改动请附上 `pytest` 结果。
- 新增的公开函数按 `pyagc/models/base.py` 的方式写文档。
- 不要提交数据集、检查点、日志或本地环境文件。

合并前，维护者可能会要求补充更小的测试或调整配置。问题可以发到 [issues](https://github.com/Cloudy1225/PyAGC/issues)。
