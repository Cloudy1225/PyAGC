# Awesome Attributed Graph Clustering (AGC) Papers

A curated list of papers on Attributed Graph Clustering (AGC). This list accompanies the survey paper **"Attributed Graph Clustering: A Unified Framework, Comprehensive Review, and Industrial Perspective"** and the benchmark paper **"Bridging Academia and Industry: A Comprehensive Benchmark for Attributed Graph Clustering"**.

---

## Table of Contents

- [Survey & Benchmark](#survey--benchmark)
- [Research Papers](#research-papers)
  - [Non-Parametric & Decoupled Methods](#non-parametric--decoupled-methods)
  - [Deep Decoupled Methods](#deep-decoupled-methods)
  - [Deep Joint Methods](#deep-joint-methods)
  - [Hybrid Coordination Methods](#hybrid-coordination-methods)
  - [Multi-View and Multimodal Graph Clustering](#multi-view-and-multimodal-graph-clustering)
  - [Attributed Hypergraph Clustering](#attributed-hypergraph-clustering)
  - [Dynamic and Temporal Graph Clustering](#dynamic-and-temporal-graph-clustering)
  - [Attribute-Missing Graph Clustering](#attribute-missing-graph-clustering)
  - [Large-Scale and Scalable Methods](#large-scale-and-scalable-methods)
  - [LLM-Enhanced Graph Clustering](#llm-enhanced-graph-clustering)
  - [Federated Graph Clustering](#federated-graph-clustering)
  - [Other Extensions](#other-extensions)
- [Application Papers](#application-papers)
  - [Fraud Detection and Anomaly Detection](#fraud-detection-and-anomaly-detection)
  - [Recommendation Systems](#recommendation-systems)
  - [Graph Condensation and Distillation](#graph-condensation-and-distillation)
  - [Bioinformatics and Medical Science](#bioinformatics-and-medical-science)
  - [Natural Language Processing](#natural-language-processing)
- [ECO Taxonomy Quick Reference](#eco-taxonomy-quick-reference)
- [Community Resources](#community-resources)
- [Citation](#citation)

---

## Survey & Benchmark

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2022 | TKDE | [A Survey of Deep Graph Clustering: Taxonomy, Challenge, Application, and Open Resource](https://arxiv.org/abs/2211.12875) | [Code](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) |
| 2023 | TCSS | [An Overview of Advanced Deep Graph Node Clustering](https://ieeexplore.ieee.org/abstract/document/10049408) | — |
| 2023 | CIKM | [A Re-evaluation of Deep Learning Methods for Attributed Graph Clustering](https://dl.acm.org/doi/abs/10.1145/3583780.3614768) | [Code](https://github.com/2100271064/A-Re-evaluation-of-Deep-Learning-Methods-for-Attributed-Graph-Clustering) |
| 2025 | CSUR | [Clustering on Attributed Graphs: From Single-view to Multi-view](https://dl.acm.org/doi/10.1145/3714407) | — |
| 2025 | NeurIPS | [DGCBench: A Deep Graph Clustering Benchmark](https://openreview.net/forum?id=dKVUUZfcW9) | [Code](https://github.com/Marigoldwu/PyDGC) |
| 2025 | TPAMI | [Deep Temporal Graph Clustering: A Comprehensive Benchmark and Datasets](https://arxiv.org/abs/2601.12903) | [Code](https://github.com/MGitHubL/BenchTGC) |
| 2026 | arXiv | [Bridging Academia and Industry: A Comprehensive Benchmark for Attributed Graph Clustering](https://arxiv.org/abs/2602.08519) | [Code](https://github.com/Cloudy1225/PyAGC) |

---

## Research Papers

Papers are organized following the Encode-Cluster-Optimize framework introduced in the survey. Within each category, papers are sorted by year (descending).

### Non-Parametric & Decoupled Methods

> Encoders apply fixed spectral filtering operations without learnable weights; cluster projectors are applied post-hoc to frozen embeddings.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | ICML | [Scalable Attribute-Missing Graph Clustering via Neighborhood Differentiation](https://arxiv.org/abs/2507.13368) | — |
| 2025 | KDD | [Spectral Subspace Clustering for Attributed Graphs](https://arxiv.org/abs/2411.11074) | [Code](https://github.com/HKBU-LAGAS/S2CAG) |
| 2024 | CIKM | [Scalable and Adaptive Spectral Embedding for Attributed Graph Clustering](https://arxiv.org/abs/2408.05765) | — |
| 2023 | AAAI | [Scalable Attributed-Graph Subspace Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/25918) | [Code](https://github.com/chakib401/sagsc) |
| 2023 | TKDE | [Adaptive Graph Convolution Methods for Attributed Graph Clustering](https://ieeexplore.ieee.org/abstract/document/10130603) | [Code](https://github.com/Karenxt/AGCandIAGC-code) |
| 2023 | TKDE | [Boosting Subspace Co-Clustering via Bilateral Graph Convolution](https://ieeexplore.ieee.org/document/10207697) | [Code](https://github.com/chakib401/sc3) |
| 2022 | TKDD | [GRACE: A General Graph Convolution Framework for Attributed Graph Clustering](https://dl.acm.org/doi/10.1145/3544977) | [Code](https://github.com/BarakeelFanseu/GRACE) |
| 2022 | ICML | [NAFS: A Simple yet Tough-to-beat Baseline for Graph Representation Learning](https://arxiv.org/abs/2206.08583) | [Code](https://github.com/zwt233/NAFS) |
| 2022 | SDM | [Fine-grained Attributed Graph Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.42) | [Code](https://github.com/sckangz/FGC) |
| 2021 | ICLR | [Simple Spectral Graph Convolution](https://openreview.net/forum?id=CYO5T-YjWZV) | [Code](https://github.com/allenhaozhu/SSGC) |
| 2021 | CIKM | [HyperGraph Convolution Based Attributed HyperGraph Clustering](https://dl.acm.org/doi/10.1145/3459637.3482437) | [Code](https://github.com/BarakeelFanseu/GRAC_CIKM) |
| 2019 | IJCAI | [Attributed Graph Clustering via Adaptive Graph Convolution](https://arxiv.org/abs/1906.01210) | [Code](https://github.com/karenlatong/AGC-master) |

### Deep Decoupled Methods

> Parametric encoders are pre-trained with self-supervised representation objectives; cluster projectors are applied post-hoc to frozen embeddings.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | ICLR | [Compactness and Consistency: A Conjoint Framework for Deep Graph Clustering](https://openreview.net/forum?id=9jdQLmPUHW) | [Code](https://github.com/juweipku/CoCo) |
| 2025 | AAAI | [One Node One Model: Featuring the Missing-Half for Graph Clustering](https://arxiv.org/abs/2412.09902) | [Code](https://github.com/XieXuanting/FPGC) |
| 2025 | TNNLS | [SynC: Synergistic Boosting of Structure and Representation for Deep Graph Clustering](https://ieeexplore.ieee.org/document/11314181) | [Code](https://github.com/Marigoldwu/SynC) |
| 2025 | TKDE | [Trustworthy Neighborhoods Mining: Homophily-Aware Neutral Contrastive Learning for Graph Clustering](https://arxiv.org/abs/2512.15027) | [Code](https://github.com/THPengL/NeuCGC) |
| 2025 | Neurocomputing | [SCGC: Self-supervised Contrastive Graph Clustering](https://arxiv.org/abs/2204.12656) | [Code](https://github.com/gayanku/SCGC) |
| 2024 | TKDE | [Reliable Node Similarity Matrix Guided Contrastive Graph Clustering](https://arxiv.org/abs/2408.03765) | [Code](https://github.com/Cloudy1225/NS4GC) |
| 2024 | Neural Networks | [Negative-Free Self-Supervised Gaussian Embedding of Graphs](https://arxiv.org/abs/2411.01157) | [Code](https://github.com/Cloudy1225/SSGE) |
| 2024 | KDD | [Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective](https://arxiv.org/abs/2406.14288) | [Code](https://github.com/EdisonLeeeee/MAGI) |
| 2024 | Nature Communications | [Network Community Detection via Neural Embeddings](https://www.nature.com/articles/s41467-024-52355-w) | [Code](https://github.com/skojaku/community-detection-via-neural-embedding) |
| 2024 | TNNLS | [Improved Dual Correlation Reduction Network With Affinity Recovery](https://ieeexplore.ieee.org/abstract/document/10605097) | [Code](https://github.com/yueliu1999/IDCRN) |
| 2024 | TNNLS | [An End-to-End Deep Graph Clustering via Online Mutual Learning](https://ieeexplore.ieee.org/document/10412657) | — |
| 2024 | ECML-PKDD | [Bootstrap Latents of Nodes and Neighbors for Graph Self-Supervised Learning](https://arxiv.org/abs/2408.05087) | [Code](https://github.com/Cloudy1225/BLNN) |
| 2024 | LoG | [Large Language Model Guided Graph Clustering](https://openreview.net/forum?id=CLyhlb5DG5) | — |
| 2023 | TNNLS | [Simple Contrastive Graph Clustering](https://ieeexplore.ieee.org/document/10163985) | [Code](https://github.com/yueliu1999/SCGC) |
| 2023 | TNNLS | [Redundancy-Free Self-Supervised Relational Learning for Graph Clustering](https://arxiv.org/abs/2309.04694) | [Code](https://github.com/yisiyu95/R2FGC) |
| 2023 | TNNLS | [Dual Contrastive Learning Network for Graph Clustering](https://ieeexplore.ieee.org/document/10097557) | [Code](https://github.com/XinPeng97/TNNLS_DCLN) |
| 2023 | TKDE | [Hierarchical Contrastive Learning Enhanced Heterogeneous Graph Neural Network](https://arxiv.org/abs/2304.12228) | — |
| 2023 | Neurocomputing | [Wasserstein Adversarially Regularized Graph Autoencoder](https://www.sciencedirect.com/science/article/pii/S0925231223003582) | [Code](https://github.com/LeonResearch/WARGA) |
| 2023 | Neurocomputing | [Neighborhood Contrastive Representation Learning for Attributed Graph Clustering](https://doi.org/10.1016/j.neucom.2023.126880) | [Code](https://github.com/wangtong627/NCAGC-NeuroCom) |
| 2023 | Neurocomputing | [Mutual Boost Network for Attributed Graph Clustering](https://www.sciencedirect.com/science/article/pii/S0957417423009818) | [Code](https://github.com/Xiaoqiang-Yan/MBN) |
| 2023 | IJCAI | [CONGREGATE: Contrastive Graph Clustering in Curvature Spaces](https://arxiv.org/abs/2305.03555) | [Code](https://github.com/CurvCluster/Congregate) |
| 2023 | ICML | [Beyond Homophily: Reconstructing Structure for Graph-agnostic Clustering](https://arxiv.org/abs/2305.02931) | [Code](https://github.com/Panern/DGCN) |
| 2023 | KDD | [CARL-G: Clustering-Accelerated Representation Learning on Graphs](https://arxiv.org/abs/2306.06936) | [Code](https://github.com/willshiao/carl-g) |
| 2023 | JMLR | [Graph Clustering with Graph Neural Networks](https://arxiv.org/abs/2006.16904) | [Code](https://github.com/google-research/google-research/tree/master/graph_embedding/dmon) |
| 2023 | PR | [A Contrastive Variational Graph Auto-Encoder for Node Clustering](https://arxiv.org/abs/2312.16830) | [Code](https://github.com/nairouz/CVGAE_PR) |
| 2023 | PR | [Graph Clustering Network with Structure Embedding Enhanced](https://www.sciencedirect.com/science/article/pii/S0031320323005319) | [Code](https://github.com/Marigoldwu/GC-SEE) |
| 2023 | SDM | [Beyond The Evidence Lower Bound: Dual Variational Graph Auto-Encoders For Node Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch12) | [Code](https://github.com/nairouz/BELBO-VGAE) |
| 2022 | NeurIPS | [Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination](https://arxiv.org/abs/2206.01535) | [Code](https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination) |
| 2022 | NeurIPS | [S3GC: Scalable Self-Supervised Graph Clustering](https://proceedings.neurips.cc/paper_files/paper/2022/hash/15972a9575e0f03bf82f00aebeb40774-Abstract-Conference.html) | [Code](https://github.com/devvrit/S3GC) |
| 2022 | ICLR | [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514) | [Code](https://github.com/nerdslab/bgrl) |
| 2022 | KDD | [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803) | [Code](https://github.com/THUDM/GraphMAE) |
| 2022 | AAAI | [Deep Graph Clustering via Dual Correlation Reduction](https://arxiv.org/abs/2112.14772) | [Code](https://github.com/yueliu1999/DCRN) |
| 2022 | AAAI | [Augmentation-Free Self-Supervised Learning on Graphs](https://arxiv.org/abs/2112.02472) | [Code](https://github.com/Namkyeong/AFGRL) |
| 2022 | IJCAI | [Attributed Graph Clustering with Dual Redundancy Reduction](https://www.ijcai.org/proceedings/2022/0418) | [Code](https://github.com/gongleii/AGC-DRR) |
| 2022 | IJCAI | [Escaping Feature Twist: A Variational Graph Auto-Encoder for Node Clustering](https://www.ijcai.org/proceedings/2022/465) | [Code](https://github.com/nairouz/FT-VGAE) |
| 2022 | KBS | [Graph Barlow Twins: A Self-Supervised Representation Learning Framework for Graphs](https://arxiv.org/abs/2106.02466) | [Code](https://github.com/pbielak/graph-barlow-twins) |
| 2022 | TKDE | [Rethinking Graph Auto-Encoder Models for Attributed Graph Clustering](https://arxiv.org/abs/2107.08562) | [Code](https://github.com/nairouz/R-GAE) |
| 2022 | TKDE | [SAGES: Scalable Attributed Graph Embedding With Sampling for Unsupervised Learning](https://ieeexplore.ieee.org/document/9705119) | [Code](https://github.com/SAGESAlgorithm/SAGES) |
| 2022 | TNNLS | [Embedding Graph Auto-Encoder for Graph Clustering](https://ieeexplore.ieee.org/document/9741755) | [Code](https://github.com/hyzhang98/EGAE) |
| 2022 | WWW | [Graph Communal Contrastive Learning](https://arxiv.org/abs/2110.14863) | [Code](https://github.com/lblaoke/gCooL) |
| 2022 | WWW | [CGC: Contrastive Graph Clustering for Community Detection and Tracking](https://arxiv.org/abs/2204.08504) | [Code](https://github.com/NamyongPark/CGC-Data) |
| 2022 | WSDM | [Cluster-Aware Heterogeneous Information Network Embedding](https://dl.acm.org/doi/10.1145/3488560.3498385) | — |
| 2022 | ASONAM | [Deep Graph Clustering with Random-walk based Scalable Learning](https://arxiv.org/abs/2112.15530) | — |
| 2021 | NeurIPS | [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484) | [Code](https://github.com/hengruizhang98/CCA-SSG) |
| 2021 | KDD | [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](https://arxiv.org/abs/2105.09111) | [Code](https://github.com/liun-online/HeCo) |
| 2021 | KDD | [Spectral Clustering of Attributed Multi-relational Graphs](https://arxiv.org/abs/2311.01840) | — |
| 2021 | Neural Networks | [Spectral Embedding Network for Attributed Graph Clustering](https://www.sciencedirect.com/science/article/pii/S0893608021002227) | — |
| 2021 | TKDE | [CaEGCN: Cross-Attention Fusion based Enhanced Graph Convolutional Network for Clustering](https://arxiv.org/abs/2101.06883) | [Code](https://github.com/huogy/CaEGCN) |
| 2021 | TPAMI | [Adaptive Graph Auto-Encoder for General Data Clustering](https://arxiv.org/abs/2002.08648) | [Code](https://github.com/hyzhang98/AdaGAE) |
| 2021 | WWW | [Effective and Scalable Clustering on Massive Attributed Graphs](https://arxiv.org/abs/2102.03826) | [Code](https://github.com/AnryYang/ACMin) |
| 2020 | ICML | [Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/abs/2006.05582) | [Code](https://github.com/kavehhassani/mvgrl) |
| 2020 | KDD | [Adaptive Graph Encoder for Attributed Graph Embedding](https://arxiv.org/abs/2007.01594) | [Code](https://github.com/thunlp/AGE) |
| 2020 | CIKM | [CommDGI: Community Detection Oriented Deep Graph Infomax](https://dl.acm.org/doi/10.1145/3340531.3412042) | [Code](https://github.com/FDUDSDE/CommDGI) |
| 2020 | AAAI | [Unsupervised Attributed Multiplex Network Embedding](https://arxiv.org/abs/1911.06750) | [Code](https://github.com/pcy1302/DMGI) |
| 2019 | ICLR | [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) | [Code](https://github.com/PetarV-/DGI) |
| 2019 | ICCV | [Symmetric Graph Convolutional Autoencoder for Unsupervised Graph Representation Learning](https://arxiv.org/abs/1908.02441) | [Code](https://github.com/sseung0703/GALA_TF2.0) |
| 2018 | IJCAI | [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407) | [Code](https://github.com/TrustAGI-Lab/ARGA) |
| 2016 | NeurIPS-W | [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) | [Code](https://github.com/tkipf/gae) |
| 2016 | KDD | [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) | [Code](https://github.com/aditya-grover/node2vec) |

### Deep Joint Methods

> Parametric encoders and cluster projectors are optimized simultaneously, allowing clustering objectives to directly shape the representation space.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | TPAMI | [ASIL: Augmented Structural Information Learning for Deep Graph Clustering in Hyperbolic Space](https://arxiv.org/abs/2504.09970) | [Code](https://github.com/RiemannGraph/DSE_clustering) |
| 2025 | TPAMI | [Clustering Diffusion Model With Frequency-Signal Modulation for Variational Graph Autoencoders](https://ieeexplore.ieee.org/document/11180140) | [Code](https://github.com/Roiko97/FVD) |
| 2025 | TPAMI | [Graph Prompt Clustering](https://ieeexplore.ieee.org/document/10935718) | [Code](https://github.com/ManshengChen/Code-for-GPC-master) |
| 2025 | KDD | [Unsupervised Graph Clustering with Deep Structural Entropy](https://arxiv.org/abs/2505.14040) | [Code](https://github.com/SELGroup/DeSE) |
| 2025 | AAAI | [Deep Multi-modal Graph Clustering via Graph Transformer Network](https://ojs.aaai.org/index.php/AAAI/article/view/32844) | — |
| 2025 | LoG | [Differentiable Community Detection with Graph Neural Networks and Stochastic Block Models](https://openreview.net/forum?id=T1vdfm1THf) | — |
| 2024 | NeurIPS | [The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks](https://arxiv.org/abs/2310.01144) | [Code](https://github.com/chrisbloecker/neuromap) |
| 2024 | ICML | [LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering](https://arxiv.org/abs/2405.11801) | [Code](https://github.com/ZhenhHuang/LSEnet/tree/main) |
| 2024 | AAAI | [DGCLUSTER: A Neural Framework for Attributed Graph Clustering via Modularity Maximization](https://arxiv.org/abs/2312.12697) | [Code](https://github.com/pyrobits/DGCluster) |
| 2024 | AAAI | [Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering](https://arxiv.org/abs/2401.06595) | [Code](https://github.com/q086/DyFSS) |
| 2024 | TNNLS | [Contrastive Multiview Attribute Graph Clustering With Adaptive Encoders](https://ieeexplore.ieee.org/document/10509800) | — |
| 2023 | ICML | [Dink-Net: Neural Clustering on Large Graphs](https://arxiv.org/abs/2305.18405) | [Code](https://github.com/yueliu1999/Dink-Net) |
| 2023 | JMLR | [Graph Clustering with Graph Neural Networks](https://arxiv.org/abs/2006.16904) | [Code](https://github.com/google-research/google-research/tree/master/graph_embedding/dmon) |
| 2022 | WSDM | [Efficient Graph Convolution for Joint Node Representation Learning and Clustering](https://dl.acm.org/doi/10.1145/3488560.3498533) | [Code](https://github.com/chakib401/graph_convolutional_clustering) |
| 2022 | CIKM | [Higher-order Clustering and Pooling for Graph Neural Networks](https://arxiv.org/abs/2209.03473) | [Code](https://github.com/AlexDuvalinho/HoscPool) |
| 2022 | TNNLS | [Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering](https://ieeexplore.ieee.org/document/9777842) | [Code](https://github.com/Jillian555/TNNLS_CDRS) |
| 2022 | PR | [Graph Clustering via Variational Graph Embedding](https://www.sciencedirect.com/science/article/pii/S0031320321005148) | — |
| 2022 | PR | [Deep Graph Clustering with Multi-level Subspace Fusion](https://www.sciencedirect.com/science/article/pii/S003132032200557X) | — |
| 2021 | AAAI | [Deep Fusion Clustering Network](https://arxiv.org/abs/2012.09600) | [Code](https://github.com/WxTu/DFCN) |
| 2021 | MM | [Attention-driven Graph Clustering Network](https://arxiv.org/abs/2108.05499) | [Code](https://github.com/ZhihaoPENG-CityU/MM21---AGCN) |
| 2020 | ICML | [Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/abs/1907.00481) | [Code](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling) |
| 2020 | NeurIPS | [Dirichlet Graph Variational Autoencoder](https://arxiv.org/abs/2010.04408) | [Code](https://github.com/xiyou3368/DGVAE) |
| 2020 | WWW | [Structural Deep Clustering Network](https://arxiv.org/abs/2002.01633) | [Code](https://github.com/bdy9527/SDCN) |
| 2019 | IJCAI | [Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://arxiv.org/abs/1906.06532) | [Code](https://github.com/Tiger101010/DAEGC) |

### Hybrid Coordination Methods

> Methods that interleave decoupled pre-training and joint fine-tuning, using confidence-gated feedback, iterative self-training, or reinforcement-based coordination.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | WWW | [Diffusion-based Graph-agnostic Clustering](https://dl.acm.org/doi/10.1145/3696410.3714652) | [Code](https://github.com/kkkkk001/DGAC) |
| 2025 | NeurIPS | [Hybrid-Collaborative Augmentation and Contrastive Sample Adaptive-Differential Awareness for Robust Attributed Graph Clustering](https://arxiv.org/abs/2510.02731) | [Code](https://github.com/TianxiangZhao0474/RAGC) |
| 2024 | TKDD | [Towards Faster Deep Graph Clustering via Efficient Graph Auto-Encoder](https://dl.acm.org/doi/abs/10.1145/3674983) | [Code](https://github.com/Marigoldwu/FastDGC) |
| 2024 | KDD | [NeuroCUT: A Neural Approach for Robust Graph Partitioning](https://arxiv.org/abs/2310.11787) | [Code](https://github.com/idea-iitd/NeuroCut) |
| 2024 | AAAI | [Upper Bounding Barlow Twins: A Novel Filter for Multi-Relational Clustering](https://arxiv.org/abs/2312.14066) | [Code](https://github.com/XweiQ/BTGF) |
| 2024 | TAI | [GLAC-GCN: Global and Local Topology-Aware Contrastive Graph Clustering Network](https://ieeexplore.ieee.org/document/10557452) | [Code](https://github.com/xuyuankun631/GLAC-GCN) |
| 2023 | AAAI | [Hard Sample Aware Network for Contrastive Deep Graph Clustering](https://arxiv.org/abs/2212.08665) | [Code](https://github.com/yueliu1999/HSAN) |
| 2023 | AAAI | [Cluster-guided Contrastive Graph Clustering Network](https://arxiv.org/abs/2301.01098) | [Code](https://github.com/xihongyang1999/CCGC) |
| 2023 | CIKM | [Homophily-enhanced Structure Learning for Graph Clustering](https://arxiv.org/abs/2308.05309) | [Code](https://github.com/galogm/HoLe) |
| 2023 | CIKM | [Robust Graph Clustering via Meta Learning for Noisy Graphs](https://arxiv.org/abs/2311.00322) | [Code](https://github.com/HyeonsooJo/MetaGC) |
| 2023 | MM | [CONVERT: Contrastive Graph Clustering with Reliable Augmentation](https://arxiv.org/abs/2308.08963) | [Code](https://github.com/xihongyang1999/CONVERT) |
| 2023 | MM | [Reinforcement Graph Clustering with Unknown Cluster Number](https://arxiv.org/abs/2308.06827) | [Code](https://github.com/yueliu1999/RGC) |
| 2023 | ECML-PKDD | [Contrastive Learning with Cluster-Preserving Augmentation for Attributed Graph Clustering](https://link.springer.com/chapter/10.1007/978-3-031-43412-9_38) | [Code](https://github.com/Zhengymm/CCA-AGC) |
| 2023 | IJCAI | [Multi-level Graph Contrastive Prototypical Clustering](https://www.ijcai.org/proceedings/2023/0513) | — |
| 2023 | TIST | [Unsupervised Graph Representation Learning with Cluster-aware Self-training and Refining](https://dl.acm.org/doi/10.1145/3608480) | — |

---

### Multi-View and Multimodal Graph Clustering

> Methods handling multiple graph views, attribute views, heterogeneous information networks, or multimodal attributed graphs.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | KDD | [Cross-Contrastive Clustering for Multimodal Attributed Graphs with Dual Graph Filtering](https://arxiv.org/abs/2511.20030) | [Code](https://github.com/HaoranZ99/DGF) |
| 2025 | MM | [Disentangling Homophily and Heterophily in Multimodal Graph Clustering](https://arxiv.org/abs/2507.15253) | [Code](https://github.com/Uncnbb/DMGC) |
| 2025 | AAAI | [Deep Multi-modal Graph Clustering via Graph Transformer Network](https://ojs.aaai.org/index.php/AAAI/article/view/32844) | — |
| 2025 | IJCAI | [TOTF: Missing-Aware Encoders for Clustering on Multi-View Incomplete Attributed Graphs](https://idealab.alibaba-inc.com/ideaTalk) | — |
| 2025 | ICML | [Multi-View Graph Clustering via Node-Guided Contrastive Encoding](https://proceedings.mlr.press/v267/ren25a.html) | [Code](https://github.com/Rirayh/NGCE) |
| 2025 | CVPR | [Attribute-Missing Multi-view Graph Clustering](https://ieeexplore.ieee.org/document/11094624) | — |
| 2025 | TMM | [Prototype-Driven Multi-View Attribute-Missing Graph Clustering](https://ieeexplore.ieee.org/abstract/document/11202623) | — |
| 2024 | TPAMI | [EBMGC-GNF: Efficient Balanced Multi-View Graph Clustering via Good Neighbor Fusion](https://ieeexplore.ieee.org/document/10522848) | [Code](https://github.com/ZaneeYang/TPAMI2024-EBMGC-GNF) |
| 2024 | TNNLS | [Contrastive Multiview Attribute Graph Clustering With Adaptive Encoders](https://ieeexplore.ieee.org/document/10509800) | — |
| 2024 | MM | [Balanced Multi-Relational Graph Clustering](https://arxiv.org/abs/2407.16863) | [Code](https://github.com/zxlearningdeep/BMGC) |
| 2024 | AAAI | [Upper Bounding Barlow Twins: A Novel Filter for Multi-Relational Clustering](https://arxiv.org/abs/2312.14066) | [Code](https://github.com/XweiQ/BTGF) |
| 2024 | IJCAI | [Dual Contrastive Graph-Level Clustering with Multiple Cluster Perspectives Alignment](https://www.ijcai.org/proceedings/2024/0417) | [Code](https://github.com/wownice333/DCGLC) |
| 2024 | TKDE | [BGAE: Auto-Encoding Multi-View Bipartite Graph Clustering](https://ieeexplore.ieee.org/document/10423800) | [Code](https://github.com/liliangnudt/BGAE) |
| 2023 | NeurIPS | [Multi-view Contrastive Graph Clustering](https://arxiv.org/abs/2110.11842) | [Code](https://github.com/panern/mcgc) |
| 2023 | TKDE | [Multi-View Bipartite Graph Clustering With Coupled Noisy Feature Filter](https://ieeexplore.ieee.org/abstract/document/10109823) | [Code](https://github.com/liliangnudt/MVBGC-NFF) |
| 2023 | TKDE | [Hierarchical Contrastive Learning Enhanced Heterogeneous Graph Neural Network](https://arxiv.org/abs/2304.12228) | — |
| 2022 | WSDM | [Cluster-Aware Heterogeneous Information Network Embedding](https://dl.acm.org/doi/10.1145/3488560.3498385) | — |
| 2021 | NeurIPS | [Multi-view Contrastive Graph Clustering](https://arxiv.org/abs/2110.11842) | [Code](https://github.com/panern/mcgc) |
| 2021 | TKDE | [Multi-View Attributed Graph Clustering](https://ieeexplore.ieee.org/document/9508843) | [Code](https://github.com/sckangz/MAGC) |
| 2021 | IJCAI | [Graph Filter-based Multi-view Attributed Graph Clustering](https://www.ijcai.org/proceedings/2021/375) | [Code](https://github.com/sckangz/MvAGC) |
| 2021 | KDD | [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](https://arxiv.org/abs/2105.09111) | [Code](https://github.com/liun-online/HeCo) |
| 2021 | KDD | [Spectral Clustering of Attributed Multi-relational Graphs](https://arxiv.org/abs/2311.01840) | — |
| 2020 | WSDM | [Deep Multi-Graph Clustering via Attentive Cross-Graph Association](https://dl.acm.org/doi/abs/10.1145/3336191.3371806) | [Code](https://github.com/flyingdoog/DMGC) |
| 2020 | IJCAI | [MAGCN: Multi-View Attribute Graph Convolution Networks for Clustering](https://www.ijcai.org/proceedings/2020/411) | [Code](https://github.com/IMKBLE/MAGCN) |
| 2020 | IJCAI | [JANE: Jointly Adversarial Network Embedding](https://www.ijcai.org/Proceedings/2020/192) | — |
| 2020 | AAAI | [Unsupervised Attributed Multiplex Network Embedding](https://arxiv.org/abs/1911.06750) | [Code](https://github.com/pcy1302/DMGI) |
| 2020 | WWW | [One2Multi Graph Autoencoder for Multi-view Graph Clustering](https://dl.acm.org/doi/abs/10.1145/3366423.3380079) | [Code](https://github.com/googlebaba/WWW2020-O2MAC) |

---

### Attributed Hypergraph Clustering

> Methods for clustering on hypergraphs where hyperedges connect arbitrary subsets of nodes.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | SIGMOD | [On Graph Representation for Attributed Hypergraph Clustering](https://dl.acm.org/doi/10.1145/3709741) | [Code](https://github.com/ForPaperSubmissions/Attributed_Hypergraph_Representation_for_Clustering_Code) |
| 2025 | ICCV | [Hypergraph Clustering Network with Partial Attribute Imputation](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Hypergraph_Clustering_Network_with_Partial_Attribute_Imputation_ICCV_2025_paper.html) | — |
| 2025 | IJCAI | [A Simple yet Effective Hypergraph Clustering Network](https://www.ijcai.org/proceedings/2025/0707) | — |
| 2024 | VLDB | [A Versatile Framework for Attributed Network Clustering via K-Nearest Neighbor Augmentation](https://arxiv.org/abs/2408.05459) | [Code](https://github.com/gongyguo/ANCKA) |
| 2023 | SIGMOD | [Efficient and Effective Attributed Hypergraph Clustering via K-Nearest Neighbor Augmentation](https://dl.acm.org/doi/10.1145/3589261) | [Code](https://github.com/CyanideCentral/AHCKA) |
| 2021 | CIKM | [HyperGraph Convolution Based Attributed HyperGraph Clustering](https://dl.acm.org/doi/10.1145/3459637.3482437) | [Code](https://github.com/BarakeelFanseu/GRAC_CIKM) |

---

### Dynamic and Temporal Graph Clustering

> Methods for clustering on evolving or temporal graphs.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | TPAMI | [Deep Temporal Graph Clustering: A Comprehensive Benchmark and Datasets](https://arxiv.org/abs/2601.12903) | [Code](https://github.com/MGitHubL/BenchTGC) |
| 2024 | ICLR | [Deep Temporal Graph Clustering](https://arxiv.org/abs/2305.10738) | [Code](https://github.com/MGitHubL/TGC) |
| 2022 | WWW | [CGC: Contrastive Graph Clustering for Community Detection and Tracking](https://arxiv.org/abs/2204.08504) | [Code](https://github.com/NamyongPark/CGC-Data) |
| 2021 | CIKM | [Robust Dynamic Clustering for Temporal Networks](https://dl.acm.org/doi/10.1145/3459637.3482473) | — |

---

### Attribute-Missing Graph Clustering

> Methods handling partially or fully missing node attributes.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | MM | [Clustering-Oriented Generative Attribute Graph Imputation](https://arxiv.org/abs/2507.19085) | — |
| 2025 | ICML | [Scalable Attribute-Missing Graph Clustering via Neighborhood Differentiation](https://arxiv.org/abs/2507.13368) | — |
| 2025 | ICCV | [Hypergraph Clustering Network with Partial Attribute Imputation](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Hypergraph_Clustering_Network_with_Partial_Attribute_Imputation_ICCV_2025_paper.html) | — |
| 2025 | CVPR | [Attribute-Missing Multi-view Graph Clustering](https://ieeexplore.ieee.org/document/11094624) | — |
| 2025 | TMM | [Prototype-Driven Multi-View Attribute-Missing Graph Clustering](https://ieeexplore.ieee.org/abstract/document/11202623) | — |
| 2025 | IJCAI | [TOTF: Missing-Aware Encoders for Clustering on Multi-View Incomplete Attributed Graphs](https://idealab.alibaba-inc.com/ideaTalk) | — |

---

### Large-Scale and Scalable Methods

> Methods specifically designed for graphs with millions to billions of nodes, emphasizing linear-time complexity or mini-batch training.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | SIGMOD | [Effective Clustering for Large Multi-Relational Graphs](https://arxiv.org/abs/2508.17388) | [Code](https://github.com/HKBU-LAGAS/DEMM) |
| 2025 | ICML | [Scalable Attribute-Missing Graph Clustering via Neighborhood Differentiation](https://arxiv.org/abs/2507.13368) | — |
| 2025 | KDD | [Spectral Subspace Clustering for Attributed Graphs](https://arxiv.org/abs/2411.11074) | [Code](https://github.com/HKBU-LAGAS/S2CAG) |
| 2024 | VLDB | [A Versatile Framework for Attributed Network Clustering via K-Nearest Neighbor Augmentation](https://arxiv.org/abs/2408.05459) | [Code](https://github.com/gongyguo/ANCKA) |
| 2024 | KDD | [Effective Clustering on Large Attributed Bipartite Graphs](https://arxiv.org/abs/2405.11922) | [Code](https://github.com/HKBU-LAGAS/TPC) |
| 2024 | KDD | [Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective](https://arxiv.org/abs/2406.14288) | [Code](https://github.com/EdisonLeeeee/MAGI) |
| 2024 | CIKM | [Scalable and Adaptive Spectral Embedding for Attributed Graph Clustering](https://arxiv.org/abs/2408.05765) | — |
| 2023 | ICML | [Dink-Net: Neural Clustering on Large Graphs](https://arxiv.org/abs/2305.18405) | [Code](https://github.com/yueliu1999/Dink-Net) |
| 2023 | SIGMOD | [Efficient and Effective Attributed Hypergraph Clustering via K-Nearest Neighbor Augmentation](https://dl.acm.org/doi/10.1145/3589261) | [Code](https://github.com/CyanideCentral/AHCKA) |
| 2022 | NeurIPS | [S3GC: Scalable Self-Supervised Graph Clustering](https://proceedings.neurips.cc/paper_files/paper/2022/hash/15972a9575e0f03bf82f00aebeb40774-Abstract-Conference.html) | [Code](https://github.com/devvrit/S3GC) |
| 2022 | NeurIPS | [Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination](https://arxiv.org/abs/2206.01535) | [Code](https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination) |
| 2022 | ICLR | [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514) | [Code](https://github.com/nerdslab/bgrl) |
| 2022 | TKDE | [SAGES: Scalable Attributed Graph Embedding With Sampling for Unsupervised Learning](https://ieeexplore.ieee.org/document/9705119) | [Code](https://github.com/SAGESAlgorithm/SAGES) |
| 2022 | ASONAM | [Deep Graph Clustering with Random-walk based Scalable Learning](https://arxiv.org/abs/2112.15530) | — |
| 2021 | WWW | [Effective and Scalable Clustering on Massive Attributed Graphs](https://arxiv.org/abs/2102.03826) | [Code](https://github.com/AnryYang/ACMin) |
| 2016 | KDD | [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) | [Code](https://github.com/aditya-grover/node2vec) |

---

### LLM-Enhanced Graph Clustering

> Methods leveraging large language models for text-attributed graph clustering.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | ACL Findings | [MARK: Multi-agent Collaboration with Ranking Guidance for Text-attributed Graph Clustering](https://aclanthology.org/2025.findings-acl.314/) | [Code](https://github.com/fuyw-aisw/MARK) |
| 2024 | LoG | [Large Language Model Guided Graph Clustering](https://openreview.net/forum?id=CLyhlb5DG5) | — |

---

### Federated Graph Clustering

> Methods for privacy-preserving distributed graph clustering across multiple clients.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | ICLR | [Federated Graph-Level Clustering Network with Dual Knowledge Separation](https://openreview.net/forum?id=FwKFjBX0PK) | — |
| 2025 | ICML | [Federated Node-Level Clustering Network with Cross-Subgraph Link Mending](https://openreview.net/forum?id=38Nh0TebXZ) | — |
| 2025 | AAAI | [Federated Graph-Level Clustering Network](https://ojs.aaai.org/index.php/AAAI/article/view/34077) | — |

---

### Other Extensions

> Methods addressing specialized graph clustering settings including signed graphs, heterophilous graphs, and unknown cluster numbers.

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | WWW | [Robust Deep Signed Graph Clustering via Weak Balance Theory](https://arxiv.org/abs/2502.05472) | [Code](https://github.com/yaoyaohuanghuang/DSGC) |
| 2024 | TKDD | [Towards Faster Deep Graph Clustering via Efficient Graph Auto-Encoder](https://dl.acm.org/doi/abs/10.1145/3674983) | [Code](https://github.com/Marigoldwu/FastDGC) |
| 2023 | MM | [Reinforcement Graph Clustering with Unknown Cluster Number](https://arxiv.org/abs/2308.06827) | [Code](https://github.com/yueliu1999/RGC) |
| 2020 | TPAMI | [Comparing Graph Clusterings: Set Partition Measures vs. Graph-Aware Measures](https://arxiv.org/abs/1806.11494) | — |
| 2014 | TKDD | [GBAGC: A General Bayesian Framework for Attributed Graph Clustering](https://dl.acm.org/doi/10.1145/2629616) | — |
| 2012 | SIGMOD | [A Model-based Approach to Attributed Graph Clustering](https://dl.acm.org/doi/abs/10.1145/2213836.2213894) | [Code](https://github.com/zhiqiangxu2001/BAGC) |
| 2011 | TKDD | [Clustering Large Attributed Graphs: A Balance between Structural and Attribute Similarities](https://dl.acm.org/doi/abs/10.1145/1921632.1921638) | [Code](https://github.com/Sionzzz/SA-cluster) |

---

## Application Papers

> Papers applying graph clustering techniques to downstream real-world tasks, organized by application domain.

### Fraud Detection and Anomaly Detection

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | ICLR | [Escaping the Homophily Trap: A Threshold-free Graph Outlier Detection Framework via Clustering-guided Edge Reweighting](https://openreview.net/forum?id=Z8f0whjttd) | — |
| 2025 | WWW | [Cluster Aware Graph Anomaly Detection](https://arxiv.org/abs/2409.09770) | [Code](https://github.com/zhenglecheng/CARE-demo) |
| 2025 | KDD | [Boosting Bot Detection via Heterophily-Aware Representation Learning and Prototype-Guided Cluster Discovery](https://arxiv.org/abs/2506.00989) | [Code](https://github.com/Peien429/BotHP) |

### Recommendation Systems

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2025 | WWW | [GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems](https://arxiv.org/abs/2412.17245) | [Code](https://github.com/snap-research/GraphHash) |
| 2024 | NeurIPS | [End-to-end Learnable Clustering for Intent Learning in Recommendation](https://arxiv.org/abs/2401.05975) | [Code](https://github.com/yueliu1999/ELCRec) |

### Graph Condensation and Distillation

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | TKDE | [DeepCGC: Unveiling the Deep Clustering Mechanism of Fast Graph Condensation](https://ieeexplore.ieee.org/document/11359095) | [Code](https://github.com/XYGaoG/DeepCGC) |
| 2025 | KDD | [Simple yet Effective Graph Distillation via Clustering](https://arxiv.org/abs/2505.20807) | [Code](https://github.com/HKBU-LAGAS/ClustGDD) |

### Bioinformatics and Medical Science

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | TPAMI | [Graph-Embedded Deep Generative Clustering for Single-Cell Multi-Omics Data Integration](https://ieeexplore.ieee.org/document/11408922) | [Code](https://github.com/LiangSDNULab/GeDGC) |
| 2025 | NeurIPS | [DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases](https://openreview.net/forum?id=ypPxYsmZPx) | [Code](https://github.com/ncclab-sustech/DCA) |
| 2025 | ICML | [GraphCL: Graph-based Clustering for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2411.13147) | [Code](https://github.com/dreamkily/GraphCL) |

### Natural Language Processing

| Year | Venue | Title | Code |
| ---- | ----- | ----- | ---- |
| 2026 | KDD | [Learning Hierarchical Knowledge in Text-Rich Networks with Taxonomy-Informed Representation Learning](https://arxiv.org/abs/2603.08159) | [Code](https://github.com/Cloudy1225/TIER) |
| 2025 | CIKM | [Cequel: Cost-Effective Querying of Large Language Models for Text Clustering](https://arxiv.org/abs/2504.15640) | [Code](https://github.com/HKBU-LAGAS/Cequel) |

---

## ECO Taxonomy Quick Reference

> The following table summarizes representative methods under the Encode-Cluster-Optimize framework proposed in the survey. For each method, we list the encoder type, cluster projector, coordination pattern, and dominant complexity.

| Method | Venue | Encoder (E) | Cluster (C) | Coordinate (O) | Complexity |
|--------|-------|-------------|-------------|----------------|------------|
| **NP & Decoupled** | | | | | |
| AGC | IJCAI'19 | Simple Filtering | Spectral | Decoupled | O(N²) |
| SSGC | ICLR'21 | Multi-Filtering | K-Means | Decoupled | O(N+M) |
| FGC | SDM'22 | Multi-Filtering | Spectral | Decoupled | O(N²) |
| GRACE | TKDD'22 | Simple Filtering | K-Means | Decoupled | O(N+M) |
| NAFS | ICML'22 | Multi-Filtering | K-Means | Decoupled | O(N+M) |
| SAGSC | AAAI'23 | Subspace-Oriented | Subspace | Decoupled | O(N+M) |
| IAGC | TKDE'23 | Simple Filtering | Spectral | Decoupled | O(N²) |
| SASE | CIKM'24 | Simple Filtering | Spectral | Decoupled | O(N+M) |
| S2CAG | KDD'25 | Subspace-Oriented | Subspace | Decoupled | O(N+M) |
| MS2CAG | KDD'25 | Subspace-Oriented | Subspace | Decoupled | O(N+M) |
| CMV-ND | ICML'25 | Multi-Filtering | K-Means | Decoupled | O(N+M) |
| **Deep Decoupled** | | | | | |
| GAE | NeurIPS-W'16 | GCN | K-Means | Decoupled | O(N²) |
| ARGA | IJCAI'18 | GCN | K-Means | Decoupled | O(N²) |
| DGI | ICLR'19 | GCN | K-Means | Decoupled | O(N+M) |
| MVGRL | ICML'20 | GCN | K-Means | Decoupled | O(N+M) |
| CCA-SSG | NeurIPS'21 | GCN | K-Means | Decoupled | O(N+M) |
| BGRL | ICLR'22 | GCN | K-Means | Decoupled | O(N+M) |
| DCRN | AAAI'22 | Mixed GNN | K-Means | Decoupled | O(N²) |
| S3GC | NeurIPS'22 | GCN | K-Means | Decoupled | O(N²) |
| DGCN | ICML'23 | Mixed GNN | K-Means | Decoupled | O(N+M) |
| NS4GC | TKDE'24 | GCN | K-Means | Decoupled | O(N²) |
| MAGI | KDD'24 | GCN/SAGE | K-Means | Decoupled | O(N²) |
| NeuCGC | TKDE'25 | GCN | K-Means | Decoupled | O(N²) |
| CoCo | ICLR'26 | Mixed GNN | K-Means | Decoupled | O(N+M) |
| **Deep Joint** | | | | | |
| DAEGC | IJCAI'19 | GAT | Prototype | Joint | O(N²) |
| SDCN | WWW'20 | GCN+AE | Prototype | Joint | O(N²) |
| MinCut | ICML'20 | GCN | Softmax | Joint | O(N+M) |
| RGAE | TKDE'22 | GCN | Prototype | Joint | O(N²) |
| DMoN | JMLR'23 | GCN | Softmax | Joint | O(N+M) |
| DinkNet | ICML'23 | GCN | Prototype | Joint | O(N+M) |
| DGCluster | AAAI'24 | GCN | Softmax | Joint | O(N+M) |
| Neuromap | NeurIPS'24 | GCN | Softmax | Joint | O(N+M) |
| LSEnet | ICML'24 | Lorentz GCN | Softmax (Tree) | Joint | O(N²) |
| GCSBM | LoG'25 | GCN | Softmax | Joint | O(N+M) |
| DeSE | KDD'25 | GCN | Softmax | Joint | O(N²) |
| FVD | TPAMI'25 | GCN+VAE | Prototype | Joint | O(N²) |
| ASIL | TPAMI'26 | Lorentz GCN | Softmax (Tree) | Joint | O(N²) |
| **Hybrid Coord.** | | | | | |
| CLEAR | TIST'23 | GCN | K-Means | Hybrid | O(N+M) |
| HoLe | CIKM'23 | GCN | K-Means | Hybrid | O(N²) |
| HSAN | AAAI'23 | GCN | K-Means | Hybrid | O(N²) |
| CCGC | AAAI'23 | GCN | K-Means | Hybrid | O(N²) |
| RGC | MM'23 | GCN | K-Means | Hybrid | O(N²) |
| CARL-G | KDD'23 | GCN | Softmax | Hybrid | O(N+M) |
| NeuroCUT | KDD'24 | GNN | Softmax | Hybrid | O(N+M) |
| FastDGC | TKDD'24 | GCN | K-Means | Hybrid | O(N²) |
| DGAC | WWW'25 | Mixed GNN | K-Means | Hybrid | O(N²) |
| RAGC | NeurIPS'25 | Mixed GNN | K-Means | Hybrid | O(N²) |

---

## Community Resources

> A collection of actively maintained repositories, benchmarks, and reading lists related to attributed graph clustering and deep graph clustering.

| Resource                                                     | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [PyAGC Reading List](https://github.com/Cloudy1225/PyAGC/blob/main/AWESOME_AGC.md) | Curated paper list accompanying the PyAGC benchmark, covering AGC methods, datasets, and applications |
| [PyAGC](https://github.com/Cloudy1225/PyAGC)                 | Production-ready benchmark library for attributed graph clustering with standardized implementations, mini-batch support, and industrial-scale evaluation |
| [Awesome Deep Graph Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) | Comprehensive collection of deep graph clustering papers, codes, and datasets, accompanying [this survey](https://arxiv.org/abs/2211.12875) |
| [PyDGC](https://github.com/Marigoldwu/PyDGC)                 | Open-source Python library for deep graph clustering, accompanying the DGCBench benchmark with unified training and evaluation paradigms |
| [BenchTGC](https://github.com/MGitHubL/BenchTGC)             | Benchmark repository for temporal graph clustering with curated datasets and standardized evaluation frameworks |

---

## Citation

If you find this list useful, please consider citing our survey and benchmark papers:

```bibtex
@article{liu2026bridging,
  title={Bridging Academia and Industry: A Comprehensive Benchmark for Attributed Graph Clustering},
  author={Yunhui Liu and Pengyu Qiu and Yu Xing and Yongchao Liu and Peng Du and Chuntao Hong and Jiajun Zheng and Tao Zheng and Tieke He},
  year={2026},
  eprint={2602.08519},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
