from .base import BaseModel, TrainableModel, ClusteringModel, LossOutput
from .dgi import DGI
from .dmon import DMoN
from .mincut import MinCut
from .neuromap import Neuromap
from .gbt import GBT
from .ccassg import CCASSG
from .ns4gc import NS4GC
from .sagsc import SAGSC
from .sgc import SGC
from .ssgc import SSGC
from .autoencoder import GAE, VGAE, ARGA, ARGVA
from .node2vec import Node2Vec
from .s3gc import S3GC
from .magi import MAGI
from .s2cag import S2CAG, MS2CAG
from .gcsbm import GCSBM
from .dinknet import DinkNet
from .daegc import DAEGC
from .nafs import NAFS

__all__ = [
    'BaseModel',
    'TrainableModel',
    'ClusteringModel',
    'LossOutput',
    'DGI',
    'DMoN',
    'MinCut',
    'Neuromap',
    'GBT',
    'CCASSG',
    'NS4GC',
    'SAGSC',
    'SGC',
    'SSGC',
    'GAE',
    'VGAE',
    'ARGA',
    'ARGVA',
    'Node2Vec',
    'S3GC',
    'MAGI',
    'S2CAG',
    'MS2CAG',
    'GCSBM',
    'DinkNet',
    'DAEGC',
    'NAFS',
]