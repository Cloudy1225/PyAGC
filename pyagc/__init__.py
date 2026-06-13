"""PyAGC: A PyTorch library for Attributed Graph Clustering"""

__version__ = '1.1.1'

# Import main modules
from . import clusters
from . import data
from . import metrics
from . import models
from . import encoders
from . import transforms
from . import utils

__all__ = [
    'clusters',
    'data',
    'metrics',
    'models',
    'encoders',
    'transforms',
    'utils',
]
