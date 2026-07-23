"""DetaNet model components used as the Hessian/spectral backbone in SENK.

This package is adapted from the DetaNet project:

    Zihan Zou, Yujin Zhang, Lijun Liang, Mingzhi Wei, Jiancai Leng,
    Jun Jiang, Yi Luo, Wei Hu. "A deep learning model for predicting
    selected organic molecular spectra." Nature Computational Science
    3(11): 957-964, 2023.  https://doi.org/10.1038/s43588-023-00550-y

Original source: Copyright (c) 2023 Zihan Zou, Wei Hu (MIT License).
The core modules (`detanet.py`, `spectra_simulator.py`, `model_loader.py`,
`constant.py`, `metrics.py`, and `modules/`) are derived from DetaNet;
SENK-specific extensions such as `electron_prior.py` and the additional
model loaders are implemented in this project.
"""

from .detanet import DetaNet
from .spectra_simulator import *
from .model_loader import *
from .constant import *
from .metrics import *
