"""DetaNet interaction modules, adapted from the DetaNet project.

Reference: Zou et al., Nat. Comput. Sci. 3, 957-964 (2023).
https://doi.org/10.1038/s43588-023-00550-y  (MIT License, (c) 2023 Z. Zou, W. Hu)
"""

from .edge_attention import Edge_Attention
from .embedding import Embedding
from .radial_basis import Radial_Basis
from .message import Message
from .update import Update
from .block import Interaction_Block
from .multilayer_perceptron import MLP,Equivariant_Multilayer
from .acts import activations
