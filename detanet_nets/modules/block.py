# Part of DetaNet (Zou et al., Nat. Comput. Sci. 3, 957-964, 2023; MIT License).

from e3nn import o3
from torch import nn
from .message import Message
from .update import Update


class Interaction_Block(nn.Module):
    '''Interaction Block = Message Module + Update Module'''
    def __init__(self, num_features, act, head, num_radial, irreps_sh, irreps_T, dropout=0.0):
        super(Interaction_Block, self).__init__()
        self.message = Message(head=head, num_radial=num_radial, num_features=num_features,
                               irreps_sh=irreps_sh, act=act)
        irreps_mout = []
        for _, ir_sh in irreps_sh:
            for ir_out in o3.Irrep('0e') * ir_sh:
                irreps_mout.append((num_features, ir_out))
        irreps_mout = o3.Irreps(irreps_mout)
        self.update = Update(num_features=num_features, act=act, irreps_mout=irreps_mout,
                             irreps_T=irreps_T, dropout=dropout)

    def forward(self, S, T, sh, rbf, index, edge_prior=None):
        mijt, mijs = self.message(S=S, rbf=rbf, sh=sh, index=index, edge_prior=edge_prior)
        T, S = self.update(T=T, S=S, mijt=mijt, mijs=mijs, index=index)
        return S, T
