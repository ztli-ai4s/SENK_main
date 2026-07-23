# Adapted from DetaNet: Zou et al., Nat. Comput. Sci. 3, 957-964 (2023).
# https://doi.org/10.1038/s43588-023-00550-y  (MIT License, (c) 2023 Z. Zou, W. Hu)
# Modifications in SENK: integration with NBOPriorBranch electron prior.

from typing import Optional, Union

import torch
from e3nn import o3,io
from torch import nn,FloatTensor
from .constant import atom_masses
from torch_geometric.nn import radius_graph
from .modules import Interaction_Block,Embedding,Radial_Basis,MLP,Equivariant_Multilayer
from .electron_prior import NBOPriorBranch
from torch.autograd import grad
from torch_scatter import scatter
"""Deep equivariant tensor attention Network(DetaNet) graph neural network model"""

class DetaNet(nn.Module):
    def __init__(self,num_features:int=128,
                 act:str='swish',
                 maxl:int=3,
                 num_block:int=3,
                 radial_type:str='trainable_bessel',
                 num_radial:int=32,
                 attention_head:int=8,
                 rc:float=5.0,
                 dropout:float=0.0,
                 use_cutoff:bool=False,
                 max_atomic_number:int=9,
                 atom_ref:FloatTensor=None,
                 scale:float=1.0,
                 scalar_outsize:int=1,
                 irreps_out: Optional[Union[str, o3.Irreps]] = None,
                 summation:bool=True,
                 norm:bool=False,
                 out_type:str='scalar',
                 grad_type:str=None,
                 electron_prior_config: Optional[dict] = None,
                 device:torch.device=torch.device('cuda')):
        super(DetaNet,self).__init__()
        assert num_features%attention_head==0,'attention head must be divisible by the number of features'
        self.scale=scale
        self.ref=atom_ref
        self.norm=norm
        self.summation=summation
        self.scalar_outsize=scalar_outsize
        self.out_type=out_type
        self.rc=rc
        self.grad=grad_type
        self.electron_prior = None
        # Generate the representation of irrep features.
        irreps_T = o3.Irreps((num_features, (l, (-1) ** l)) for l in range(1, maxl + 1))
        self.vdim = o3.Irreps(irreps_T).dim
        self.features=num_features
        self.T=num_block
        self.irreps_out=irreps_out
        # Generate the spherical harmonic function.
        irrs_sh=o3.Irreps.spherical_harmonics(lmax=maxl, p=-1)
        # Removal of scalars with l=0
        self.irreps_sh=irrs_sh[1:]
        self.Embedding=Embedding(num_features=num_features,act=act,device=device,max_atomic_number=max_atomic_number)
        self.Radial=Radial_Basis(radial_type=radial_type,num_radial=num_radial,use_cutoff=use_cutoff)
        blocks = []
        # interaction layers
        for _ in range(num_block):
            block=Interaction_Block(num_features=num_features,
                        act=act,
                        head=attention_head,
                        num_radial=num_radial,
                        irreps_sh=self.irreps_sh,
                        irreps_T=irreps_T,
                        dropout=dropout
                        )
            blocks.append(block)
        self.blocks=nn.Sequential(*blocks)
        # generate output layer
        if irreps_out is not None:
            mid = []
            for _, (l, p) in o3.Irreps(irreps_out):
                mid.append((num_features, (l, p)))
            irreps_mid = o3.Irreps(mid)
            self.tout=Equivariant_Multilayer(irreps_list=[irreps_T,irreps_mid,irreps_out],act=act)
        if scalar_outsize !=0:
            self.sout=MLP(size=(num_features,num_features,scalar_outsize),act=act,dropout=dropout)

        self.register_buffer("mass", atom_masses.to(device), persistent=False)
        #Module for conversion from irrep tensor to Cartesian tensor
        if out_type == '2_tensor':
            self.ct = io.CartesianTensor('ij=ji')

        elif out_type == '3_tensor':
            self.ct = io.CartesianTensor("ijk=jik=ikj")

        if self.grad=='polar':
            self.mask=torch.tril(torch.ones(size=(3,3)),diagonal=0).flatten()

        if electron_prior_config is not None and str(electron_prior_config.get('mode', 'off')).lower() != 'off':
            self.electron_prior = NBOPriorBranch(
                num_features=num_features,
                num_radial=num_radial,
                mode=electron_prior_config['mode'],
                checkpoint_path=electron_prior_config['checkpoint_path'],
                stats_path=electron_prior_config.get('stats_path'),
                hidden_dim=electron_prior_config.get('hidden_dim', num_features),
                max_atomic_number=electron_prior_config.get('max_atomic_number', max_atomic_number),
                feature_scale=electron_prior_config.get('feature_scale', 1e-2),
                use_auxiliary=electron_prior_config.get('use_auxiliary', True),
                freeze_predictor=electron_prior_config.get('freeze_predictor', True),
                runtime_mode=electron_prior_config.get('runtime_mode', 'full'),
            )

    def centroid_coordinate(self, z, pos, batch):
        '''Calculate the centre-of-mass coordinates of each atom.'''
        mass = self.mass[z].view(-1, 1)
        if batch is None:
            c = torch.sum(pos * mass, dim=0) / torch.sum(mass, dim=0)
            ra = (pos - c)
        else:
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            ra = (pos - c[batch])
        return ra

    def cal_dipole(self,z,pos,batch,outs,outt):
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        return outs*ra+outt

    def cal_p_tensor(self,z,pos,batch,outs,outt):
        sa,sb=torch.split(outs,dim=-1,split_size_or_sections=[1,1])
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        ta=o3.spherical_harmonics(l=self.irreps_out,x=ra,normalize=False)*sa
        return self.ct.to_cartesian(torch.concat(tensors=(sb,outt+ta),dim=-1))

    def cal_R_sq(self,z,pos,batch,outs):
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        return ((torch.norm(ra, p=2, dim=-1, keepdim=True) ** 2) * outs).reshape(-1)

    def cal_3_p_tensor(self,z,pos,batch,outs,outt):
        sa, sb = torch.split(outs, dim=-1, split_size_or_sections=[1, 1])
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        ta=o3.spherical_harmonics(l='1o',x=ra,normalize=False)*sa
        tb=o3.spherical_harmonics(l='3o',x=ra,normalize=False)*sb
        return self.ct.to_cartesian(outt+torch.concat(tensors=(ta,tb),dim=-1))

    def grad_hess_ij(self, energy, posj, posi, create_graph=True):
        fj = -grad([torch.sum(energy)], [posj], create_graph=create_graph)[0]
        Hji = torch.zeros((fj.shape[0], 3, 3), device=fj.device)
        for i in range(3):
            gji = -grad([fj[:, i].sum()], [posi], create_graph=create_graph, retain_graph=True)[0]
            Hji[:, i] = gji
        return Hji

    def grad_hess_ii(self, energy, posa, posb, create_graph=True):
        f = -grad([torch.sum(energy)], [posa], create_graph=create_graph)[0]
        Hii = torch.zeros((f.shape[0], 3, 3), device=f.device)
        for i in range(3):
            gii = -grad([f[:, i].sum()], [posb], create_graph=create_graph, retain_graph=True)[0]
            Hii[:, i] = gii
        return Hii

    def grad_force(self, energy, pos, create_graph=True):
        force=-grad([torch.sum(energy)], [pos], create_graph=create_graph)[0]
        return force

    def grad_dipole(self, dipole, pos):
        dedipole = torch.zeros(size=(pos.shape[0], 3, 3), device=pos.device)
        for i in range(0, 3):
            dedipole[:, :, i] = -grad([dipole[:, i].sum()], [pos], create_graph=True)[0]
        return dedipole

    def grad_polarzability(self, polars, pos):
        polars = polars.flatten(start_dim=1)[:, self.mask == 1]
        depolar = torch.zeros(size=(pos.shape[0], 3, 6), device=pos.device)
        for i in range(0, 6):
            depolar[:, :, i] = -grad([polars[:, i].sum()], [pos], create_graph=True)[0]
        return depolar

    def forward(self,
                z=None,
                pos=None,
                edge_index=None,
                batch=None,
                data=None):
        if data is not None:
            if z is None:
                z = data.z
            if pos is None:
                pos = data.pos
            if edge_index is None:
                edge_index = getattr(data, 'edge_index', None)
            if batch is None:
                batch = getattr(data, 'batch', None)

        if self.grad is not None:
            pos.requires_grad=True

        if edge_index is None:
            edge_index=radius_graph(x=pos,r=self.rc,batch=batch)

        S=self.Embedding(z)
        T=torch.zeros(size=(S.shape[0],self.vdim),device=S.device,dtype=S.dtype)
        i,j=edge_index

        if self.grad=='Hi':
            posa=pos.clone()
            posb=pos.clone()
            posj = posa[j]
            posi = posb[i]
        else:
            posi=pos[i]
            posj=pos[j]

        rij = posj - posi
        r=torch.norm(rij,dim=-1)

        sh = o3.spherical_harmonics(l=self.irreps_sh, x=rij/(r.view(-1,1)), normalize=True, normalization="component")
        rbf = self.Radial(r)

        edge_prior = None
        if self.electron_prior is not None:
            atom_prior, edge_prior = self.electron_prior(z=z, pos=pos, edge_index=edge_index, rbf=rbf, data=data)
            S = S + atom_prior.to(S.dtype)

        for block in self.blocks:
            S,T=block(S=S,T=T,sh=sh,rbf=rbf,index=edge_index,edge_prior=edge_prior)

        if self.irreps_out is not None:
            outt=self.tout(T)

        if self.scalar_outsize!=0:
            outs=self.sout(S)

        if self.out_type=='scalar':
            out=outs

        elif self.out_type=='dipole':
            out=self.cal_dipole(z=z,pos=pos,batch=batch,outs=outs,outt=outt)

        elif self.out_type=='2_tensor':
            out=self.cal_p_tensor(z=z,pos=pos,batch=batch,outs=outs,outt=outt)

        elif self.out_type=='R2':
            out=self.cal_R_sq(z=z,pos=pos,batch=batch,outs=outs)

        elif self.out_type=='3_tensor':
            out=self.cal_3_p_tensor(z=z,pos=pos,batch=batch,outs=outs,outt=outt)

        elif self.out_type=='latent':
            out=S,T
        else:
            out=outs,outt

        if self.ref is not None:
            out=out+self.ref[z].to(out.device).reshape(-1,1)

        if self.summation:
            if batch is not None:
                out = scatter(src=out, index=batch, dim=0)
            else:
                out = torch.sum(input=out, dim=0)

        if self.grad=='force':
            out=self.grad_force(energy=out,pos=pos)

        elif self.grad=='dipole':
            out=self.grad_dipole(dipole=out.reshape(-1,3),pos=pos)

        elif self.grad=='polar':
            out=self.grad_polarzability(polars=out.reshape(-1,3,3),pos=pos)

        elif self.grad=='Hij':
            out=self.grad_hess_ij(energy=out,posj=posj,posi=posi)

        elif self.grad=='Hi':
            out=self.grad_hess_ii(energy=out,posa=posa,posb=posb)

        if self.norm:
            out=out.norm(dim=-1,keepdim=False)

        if self.scale is not None:
            out=out*self.scale
        return out
