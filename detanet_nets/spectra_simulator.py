# Adapted from DetaNet: Zou et al., Nat. Comput. Sci. 3, 957-964 (2023).
# https://doi.org/10.1038/s43588-023-00550-y  (MIT License, (c) 2023 Z. Zou, W. Hu)
# Modifications in SENK: NaN guard and numerical stabilization in `hessfreq`,
# sign-safe sqrt for imaginary modes, removed unused `sers` parameter.

import torch
from .constant import *
from .model_loader import *
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

def hessfreq(Hi,Hij,edge_index,masses,normal=False,linear=False,scale=0.965):
    '''Combine the Hessian matrices of the Hi and Hij parts and calculate the frequencies and normal coordinates'''
    wmasses=torch.pow(masses.to(Hi.device),-0.5).repeat_interleave(3, dim=0)
    wmat=wmasses[:,None]*wmasses[None,:]

    i, j = edge_index
    dia=torch.arange(0,len(Hi),step=1,device=Hi.device,dtype=int)
    hessian=torch.zeros(size=(Hi.shape[0],3,Hi.shape[0],3),device=Hi.device)
    hessian[j,:,i,:]=Hij
    hessian[dia,:,dia,:]=Hi
    hessian=hessian.reshape(Hi.shape[0]*3,Hi.shape[0]*3)

    hessian=(hessian+hessian.permute(1,0))/2

    hessian=hessian*wmat

# NaN guard: replace any NaN/Inf in the raw Hessian with zero
    if not torch.isfinite(hessian).all():
        hessian = torch.nan_to_num(hessian, nan=0.0, posinf=0.0, neginf=0.0)

# Numerical stabilisation for ill-conditioned mass-weighted Hessians
    # When the model predicts near-zero force constants for soft torsions
    # (e.g. S–N / SO2 groups in sulfonamides), the mass-weighted matrix can
    # become nearly singular.  A tiny relative diagonal shift prevents
    # torch.linalg.eigh convergence failures at a level far below any
    # physically meaningful frequency (< 0.01 cm⁻¹).
    _dmean = hessian.diagonal().abs().mean()
    _shift0 = _dmean * 1e-6 if _dmean > 0 else 1e-12
    hessian = hessian + torch.eye(hessian.size(0), device=hessian.device, dtype=hessian.dtype) * _shift0

    try:
        eva, evec=torch.linalg.eigh(hessian)
    except torch._C._LinAlgError:
        _shift1 = _dmean * 1e-4 if _dmean > 0 else 1e-10
        hessian = hessian + torch.eye(hessian.size(0), device=hessian.device, dtype=hessian.dtype) * _shift1
        eva, evec=torch.linalg.eigh(hessian)
    eva = eva * hess_t
    # Use sign-safe sqrt: negative eigenvalues (imaginary modes) become negative frequencies
    # instead of NaN, allowing callers to filter them cleanly.
    freq = torch.sign(eva) * torch.sqrt(torch.abs(eva)) / (2 * torch.pi)
    freq=freq/cm_hz
    p=-evec.t()*wmasses
    normals = torch.norm(p, dim=1).unsqueeze(1)
    if normal:
        p=p/normals
    p=p.reshape(len(freq),-1,3)
    if scale is not None:
        freq=freq*scale
    if linear:
        return freq[5:],p[5:]
    else:
        return freq[6:],p[6:]


def chain_rule_ir(dd,modes):
    '''Calculation of infrared intensity by the chain rule'''
    irs=(modes[...,None]*dd[None,:,:,:]).reshape(modes.shape[0],-1,3)
    irxyz=torch.sum(irs,dim=1)
    ir=irxyz.norm(dim=-1)**2
    return ir*ir_coff

def chain_rule_raman(dp,modes):
    '''Calculation of raman tensor by the chain rule'''
    ramans=(modes[...,None]*dp[None,:,:,:]).reshape(modes.shape[0],-1,6)
    raman_tensor=torch.sum(ramans,dim=1)
    return raman_tensor

def get_raman_act(raman_tensor):
    '''Calculation of Raman activity from the Raman tensor'''
    xx=raman_tensor[:,0]
    xy=raman_tensor[:,1]
    yy=raman_tensor[:,2]
    xz=raman_tensor[:,3]
    zy=raman_tensor[:,4]
    zz=raman_tensor[:,5]
    alpha=(xx+yy+zz)/3
    gamma_sq1=0.5*(((xx-yy)**2)+((yy-zz)**2)+((zz-xx)**2))
    gamma_sq2=3*((xy**2)+(xz**2)+(zy**2))
    gamma_sq=gamma_sq1+gamma_sq2
    return (45*(alpha**2)+7*gamma_sq)*raman_coff

def get_raman_intensity(freq,raman_act,temp=298,init_wl=532):
    '''Raman intensity from Raman activity and frequency'''
    init_freq_si=cm_hz/ (init_wl * 1e-7)
    freq_si=freq.to(torch.float64)*cm_hz
    act_si=raman_act.to(torch.float64)*(A_m**4)/45
    last=1/(1-torch.exp(-hp*c*freq_si/(Kb*temp)))
    dv=((init_freq_si-freq_si)**4)/freq_si
    return act_si*dv*last

class nn_vib_analysis(torch.nn.Module):
    '''Complete IR and Raman simulations by coordinates and atomic types.'''
    def __init__(self,device,Linear=False,scale=0.965):
        super(nn_vib_analysis,self).__init__()
        self.device=device
        self.linear=Linear
        self.scale=scale
        self.model_Hi=Hi_model(device=device)
        self.model_Hij=Hij_model(device=device)
        self.model_dd=dedipole_model(device=device)
        self.model_dp=depolar_model(device=device)

    def forward(self,pos,z,edge_index=None,batch=None):
        pos=pos.to(self.device)
        z=z.to(self.device)
        if batch is not None:
            batch=batch.to(self.device)
        if edge_index is None:
            edge_index=radius_graph(x=pos,r=5.0,batch=batch)
        Hi= self.model_Hi(pos=pos,z=z,batch=batch)
        Hij= self.model_Hij(pos=pos,z=z,edge_index=edge_index,batch=batch)
        dd= self.model_dd(pos=pos,z=z,batch=batch)
        dp= self.model_dp(pos=pos,z=z,batch=batch)
        if batch is None:
            freq,modes=hessfreq(Hi=Hi,Hij=Hij,masses=atom_masses[z], edge_index=edge_index, normal=False
                            , linear=self.linear,scale=self.scale)
            ir_int=chain_rule_ir(dd=dd,modes=modes)
            raman_act=get_raman_act(chain_rule_raman(dp=dp,modes=modes))
            return freq,ir_int,raman_act
        else:
            vib_list = []
            atom_num=0
            for n in range(0,batch.max()):
                bat_edge_index_mid = edge_index[:,edge_index[1]>=atom_num]
                Hij_m=Hij[edge_index[1]>=atom_num]
                atom_num=atom_num+len(batch[batch==n])
                bat_edge_index = bat_edge_index_mid[:,bat_edge_index_mid[1] < atom_num]
                Hij_o = Hij_m[bat_edge_index_mid[1] < atom_num]
                freq, modes = hessfreq(Hi=Hi[batch==n], Hij=Hij_o,
                                       masses=atom_masses[z[batch==n]], edge_index=bat_edge_index-bat_edge_index.min()
                                       , normal=False, linear=self.linear, scale=self.scale)
                ir_int = chain_rule_ir(dd=dd[batch==n], modes=modes)
                raman_act = get_raman_act(chain_rule_raman(dp=dp[batch==n], modes=modes))
                vib_list.append([freq,ir_int, raman_act])
            return vib_list

class nmr_calculator(torch.nn.Module):
    def __init__(self,device,refc=187.653,refh=31.751):
        super(nmr_calculator, self).__init__()
        self.device=device
        self.nmrc_model=nmr_model(device,params='trained_param/qm9nmr/shield_iso_c.pth')
        self.nmrh_model=nmr_model(device,params='trained_param/qm9nmr/shield_iso_h.pth')
        self.refc=refc
        self.refh=refh

    def forward(self,pos,z,batch=None):
        sc=self.nmrc_model(z=z,pos=pos,batch=batch)[z==6]
        sh=self.nmrh_model(z=z,pos=pos,batch=batch)[z==1]
        return -sc+self.refc,-sh+self.refh

def nmr_sca(nc,nh,indexc,indexh):
    shiftc=scatter(src=nc,index=indexc,dim=-1,reduce='mean')
    shifth=scatter(src=nh,index=indexh,dim=-1,reduce='mean')
    intc=scatter(src=torch.ones_like(nc),index=indexc,dim=-1,reduce='sum')
    inth=scatter(src=torch.ones_like(nh),index=indexh,dim=-1,reduce='sum')
    return shiftc,intc,shifth,inth

def Lorenz_broadening(x0, y0,c=torch.linspace(500, 4000, 3501), sigma=12):
    '''Lorenz broadening for Vibration and NMR spectroscopies'''
    lx= x0[:,None]-c[None,:]
    ly= (sigma/(2*3.1415926))/(lx**2 + 0.25*(sigma**2))
    y= torch.sum(y0[:,None]*ly,dim=0)
    return y.view(-1)
