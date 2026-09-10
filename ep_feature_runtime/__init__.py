"""Feature EP integration; imported only when a calibrator is constructed."""
import json
from pathlib import Path


def make_calibrator(args, **legacy_kwargs):
    if getattr(args, 'ep_output_policy', 'legacy') != 'feature_spring':
        from nbo_spectral_calibration import NBOGuidedCalibrator
        return NBOGuidedCalibrator(**legacy_kwargs)
    import torch
    from .evidence import NBOGuidedCalibrator
    from .support import load_support_artifact
    settings = json.loads((Path(__file__).parent/'settings.json').read_text())
    files = {key:Path(getattr(args, name)) for key,name in
             [('nbo_train_stats','nbo_train_stats'),('support_stats','ep_support_stats'),('support_artifact','ep_support_artifact')]}
    import hashlib
    for key,path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f'Feature EP requires {key}: {path}. See EP_FEATURE_RELEASE.md')
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings['artifact_hashes'][key]:
            raise ValueError('Feature EP statistics version mismatch: '+str(path))
    cal = NBOGuidedCalibrator(**settings['evidence_kwargs'], train_stats=legacy_kwargs['train_stats'])
    cal.feature_support_stats = torch.load(files['support_stats'], map_location='cpu', weights_only=True)
    cal.feature_support_artifact = load_support_artifact(str(files['support_artifact']))
    cal.feature_ep = True
    return cal


def apply_feature(cal, Hi, Hij, dd, dp, nbo, edges, z, pos, masses, context, pre_freq, pre_modes, skeleton, scale):
    import numpy as np
    import torch
    import types
    from . import core, support
    from .evidence import NBOGuidedCalibrator, _match_undirected_edges_simple
    def array(v):
        return v.detach().cpu().numpy()
    observed = {}
    original = cal._hij_xh_interaction_softening_signal
    def observe(self, *args, **kwargs):
        result = original(*args, **kwargs)
        details = kwargs.get('hbond_details', args[7] if len(args)>7 else {})
        observed['xh_evidence'] = array(result[0])
        if 'interaction_acceptor_atom' in details:
            observed['acceptor'] = array(details['interaction_acceptor_atom'])
        return result
    cal._hij_xh_interaction_softening_signal = types.MethodType(observe, cal)
    try:
        # Generate frozen evidence only: discard the legacy Hessian proposal.
        cal.calibrate_hij(Hij, nbo, edges, len(z), pos=pos, z=z,
                          enk_ood_score=context.get('atom_ood'), enk_context=context,
                          mode_freq=pre_freq, modes=pre_modes, skeleton_data=skeleton)
    finally:
        cal._hij_xh_interaction_softening_signal = original
    _, mask = _match_undirected_edges_simple(edges, nbo['atom_bond_local'].to(edges.device), len(z))
    src,dst = edges[:,mask]
    settings = json.loads((Path(__file__).parent/'settings.json').read_text())
    sc = NBOGuidedCalibrator(**settings['evidence_kwargs'], train_stats=cal.feature_support_stats)
    classes = sc._hij_bond_context(z,nbo['atom_bond_local'].to(z.device),src,dst,pos=pos)[0]
    features = support.compute_four_level_support(sc,z,nbo['atom_bond_local'].to(z.device),src,dst,classes)
    if not bool((features['support_stats_available']>.5).all()):
        raise ValueError('Feature EP requires schema-v2 unique-molecule support counts')
    ps = support.apply_support_artifact(features,cal.feature_support_artifact)
    route='production_interaction_rule'
    c=dict(z=array(z),pos=array(pos),edge_index=array(edges),dd=array(dd),dp=array(dp),
           baseline__H=core.assemble_hessian(array(Hi),array(Hij),array(edges)))
    for key,value in nbo.items():
        if isinstance(value,torch.Tensor):c['nbo__'+key]=array(value)
    c[route+'__delta_hij']=np.zeros_like(array(Hij))
    c[route+'__comp_strength']=np.asarray(1.)
    c[route+'__edge_src'],c[route+'__edge_dst']=array(src),array(dst)
    c[route+'__edge_p_support']=array(ps)
    for key,value in observed.items():c[route+'__edge_'+key]=value
    if mask.any() and 'xh_evidence' not in observed:
        raise ValueError('Feature EP evidence was not generated')
    actions=core.build_actions(c,array(masses).astype(float),route,scale)
    h=core.apply_policy(c,actions,core.POLICY)
    if not np.isfinite(h).all():raise ValueError('Nonfinite feature EP Hessian')
    delta=h-.5*(c['baseline__H']+c['baseline__H'].T)
    w=np.repeat(array(masses).astype(float)**-.5,3)
    dw=delta*w[:,None]*w[None,:]
    rigid,_=core.rigid_basis(c['pos'],array(masses))
    leakage=float(np.linalg.norm(dw@rigid)/max(np.linalg.norm(dw),1e-20))
    if leakage>1e-6:raise ValueError('Feature EP rigid-mode constraint failed')
    n=len(z);tensor=torch.as_tensor(h.reshape(n,3,n,3),device=Hi.device,dtype=Hi.dtype)
    ids=torch.arange(n,device=Hi.device)
    new_hi,new_hij=tensor[ids,:,ids,:],tensor[edges[1],:,edges[0],:]
    rebuilt=core.assemble_hessian(array(new_hi),array(new_hij),array(edges))
    if not np.allclose(rebuilt,h,atol=2e-5,rtol=1e-6):raise ValueError('Feature EP response graph cannot represent corrected Hessian')
    cal.last_hij_delta=new_hij-Hij
    cal.last_hij_debug=dict(policy='feature_spring',active_edges=len(actions),increment_rigid_leakage=leakage)
    cal.feature_ep_diagnostics=dict(policy=core.POLICY,support_role='diagnostic_only',output_risk_head='disabled',
        actions=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k not in ('b','idx','native') and not k.startswith('enk_')} for r in actions],
        support_features={k:array(v).tolist() for k,v in features.items() if isinstance(v,torch.Tensor)},
        p_insufficient_support=array(ps).tolist(),increment_rigid_leakage=leakage)
    return new_hi,new_hij
