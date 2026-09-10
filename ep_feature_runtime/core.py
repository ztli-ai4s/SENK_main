"""Frozen inference-only numerical kernel extracted from the validated feature suite."""
import numpy as np
VERSION = 1
FREQ_FACTOR = np.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (2*np.pi*2.99792458e10)
POLICY = {'version': 1, 'kernel': 'spring', 'sulfur': 0.03, 'other': 0.03, 'cap': 0.05, 'evidence_scale': 0.35, 'non_target': 0.0, 'scope': 'xh'}


def rigid_basis(pos, masses):
    pos = np.asarray(pos, float)
    masses = np.asarray(masses, float)
    center = np.average(pos, axis=0, weights=masses)
    r = pos-center
    axes = np.eye(3)
    cols = [(np.broadcast_to(a, r.shape)*np.sqrt(masses)[:, None]).ravel() for a in axes]
    cols += [(np.cross(a, r)*np.sqrt(masses)[:, None]).ravel() for a in axes]
    u, s, _ = np.linalg.svd(np.column_stack(cols), full_matrices=True)
    rank = int(np.sum(s > max(s[0], 1.)*1e-10))
    return u[:, :rank], u[:, rank:]

def dynamical(h, masses):
    w = np.repeat(np.asarray(masses, float)**-.5, 3)
    return np.asarray(h, float)*w[:, None]*w[None, :]

def signed_freq(evals):
    return np.sign(evals)*np.sqrt(np.abs(evals))*FREQ_FACTOR

def modes(h, pos, masses):
    rigid, vib = rigid_basis(pos, masses)
    d = dynamical((h+h.T)*.5, masses)
    values, vectors = np.linalg.eigh(vib.T@d@vib)
    return signed_freq(values), vib@vectors

def assemble_hessian(hi, hij, edges):
    n = len(hi)
    h = np.zeros((n, 3, n, 3))
    i, j = np.asarray(edges, int)
    if np.any(i == j) or len(set(zip(i.tolist(), j.tolist()))) != len(i):
        raise ValueError('Response graph must have unique non-self directed edges')
    h[j, :, i, :] = hij
    h[np.arange(n), :, np.arange(n), :] = hi
    return h.reshape(3*n, 3*n)

def pair_actions(delta_hij, edges, comp, n):
    """Reconstruct native local actions INCLUDING its actual Hii compensation.

    Uses both directed blocks; no double counting of an undirected bond.
    Returned blocks have shape 6x6 and sum exactly to the native full delta.
    """
    grouped = {}
    for k, (i, j) in enumerate(np.asarray(edges).T):
        grouped.setdefault(tuple(sorted((int(i), int(j)))), []).append(k)
    for (i, j), ids in sorted(grouped.items()):
        block = np.zeros((2, 3, 2, 3))
        for k in ids:
            src, dst = np.asarray(edges)[:, k]
            a, b = int(src == j), int(dst == j)
            block[b, :, a, :] += delta_hij[k]
            block[b, :, b, :] -= comp*delta_hij[k]
        block = block.reshape(6, 6)
        indices = np.r_[np.arange(3*i, 3*i+3), np.arange(3*j, 3*j+3)]
        yield i, j, ids, indices, (block+block.T)*.5

def spectra(h, capture, masses, scale=.965):
    f, v = modes(h, capture['pos'], masses)
    cart = (v * np.repeat(masses**-.5, 3)[:, None]).T.reshape(len(f), len(masses), 3)
    dip = np.einsum('mni,nij->mj', cart, capture['dd'])
    ir = (dip*dip).sum(axis=1)*42.2561
    alpha = np.einsum('mni,nij->mj', cart, capture['dp'])
    xx, xy, yy, xz, yz, zz = alpha.T
    iso = (xx+yy+zz)/3
    anis = .5*((xx-yy)**2+(yy-zz)**2+(zz-xx)**2)+3*(xy*xy+xz*xz+yz*yz)
    return f*scale, v, ir, (45*iso*iso+7*anis)*.021958718449

def nearest_acceptor(z, pos, bonds, x, hydrogen):
    excluded = {x, hydrogen}
    for i, j in bonds.T:
        if i == hydrogen: excluded.add(int(j))
        if j == hydrogen: excluded.add(int(i))
    candidates = []
    hx = pos[x]-pos[hydrogen]
    for k in range(len(z)):
        if k in excluded or z[k] not in (7, 8, 16): continue
        ha = pos[k]-pos[hydrogen]
        distance = float(np.linalg.norm(ha))
        if not .7 < distance <= 3.2: continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(hx, ha)/np.linalg.norm(hx)/distance, -1, 1))))
        if angle >= 85: candidates.append((distance, -angle, k))
    if not candidates: return -1, 0., 0.
    distance, negative_angle, k = min(candidates)
    return k, distance, -negative_angle

def build_actions(c, masses, route='production_interaction_rule', scale=.965):
    """Predictive eligibility uses geometry, NBO evidence and baseline modes only.

    A historical capture may lack the model-selected acceptor. Geometry fallback
    is recorded and disqualifies it from exact online provenance claims.
    """
    if route+'__delta_hij' not in c: raise ValueError('Missing action route: '+route)
    h0 = .5*(c['baseline__H']+c['baseline__H'].T)
    f, v, ir, ra = spectra(h0, c, masses, scale)
    z, pos = c['z'], c['pos']
    cart = v*np.repeat(masses**-.5, 3)[:, None]
    total_cart = np.sum(cart*cart, axis=0).clip(1e-20)
    band = (f >= 2200) & (f <= 4200)
    prominence = np.maximum(ir/max(ir[band].max(), 1e-20), ra/max(ra[band].max(), 1e-20)) if band.any() else np.zeros_like(f)
    src, dst = c[route+'__edge_src'], c[route+'__edge_dst']
    bonds = c['nbo__atom_bond_local']
    records = []
    native = {(i, j): block for i, j, _, _, block in pair_actions(c[route+'__delta_hij'], c['edge_index'], float(c[route+'__comp_strength']), len(z))}
    seen = set()
    for i0, j0 in bonds.T:
        i, j = sorted((int(i0), int(j0)))
        if (i,j) in seen: continue
        seen.add((i,j))
        if not ((z[i] == 1) ^ (z[j] == 1)): continue
        hydrogen, x = (i,j) if z[i] == 1 else (j,i)
        if z[x] not in (7, 8): continue
        mask = ((src == i)&(dst == j)) | ((src == j)&(dst == i))
        if not mask.any(): continue
        def edge(key, default=np.nan):
            value = c.get(route+'__edge_'+key)
            return float(np.mean(value[mask])) if value is not None else default
        evidence = edge('xh_evidence', 0.)
        acceptor = int(edge('acceptor', -1))
        source = 'model_selected'
        if acceptor < 0:
            acceptor, distance, angle = nearest_acceptor(z,pos,bonds,x,hydrogen)
            source = 'geometry_fallback'
        else:
            hx, ha = pos[x]-pos[hydrogen], pos[acceptor]-pos[hydrogen]
            distance = float(np.linalg.norm(ha))
            angle = float(np.degrees(np.arccos(np.clip(np.dot(hx,ha)/np.linalg.norm(hx)/distance,-1,1))))
        if acceptor < 0 or evidence < .15: continue
        t = pos[j]-pos[i]; t /= np.linalg.norm(t)
        b = np.zeros(len(z)*3); b[3*i:3*i+3] = -t; b[3*j:3*j+3] = t
        projection = b@cart
        participation = projection**2/(2*total_cart)
        eligible = band & (prominence >= .05) & (participation >= .015)
        if not eligible.any(): continue
        mode = int(np.argmax(np.where(eligible, participation, -1)))
        # Relative-displacement curvature, eV/A^2; positive only.
        stiffness = float(b@h0@b/4)
        if stiffness <= 0: continue
        idx = np.r_[np.arange(3*i,3*i+3),np.arange(3*j,3*j+3)]
        row = dict(i=i,j=j,x=x,hydrogen=hydrogen,acceptor=acceptor,
                   acceptor_z=int(z[acceptor]),donor_z=int(z[x]),acceptor_source=source,
                   distance=distance,angle=angle,evidence=evidence,stiffness=stiffness,
                   mode=mode,frequency=float(f[mode]),participation=float(participation[mode]),
                   prominence=float(prominence[mode]),support=edge('p_support'),
                   idx=idx,b=b,native=native.get((i,j),np.zeros((6,6))),
                   native_comp=float(c[route+'__comp_strength']))
        for key in ('k','q','r','innovation_rms','update_rms','delta_abs_mean','state_rms'):
            value = c.get('enk__hij__l0__'+key)
            row['enk_'+key] = float(np.mean(value[[i,j]])) if value is not None else np.nan
        records.append(row)
    return records

def action_blocks(actions, policy):
    if policy['version'] != VERSION: raise ValueError('Unsupported policy version')
    if policy['kernel'] not in ('spring','native'): raise ValueError('Unknown action kernel')
    if min(policy['sulfur'],policy['other'],policy['cap']) < 0 or policy['cap'] > .2:
        raise ValueError('Invalid local softening bounds')
    result = []
    for row in actions:
        if policy.get('scope','xh') == 'oh' and row['donor_z'] != 8:
            result.append(np.zeros((6,6)));continue
        gain = policy['sulfur'] if row['acceptor_z'] == 16 else policy['other']
        if policy['kernel'] == 'spring':
            raw = gain*(1-np.exp(-row['evidence']/policy['evidence_scale']))
            fraction = policy['cap']*np.tanh(raw/max(policy['cap'],1e-20))
            local_b = row['b'][row['idx']]
            block = -fraction*row['stiffness']*np.outer(local_b,local_b)
        else:
            block = gain*row['native']
            b = row['b'][row['idx']]
            ratio = abs(float(b@block@b/4))/row['stiffness']
            block = block*min(1.,policy['cap']/max(ratio,1e-20))
        result.append(block)
    return result

def apply_policy(c, actions, policy, weights=None):
    h = .5*(c['baseline__H']+c['baseline__H'].T)
    blocks = action_blocks(actions,policy)
    weights = np.ones(len(blocks)) if weights is None else np.asarray(weights,float)
    if weights.shape != (len(blocks),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Invalid action weights')
    for row,block,weight in zip(actions,blocks,weights):
        b=row['b'][row['idx']]
        curvature=abs(float(b@block@b/4))
        weight=min(float(weight),policy['cap']*row['stiffness']/max(curvature,1e-20))
        h[np.ix_(row['idx'],row['idx'])] += weight*block
    return h
