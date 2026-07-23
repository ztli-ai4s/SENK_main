import torch
import os
import random
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch_geometric.data as pyg_data


def _torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

class NBODatum(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['interaction_edge_index', 'atom_bond_index', 'atom_to_nbo_index', 'link_prediction_candidates']:
            return 1
        return 0
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['interaction_edge_index', 'atom_bond_index', 'atom_to_nbo_index', 'link_prediction_candidates']:
            # increment by number of nodes in this graph
            return int(self.x.size(0))
        return 0

class NBO_Dataset_LP(Dataset):
    """
    A custom Dataset for NBO pre-training that supports dynamic negative sampling
    for the link prediction task.
    """
    def __init__(self, dataset_path, split_indices=None, neg_to_pos_ratio=1.0):
        super().__init__()
        self.neg_to_pos_ratio = neg_to_pos_ratio
        
        print(f"Loading raw mirrored dataset from {dataset_path}...")
        full_data_list = _torch_load_compat(dataset_path)
        
        if split_indices is not None:
            self.data_list = [full_data_list[i] for i in split_indices]
        else:
            self.data_list = full_data_list
        print(f"Initialized dataset with {len(self.data_list)} molecules.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        molecule_dict = self.data_list[idx]
        
        # --- Mirror process logic ---
        node_type = molecule_dict['node_type']
        y_targets = molecule_dict['y_targets']
        atom_to_bond_map = molecule_dict['atom_to_bond_index']

        # Unify node_type to 0=Atom, 1=Bond, 2=LP
        node_type = torch.as_tensor(node_type).long()
        if (node_type == 3).any():
            node_type = node_type.clone()
            node_type[node_type == 3] = 2
        # If legacy label 2 exists without label 1, split 2 into 1(bond)/2(LP) based on number of connected atoms
        if (node_type == 2).any() and (node_type == 1).sum() == 0:
            if isinstance(atom_to_bond_map, torch.Tensor) and atom_to_bond_map.numel() > 0:
                tgt = atom_to_bond_map[1]
                counts = torch.bincount(tgt, minlength=node_type.shape[0])
                combined_mask = (node_type == 2)
                bond_mask = combined_mask & (counts >= 2)
                lp_mask = combined_mask & (counts < 2)
                node_type = node_type.clone()
                node_type[bond_mask] = 1
                node_type[lp_mask] = 2
        
        # Build atom-atom edge index from a2b mapping (by atom local space)
        bond_nodes = torch.unique(atom_to_bond_map[1]) if isinstance(atom_to_bond_map, torch.Tensor) and atom_to_bond_map.numel() > 0 else torch.tensor([], dtype=torch.long)
        atom_adj = {}
        bond_node_to_atom_pair = {}
        for bond_node_idx in bond_nodes:
            connected_atoms = atom_to_bond_map[0][atom_to_bond_map[1] == bond_node_idx]
            if connected_atoms.numel() == 2:
                u, v = connected_atoms[0].item(), connected_atoms[1].item()
                u_canon, v_canon = (u, v) if u < v else (v, u)
                atom_adj.setdefault(u_canon, set()).add(v_canon)
                bond_node_to_atom_pair[bond_node_idx.item()] = (u_canon, v_canon)
        
        atom_bond_index_list = []
        for u, neighbors in sorted(atom_adj.items()):
            for v in sorted(list(neighbors)):
                atom_bond_index_list.append([u, v])
        atom_bond_index = torch.tensor(atom_bond_index_list, dtype=torch.long).t().contiguous() if atom_bond_index_list else torch.empty((2, 0), dtype=torch.long)
        
        # Extract atom and bond labels
        is_atom_mask = (node_type == 0)
        is_bond_mask = (node_type == 1)
        y_atomic_props = torch.as_tensor(y_targets)[is_atom_mask, :4]
        
        bond_node_indices = torch.where(is_bond_mask)[0]
        bond_node_targets = torch.as_tensor(y_targets)[is_bond_mask, 9:11]
        bond_node_idx_to_target = {idx.item(): target for idx, target in zip(bond_node_indices, bond_node_targets)}
        
        # Strictly align bond labels to edge order (one label per edge; missing entries get mask=False and zero placeholder)
        if atom_bond_index.numel() > 0:
            E = atom_bond_index.shape[1]
            y_bond_aligned = torch.zeros((E, 2), dtype=torch.float)
            bond_supervised_mask = torch.zeros((E,), dtype=torch.bool)
            atom_pair_to_bond_node = {v: k for k, v in bond_node_to_atom_pair.items()}
            for i in range(E):
                u, v = atom_bond_index[0, i].item(), atom_bond_index[1, i].item()
                bond_node = atom_pair_to_bond_node.get((u, v))
                if bond_node is not None and bond_node in bond_node_idx_to_target:
                    y_bond_aligned[i] = bond_node_idx_to_target[bond_node]
                    bond_supervised_mask[i] = True
        else:
            y_bond_aligned = torch.empty((0, 2), dtype=torch.float)
            bond_supervised_mask = torch.empty((0,), dtype=torch.bool)
        y_bond_props = y_bond_aligned
        
        # Interaction property label
        y_interaction_props = torch.as_tensor(molecule_dict['interaction_targets']) if 'interaction_targets' in molecule_dict else torch.empty((0, 3), dtype=torch.float)
        
        # --- Link prediction candidates (pos/neg) ---
        raw_edges = molecule_dict.get('interaction_edges', None)
        positive_edges = torch.empty((2, 0), dtype=torch.long)
        if raw_edges is not None:
            if isinstance(raw_edges, torch.Tensor):
                if raw_edges.numel() == 0:
                    positive_edges = torch.empty((2, 0), dtype=torch.long)
                elif raw_edges.dim() == 1:
                    if raw_edges.numel() % 2 == 0:
                        positive_edges = raw_edges.view(2, -1).long().contiguous()
                elif raw_edges.dim() == 2:
                    if raw_edges.size(0) == 2:
                        positive_edges = raw_edges.long().contiguous()
                    elif raw_edges.size(1) == 2:
                        positive_edges = raw_edges.t().long().contiguous()
            elif isinstance(raw_edges, (list, tuple)):
                try:
                    edge_tensor = torch.as_tensor(raw_edges, dtype=torch.long)
                    if edge_tensor.numel() == 0:
                        positive_edges = torch.empty((2, 0), dtype=torch.long)
                    elif edge_tensor.dim() == 2 and edge_tensor.size(1) == 2:
                        positive_edges = edge_tensor.t().contiguous()
                except Exception:
                    positive_edges = torch.empty((2, 0), dtype=torch.long)
        
        num_nodes = int(torch.as_tensor(node_type).numel())
        num_positive = int(positive_edges.size(1))
        num_negative = int(self.neg_to_pos_ratio * max(1, num_positive))
        
        if num_positive == 0 or num_negative == 0:
            link_prediction_candidates = torch.empty((2, 0), dtype=torch.long)
            link_prediction_labels = torch.tensor([], dtype=torch.float)
        else:
            import random
            negative_edges = []
            positive_set = set(tuple(e) for e in positive_edges.t().tolist())
            while len(negative_edges) < num_negative:
                u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
                if u == v or (u, v) in positive_set or (v, u) in positive_set:
                    continue
                negative_edges.append([u, v])
            negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()
            link_prediction_candidates = torch.cat([positive_edges, negative_edges], dim=1)
            positive_labels = torch.ones(num_positive)
            negative_labels = torch.zeros(num_negative)
            link_prediction_labels = torch.cat([positive_labels, negative_labels], dim=0)
        
        # qm9_id
        qm9_raw = molecule_dict['qm9_id']
        qm9_id_value = int(qm9_raw) if not isinstance(qm9_raw, (list, tuple, torch.Tensor)) else int(qm9_raw[0])
        qm9_id_tensor = torch.as_tensor([qm9_id_value], dtype=torch.long)
        
        data = NBODatum(
            qm9_id=qm9_id_tensor,
            x=torch.as_tensor(molecule_dict['x']).float(),
            pos=torch.as_tensor(molecule_dict['positions']).float(),
            node_type=node_type.long(),
            atom_to_nbo_index=torch.as_tensor(molecule_dict['atom_to_bond_index']).long(),
            atom_bond_index=atom_bond_index,
            interaction_edge_index=positive_edges,
            y_atomic_props=torch.as_tensor(y_atomic_props).float(),
            y_bond_props=torch.as_tensor(y_bond_props).float(),
            y_interaction_props=torch.as_tensor(y_interaction_props).float(),
            y_all_targets=torch.as_tensor(molecule_dict['y_targets']).float(),
            link_prediction_candidates=link_prediction_candidates.long(),
            link_prediction_labels=link_prediction_labels.float(),
            bond_supervised_mask=bond_supervised_mask
        )
        return data

def collate_for_nbo_lp(batch):
    """Custom collate function to handle pyg Data objects in a standard DataLoader."""
    try:
        return pyg_data.Batch.from_data_list(batch)
    except Exception as e:
        print("[Collate Debug] Batch failure. Inspecting items:")
        for i, d in enumerate(batch):
            print(f"  -- Item {i}:")
            for key, val in d:
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: tensor shape={tuple(val.shape)} dtype={val.dtype}")
                else:
                    print(f"    {key}: type={type(val)} value={val}")
        raise

def get_nbo_dataloaders(dataset_root, batch_size, num_workers=0):
    raw_path = os.path.join(dataset_root, 'simg_mirrored_dataset.pt')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
    
    # Use a standard 80/10/10 split
    full_data_list = _torch_load_compat(raw_path)
    num_data = len(full_data_list)
    indices = list(range(num_data))
    random.seed(42)
    random.shuffle(indices)
    
    num_train = int(num_data * 0.8)
    num_val = int(num_data * 0.1)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]

    train_dataset = NBO_Dataset_LP(raw_path, split_indices=train_indices)
    val_dataset = NBO_Dataset_LP(raw_path, split_indices=val_indices)
    test_dataset = NBO_Dataset_LP(raw_path, split_indices=test_indices)
    
    # --- Normalization (slightly adapted for the new Dataset class) ---
    print("\nCalculating normalization stats from the training set...")
    all_train_atomic_targets = []
    all_train_bond_targets = []
    for i in tqdm(range(len(train_dataset)), desc="Collecting training targets"):
        data = train_dataset[i]
        all_train_atomic_targets.append(data.y_atomic_props)
        all_train_bond_targets.append(data.y_bond_props)

    cat_atom_targets = torch.cat(all_train_atomic_targets, dim=0)
    cat_bond_targets = torch.cat(all_train_bond_targets, dim=0)

    atom_mean = cat_atom_targets.mean(dim=0)
    atom_std = cat_atom_targets.std(dim=0)
    atom_std[atom_std < 1e-8] = 1.0

    bond_mean = torch.tensor([0.0, 0.0])
    if cat_bond_targets.numel() > 0:
        bond_mean = cat_bond_targets.mean(dim=0)
    bond_std = torch.tensor([1.0, 1.0])
    if cat_bond_targets.numel() > 0:
        bond_std = cat_bond_targets.std(dim=0)
    bond_std[bond_std < 1e-8] = 1.0

    print("Computed Normalization Stats (from training set):")
    print(f"  - Atomic Props Mean: {atom_mean.tolist()}")
    print(f"  - Atomic Props Std:  {atom_std.tolist()}")
    print(f"  - Bond Props Mean:   {bond_mean.tolist()}")
    print(f"  - Bond Props Std:    {bond_std.tolist()}")

    norm_stats = {'atom_mean': atom_mean, 'atom_std': atom_std, 'bond_mean': bond_mean, 'bond_std': bond_std}
    
    print(f"\nDataset split created:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_for_nbo_lp)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_for_nbo_lp)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_for_nbo_lp)

    return train_loader, val_loader, test_loader, norm_stats 