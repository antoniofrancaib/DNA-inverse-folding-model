#!/usr/bin/env python
"""
featurize_pdb.py

Usage:
    python featurize_pdb.py --pdb_filepath <your_input.pdb>

Outputs a single torch_geometric.data.Data object describing
the RNA backbone graph and features.
"""
import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_cluster

# For parsing PDB
import biotite
import biotite.structure as bs
import biotite.structure.io as bsio

from Bio.Seq import Seq

############################################################
# Constants & Helpers
############################################################

RNA_ATOMS = [
    'P', "C4'", "N1", "N9", # includes pyrimidine (N1) or purine (N9)
    # But in practice we keep the 3-bead "backbone": P, C4', N1/N9
]

RNA_NUCLEOTIDES = ["A", "G", "C", "U"]
LETTER_TO_NUM = dict(zip(RNA_NUCLEOTIDES, range(len(RNA_NUCLEOTIDES))))
FILL_VALUE = 1e-5

def safe_arange(n):
    return np.arange(n, dtype=np.int64)

def parse_pdb_to_array(pdb_filepath):
    """
    Read a PDB file and return a biotite AtomArray.
    """
    array = bsio.load_structure(pdb_filepath)
    # By default, loads as an AtomArray or AtomArrayStack
    # If it's a stack (multiple models), just take the first
    if isinstance(array, bs.AtomArrayStack):
        array = array[0]
    return array

def extract_rna_residues(atom_array):
    """
    Biotite-based filter to keep only RNA nucleotides.
    """
    # Biotite codes for nucleotides, e.g. "A", "C", "G", "U", etc.
    # We can keep all residues that have a 'nucleotide' classification.
    # Alternatively, we can guess from the residue_name...
    # For now, let's do a direct approach for single chain, no insertion codes, etc.
    # We'll keep "A", "C", "G", "U" and ignore others:
    mask = np.isin(atom_array.res_name.astype(str), RNA_NUCLEOTIDES)
    rna_array = atom_array[mask]
    return rna_array

def remove_insertions(atom_array):
    """
    For simplicity, remove insertion codes by dropping duplicates
    of the same residue number. 
    """
    # We do a rough approach:  keep the first occurrence of each (chain, res_id).
    # This is a simplistic approach but sufficient for demonstration.
    unique_res = set()
    keep = []
    for i, (chain, resnum) in enumerate(zip(atom_array.chain_id, atom_array.res_id)):
        if (chain, resnum) not in unique_res:
            unique_res.add((chain, resnum))
            keep.append(True)
        else:
            keep.append(False)
    keep = np.array(keep)
    return atom_array[keep]

def get_sequence_and_coords(atom_array):
    """
    Extract a string sequence and a (N_res, N_atoms, 3) coordinate tensor
    from an RNA structure (with only relevant backbone atoms).
    We'll center them at origin.
    """
    # Group by (chain, residue_id) in the array:
    unique_res = []
    seq = []
    coords_list = []
    
    res_labels = (atom_array.chain_id.astype(str), atom_array.res_id)
    # We'll detect changes whenever chain+res_id changes:
    from itertools import groupby
    
    # Sort by (chain, residue_number) to group them in order
    # Actually, Biotite might already be sorted, but let's ensure:
    # We assume the array is sorted in ascending residue order.
    
    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id
    # We'll store sub-atom arrays for each residue
    all_res_indices = []
    
    i = 0
    while i < len(atom_array):
        c = chain_ids[i]
        r = res_ids[i]
        # gather all atoms with that c,r
        same_mask = (chain_ids == c) & (res_ids == r)
        subset = atom_array[same_mask]
        # residue name (like "A","G","C","U" if standard)
        rname = subset.res_name[0]
        # store
        seq.append(rname)
        all_res_indices.append(subset)
        i += np.sum(same_mask)
    
    # Build a (N_res, len(RNA_ATOMS), 3) array:
    # We'll do a quick approach: for each residue, pick out P, C4', N1 or N9, etc.
    # fill missing with FILL_VALUE
    N = len(all_res_indices)
    n_backbone = len(RNA_ATOMS)  # for 3-bead
    coords = np.full((N, n_backbone, 3), FILL_VALUE, dtype=np.float32)
    
    for i_res, subset in enumerate(all_res_indices):
        # find the relevant atoms
        for a_i, atom_name in enumerate(RNA_ATOMS):
            # find that atom in subset
            mask_atom = (subset.atom_name == atom_name)
            if np.sum(mask_atom) == 1:
                # we have that atom
                x, y, z = subset.coord[mask_atom][0]
                coords[i_res, a_i] = [x, y, z]
            # else it remains fill_value
    
    # center coords
    mean_xyz = np.mean(coords[coords != FILL_VALUE].reshape(-1, 3), axis=0)
    coords[coords != FILL_VALUE] -= mean_xyz
    
    # build final sequence string
    sequence = "".join(seq)
    return sequence, coords

def letter_to_int_seq(seq_str):
    """Convert an RNA sequence (A/G/C/U) into integer array."""
    arr = []
    for c in seq_str:
        if c in LETTER_TO_NUM:
            arr.append(LETTER_TO_NUM[c])
        else:
            # unknown or weird letter
            arr.append(len(LETTER_TO_NUM))  # e.g. 4
    return np.array(arr, dtype=np.int64)

############################################################
# Graph Featurization
############################################################

def build_knn_graph(node_coords, top_k=16):
    """
    Return edge_index for a KNN graph on the node coordinates of shape (N,3).
    We'll use torch_cluster.knn_graph.
    """
    # node_coords: shape (N, 3) as torch
    # produce a (2, E) index
    # note: knn_graph returns edges with each node connected to top_k neighbors.
    # By default, it doesn't do a symmetrical. We can do symmetrical with to_undirected.
    import torch_geometric
    N = node_coords.shape[0]
    coords_t = torch.tensor(node_coords, dtype=torch.float)
    edge_idx = torch_cluster.knn_graph(coords_t, k=top_k)  # shape [2, E]
    # to undirected
    edge_idx = torch_geometric.utils.to_undirected(edge_idx, num_nodes=N)
    return edge_idx

def rbf_expansion(d, D_min=0.0, D_max=30.0, D_count=32):
    """
    Radial basis expansion of distance d (Tensor shape [E]).
    """
    import torch
    rbf_centers = torch.linspace(D_min, D_max, D_count, device=d.device)
    rbf_std = (rbf_centers[1] - rbf_centers[0])
    out = torch.exp(-((d.unsqueeze(-1) - rbf_centers)**2) / (2 * (rbf_std**2)))
    return out  # shape [E, D_count]

def featurize_single_conf(seq_str, coords):
    """
    Featurize a single conformation's coords into a minimal example:

    - node_s: [N, 1, n_node_scalar_features]
    - node_v: [N, 1, n_node_vector_features, 3]
    - edge_s, edge_v, etc.

    We'll do a minimal design: node scalar=0, node vector=0, etc.
    And we'll put more interesting features on edges (like distance).
    """
    import torch
    
    N = coords.shape[0]
    # A single "conformation" => n_conf=1
    n_conf = 1
    
    # (1) seq
    seq_t = torch.as_tensor(letter_to_int_seq(seq_str), dtype=torch.long)  # shape [N]
    
    # (2) node_s
    # Minimal: let's do a zero "scalar" of shape (N, 1, 4) for ex
    node_s = torch.zeros((N, n_conf, 4), dtype=torch.float)
    
    # (3) node_v
    # Minimal: let's do a zero "vector" shape (N, n_conf, 2, 3)
    node_v = torch.zeros((N, n_conf, 2, 3), dtype=torch.float)
    
    # Build edges
    edge_index = build_knn_graph(coords, top_k=8)  # shape [2,E]
    E = edge_index.shape[1]
    
    # (4) edge_s
    # compute distances => shape [E]
    c_t = torch.tensor(coords, dtype=torch.float)
    src, dst = edge_index
    dist = (c_t[src] - c_t[dst]).norm(dim=-1)  # shape [E]
    # do an RBF expansion => shape [E, 32]
    dist_rbf = rbf_expansion(dist, D_min=0.0, D_max=10.0, D_count=16)
    # We'll store shape [E, n_conf, 16]
    edge_s = dist_rbf.unsqueeze(1)  # [E, 1, 16]
    
    # (5) edge_v
    # We'll store the normalized vector => shape [E, 3]
    disp = c_t[src] - c_t[dst]  # shape [E, 3]
    disp_norm = torch.norm(disp, dim=-1, keepdim=True) + 1e-6
    disp_unit = disp / disp_norm
    # place into shape [E, n_conf, 1, 3]
    edge_v = disp_unit.view(E, 1, 1, 3)
    
    # (6) mask_confs: shape [N, n_conf], all True
    mask_confs = torch.ones((N, n_conf), dtype=torch.bool)
    
    # (7) mask_coords: shape [N], all True
    mask_coords = torch.ones((N,), dtype=torch.bool)
    
    data = Data(
        seq=seq_t,
        node_s=node_s,
        node_v=node_v,
        edge_s=edge_s,
        edge_v=edge_v,
        edge_index=edge_index,
        mask_confs=mask_confs,
        mask_coords=mask_coords
    )
    
    return data

############################################################
# Main script
############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_filepath", type=str, required=True,
                        help="Path to an RNA PDB file")
    args = parser.parse_args()
    
    # 1) Parse the PDB
    arr = parse_pdb_to_array(args.pdb_filepath)
    
    # 2) Keep only RNA
    arr = extract_rna_residues(arr)
    
    # 3) Remove insertion duplicates
    arr = remove_insertions(arr)
    
    # 4) Build a single sequence + coords
    seq_str, coords = get_sequence_and_coords(arr)
    print(f"Extracted sequence: {seq_str} (length {len(seq_str)})")
    
    # 5) Featurize into a PyG Data
    data_point = featurize_single_conf(seq_str, coords)
    
    # 6) Print or return
    print("Featurized Data object:\n", data_point)
    print("\nShapes:")
    print("  seq:", data_point.seq.shape)
    print("  node_s:", data_point.node_s.shape)
    print("  node_v:", data_point.node_v.shape)
    print("  edge_index:", data_point.edge_index.shape, f"(E={data_point.edge_index.shape[1]})")
    print("  edge_s:", data_point.edge_s.shape)
    print("  edge_v:", data_point.edge_v.shape)
    print("  mask_confs:", data_point.mask_confs.shape)
    print("  mask_coords:", data_point.mask_coords.shape)

if __name__ == "__main__":
    main()
