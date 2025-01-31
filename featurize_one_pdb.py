# featurize_one_pdb.py
import sys
import torch
from src.data.featurizer import RNAGraphFeaturizer

def main(pdb_path):
    # 1) Instantiate featurizer
    featurizer = RNAGraphFeaturizer(
        split='test',        # 'train' if you want coordinate noise
        top_k=10,            # number of neighbors for KNN
        num_rbf=16,
        num_posenc=16,
        max_num_conformers=3,
        noise_scale=0.0,     # no noise for 'test'
        device='cpu'         # or 'cuda' if you want GPU
    )

    # 2) Featurize the single PDB
    data, rna_dict = featurizer.featurize_from_pdb_file(pdb_path)

    # 3) Print or save the result
    print("Featurized Data:\n", data)
    print("Sequence tensor:", data.seq)
    print("Node scalar shape:", data.node_s.shape)
    print("Edge scalar shape:", data.edge_s.shape)
    print("Edge index shape:", data.edge_index.shape)

    # Optionally save the PyG Data object
    # torch.save(data, "my_rna_data.pt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python featurize_one_pdb.py <path_to_rna.pdb>")
        sys.exit(1)

    pdb_file = sys.argv[1]
    main(pdb_file)
