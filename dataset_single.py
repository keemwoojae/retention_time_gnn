"""
Single-Head Dataset with Categorical Features

Supports:
- System ID as categorical feature (one-hot or embedding)
- Chromatography type (HILIC/RP) as categorical feature
- Task-wise or Global normalization
"""

import os
import numpy as np
import torch
from dgl import graph
from sklearn.model_selection import KFold


# Chromatography type mapping based on dataset names
# HILIC: Hydrophilic Interaction Liquid Chromatography
# RP: Reverse-Phase chromatography
CHROMATOGRAPHY_TYPE = {
    'Cao_HILIC': 'HILIC',
    'Eawag_XBridgeC18': 'RP',
    'FEM_lipids': 'RP',
    'FEM_long': 'RP',
    'FEM_orbitrap_plasma': 'RP',
    'FEM_orbitrap_urine': 'RP',
    'FEM_short': 'RP',
    'IPB_Halle': 'RP',
    'LIFE_new': 'RP',
    'LIFE_old': 'RP',
    'MassBank1': 'RP',
    'MassBank2': 'RP',
    'MetaboBase': 'RP',
    'MPI_Symmetry': 'RP',
    'MTBLS87': 'HILIC',
    'PFR-TK72': 'RP',
    'RIKEN': 'RP',
    'RIKEN_PlaSMA': 'RP',
    'UFZ_Phenomenex': 'RP',
    'UniToyama_Atlantis': 'RP',
}

# Convert to numeric: HILIC=0, RP=1
CHROM_TYPE_TO_ID = {'HILIC': 0, 'RP': 1}


class SingleHeadDataset:
    """
    Dataset for single-head model with categorical features.

    Combines multiple chromatographic systems into one dataset,
    using system_id and chromatography_type as categorical features.
    """

    def __init__(self, dataset_names, cv_id=0, split='trn', seed=134,
                 normalization='task_wise', include_system_id=True,
                 include_chrom_type=False):
        """
        Args:
            dataset_names: List of dataset names to combine
            cv_id: Cross-validation fold id (0-9)
            split: 'trn' or 'tst'
            seed: Random seed for reproducibility
            normalization: 'task_wise' or 'global'
            include_system_id: Whether to include system_id as feature
            include_chrom_type: Whether to include chromatography type as feature
        """
        self.n_splits = 10

        assert cv_id in list(range(self.n_splits))
        assert split in ['trn', 'tst']
        assert normalization in ['task_wise', 'global']

        self.dataset_names = dataset_names
        self.n_systems = len(dataset_names)
        self.cv_id = cv_id
        self.split = split
        self.seed = seed
        self.normalization = normalization
        self.include_system_id = include_system_id
        self.include_chrom_type = include_chrom_type

        # Statistics for normalization
        self.task_means = {}
        self.task_stds = {}
        self.global_mean = None
        self.global_std = None

        # Feature dimensions for categorical features
        self.system_id_dim = len(dataset_names) if include_system_id else 0
        self.chrom_type_dim = 2 if include_chrom_type else 0  # HILIC, RP
        self.cat_feat_dim = self.system_id_dim + self.chrom_type_dim

        self.load()

    def load(self):
        """Load and combine multiple datasets."""
        all_labels = []
        all_system_ids = []
        all_chrom_types = []
        all_n_node = []
        all_n_edge = []
        all_node_attr = []
        all_edge_attr = []
        all_src = []
        all_dst = []

        for system_id, name in enumerate(self.dataset_names):
            data_path = f'./data/dataset_graph_{name}.npz'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found: {data_path}")

            [mol_dict] = np.load(data_path, allow_pickle=True)['data']

            # Cross-validation split
            kf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
            cv_splits = [split for split in kf.split(range(len(mol_dict['label'])))]
            cv_splits = cv_splits[self.cv_id]

            if self.split == 'trn':
                mol_indices = np.array([i in cv_splits[0] for i in range(len(mol_dict['label']))], dtype=bool)
            else:  # 'tst'
                mol_indices = np.array([i in cv_splits[1] for i in range(len(mol_dict['label']))], dtype=bool)

            # Extract data for this split
            labels = mol_dict['label'][mol_indices].reshape(-1, 1)
            n_node = mol_dict['n_node'][mol_indices]
            n_edge = mol_dict['n_edge'][mol_indices]

            node_indices = np.repeat(mol_indices, mol_dict['n_node'])
            edge_indices = np.repeat(mol_indices, mol_dict['n_edge'])

            node_attr = mol_dict['node_attr'][node_indices]
            edge_attr = mol_dict['edge_attr'][edge_indices]
            src = mol_dict['src'][edge_indices]
            dst = mol_dict['dst'][edge_indices]

            # Store task-wise statistics (always compute for potential use)
            self.task_means[system_id] = np.mean(labels)
            self.task_stds[system_id] = np.std(labels)

            # Create system_id array
            system_ids = np.full(len(labels), system_id, dtype=np.int64)

            # Create chromatography type array
            chrom_type = CHROMATOGRAPHY_TYPE.get(name, 'RP')
            chrom_type_id = CHROM_TYPE_TO_ID[chrom_type]
            chrom_types = np.full(len(labels), chrom_type_id, dtype=np.int64)

            all_labels.append(labels)
            all_system_ids.append(system_ids)
            all_chrom_types.append(chrom_types)
            all_n_node.append(n_node)
            all_n_edge.append(n_edge)
            all_node_attr.append(node_attr)
            all_edge_attr.append(edge_attr)
            all_src.append(src)
            all_dst.append(dst)

        # Concatenate all data
        self.label = np.vstack(all_labels)
        self.system_id = np.concatenate(all_system_ids)
        self.chrom_type = np.concatenate(all_chrom_types)
        self.n_node = np.concatenate(all_n_node)
        self.n_edge = np.concatenate(all_n_edge)
        self.node_attr = np.vstack(all_node_attr)
        self.edge_attr = np.vstack(all_edge_attr)
        self.src = np.concatenate(all_src)
        self.dst = np.concatenate(all_dst)

        # Compute global statistics
        self.global_mean = np.mean(self.label)
        self.global_std = np.std(self.label)

        # Compute cumulative sums for indexing
        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])

        assert len(self.n_node) == len(self.label) == len(self.system_id)

        self._print_info()

    def _print_info(self):
        """Print dataset information."""
        print(f"\n[SingleHeadDataset] Loaded {len(self.dataset_names)} systems, "
              f"total samples: {len(self.label)}, split: {self.split}")
        print(f"  Normalization: {self.normalization}")
        print(f"  Include system_id: {self.include_system_id} (dim={self.system_id_dim})")
        print(f"  Include chrom_type: {self.include_chrom_type} (dim={self.chrom_type_dim})")

        if self.normalization == 'global':
            print(f"  Global stats: mean={self.global_mean:.2f}, std={self.global_std:.2f}")

        for system_id, name in enumerate(self.dataset_names):
            n_samples = np.sum(self.system_id == system_id)
            chrom_type = CHROMATOGRAPHY_TYPE.get(name, 'RP')
            print(f"  System {system_id} ({name}, {chrom_type}): {n_samples} samples, "
                  f"mean={self.task_means[system_id]:.2f}, std={self.task_stds[system_id]:.2f}")

    def __getitem__(self, idx):
        """Get a single sample with categorical features."""
        g = graph(
            (self.src[self.e_csum[idx]:self.e_csum[idx+1]],
             self.dst[self.e_csum[idx]:self.e_csum[idx+1]]),
            num_nodes=self.n_node[idx]
        )
        g.ndata['node_attr'] = torch.from_numpy(
            self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]
        ).float()
        g.edata['edge_attr'] = torch.from_numpy(
            self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]
        ).float()

        label = self.label[idx]
        system_id = self.system_id[idx]
        chrom_type = self.chrom_type[idx]

        # Build categorical features
        cat_feats = self._build_categorical_features(system_id, chrom_type)

        return g, label, system_id, chrom_type, cat_feats

    def _build_categorical_features(self, system_id, chrom_type):
        """
        Build categorical feature vector.

        Returns:
            np.array of shape (cat_feat_dim,)
        """
        features = []

        if self.include_system_id:
            # One-hot encoding for system_id
            system_onehot = np.zeros(self.n_systems, dtype=np.float32)
            system_onehot[system_id] = 1.0
            features.append(system_onehot)

        if self.include_chrom_type:
            # One-hot encoding for chromatography type
            chrom_onehot = np.zeros(2, dtype=np.float32)
            chrom_onehot[chrom_type] = 1.0
            features.append(chrom_onehot)

        if features:
            return np.concatenate(features)
        else:
            return np.array([], dtype=np.float32)

    def __len__(self):
        return len(self.label)

    def get_system_indices(self, system_id):
        """Get indices of samples belonging to a specific system."""
        return np.where(self.system_id == system_id)[0]

    def get_node_feat_dim(self):
        """Get node feature dimension."""
        return self.node_attr.shape[1]

    def get_edge_feat_dim(self):
        """Get edge feature dimension."""
        return self.edge_attr.shape[1]

    def get_cat_feat_dim(self):
        """Get categorical feature dimension."""
        return self.cat_feat_dim

    def get_normalization_stats(self):
        """
        Get normalization statistics based on normalization mode.

        Returns:
            dict with 'mode', 'global_mean', 'global_std', 'task_means', 'task_stds'
        """
        return {
            'mode': self.normalization,
            'global_mean': self.global_mean,
            'global_std': self.global_std,
            'task_means': self.task_means,
            'task_stds': self.task_stds
        }


def collate_single_graphs(batch):
    """
    Collate function for single-head dataset.
    Returns batched graphs with categorical features.
    """
    import dgl

    g_list, label_list, system_id_list, chrom_type_list, cat_feat_list = map(list, zip(*batch))

    g_batch = dgl.batch(g_list)
    labels = torch.FloatTensor(np.vstack(label_list))
    system_ids = torch.LongTensor(system_id_list)
    chrom_types = torch.LongTensor(chrom_type_list)
    cat_feats = torch.FloatTensor(np.vstack(cat_feat_list))

    return g_batch, g_batch.ndata['node_attr'], g_batch.edata['edge_attr'], labels, system_ids, chrom_types, cat_feats
