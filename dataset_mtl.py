import os
import numpy as np
import torch
from dgl import graph
from sklearn.model_selection import KFold


class MTLDataset:
    """
    Multi-Task Learning Dataset that combines multiple datasets.
    Each sample includes a task_id indicating which dataset/instrument it belongs to.
    """

    def __init__(self, dataset_names, cv_id=0, split='trn', seed=134):
        """
        Args:
            dataset_names: List of dataset names to combine
            cv_id: Cross-validation fold id (0-9)
            split: 'trn' or 'tst'
            seed: Random seed for reproducibility
        """
        self.n_splits = 10

        assert cv_id in list(range(self.n_splits))
        assert split in ['trn', 'tst']

        self.dataset_names = dataset_names
        self.n_tasks = len(dataset_names)
        self.cv_id = cv_id
        self.split = split
        self.seed = seed

        # Task-wise statistics for normalization
        self.task_means = {}
        self.task_stds = {}

        self.load()

    def load(self):
        """Load and combine multiple datasets."""
        all_labels = []
        all_task_ids = []
        all_n_node = []
        all_n_edge = []
        all_node_attr = []
        all_edge_attr = []
        all_src = []
        all_dst = []

        for task_id, name in enumerate(self.dataset_names):
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

            # Store task-wise statistics
            self.task_means[task_id] = np.mean(labels)
            self.task_stds[task_id] = np.std(labels)

            # Create task_id array
            task_ids = np.full(len(labels), task_id, dtype=np.int64)

            all_labels.append(labels)
            all_task_ids.append(task_ids)
            all_n_node.append(n_node)
            all_n_edge.append(n_edge)
            all_node_attr.append(node_attr)
            all_edge_attr.append(edge_attr)
            all_src.append(src)
            all_dst.append(dst)

        # Concatenate all data
        self.label = np.vstack(all_labels)
        self.task_id = np.concatenate(all_task_ids)
        self.n_node = np.concatenate(all_n_node)
        self.n_edge = np.concatenate(all_n_edge)
        self.node_attr = np.vstack(all_node_attr)
        self.edge_attr = np.vstack(all_edge_attr)
        self.src = np.concatenate(all_src)
        self.dst = np.concatenate(all_dst)

        # Compute cumulative sums for indexing
        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])

        assert len(self.n_node) == len(self.label) == len(self.task_id)

        print(f"[MTLDataset] Loaded {len(self.dataset_names)} datasets, "
              f"total samples: {len(self.label)}, split: {self.split}")
        for task_id, name in enumerate(self.dataset_names):
            n_samples = np.sum(self.task_id == task_id)
            print(f"  Task {task_id} ({name}): {n_samples} samples, "
                  f"mean={self.task_means[task_id]:.2f}, std={self.task_stds[task_id]:.2f}")

    def __getitem__(self, idx):
        """Get a single sample."""
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
        task_id = self.task_id[idx]

        return g, label, task_id

    def __len__(self):
        return len(self.label)

    def get_task_indices(self, task_id):
        """Get indices of samples belonging to a specific task."""
        return np.where(self.task_id == task_id)[0]

    def get_node_feat_dim(self):
        """Get node feature dimension."""
        return self.node_attr.shape[1]

    def get_edge_feat_dim(self):
        """Get edge feature dimension."""
        return self.edge_attr.shape[1]


class TaskBalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures balanced sampling across tasks.
    Can be used for stratified batching.
    """

    def __init__(self, dataset, batch_size, shuffle=True, use_balanced=True):
        """
        Args:
            dataset: MTLDataset instance or Subset of MTLDataset
            batch_size: Total batch size
            shuffle: Whether to shuffle samples within each task
            use_balanced: If True, balance samples across tasks (may oversample small tasks).
                          If False, use simple sequential/shuffled sampling without balancing.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_balanced = use_balanced

        # Handle Subset objects (e.g., from random_split)
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            self.n_tasks = base_dataset.n_tasks

            # Create mapping: original index -> subset index
            orig_to_subset = {orig_idx: subset_idx for subset_idx, orig_idx in enumerate(subset_indices)}

            # Get task indices mapped to subset indices
            self.task_indices = {}
            for task_id in range(self.n_tasks):
                orig_task_indices = base_dataset.get_task_indices(task_id)
                # Filter to subset and map to subset indices
                subset_task_indices = [orig_to_subset[i] for i in orig_task_indices if i in orig_to_subset]
                self.task_indices[task_id] = np.array(subset_task_indices)
        else:
            # Direct MTLDataset
            self.n_tasks = dataset.n_tasks
            self.task_indices = {}
            for task_id in range(dataset.n_tasks):
                self.task_indices[task_id] = dataset.get_task_indices(task_id)

        # Calculate samples per task per batch
        self.samples_per_task = batch_size // self.n_tasks

    def __iter__(self):
        if not self.use_balanced:
            return self._iter_simple()
        return self._iter_balanced()

    def _iter_simple(self):
        """Simple sampling: just shuffle all indices without balancing."""
        all_indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(all_indices)
        return iter(all_indices.tolist())

    def _iter_balanced(self):
        """Balanced sampling: ensure equal representation from each task."""
        # Shuffle within each task if needed
        if self.shuffle:
            task_indices = {
                k: np.random.permutation(v)
                for k, v in self.task_indices.items()
            }
        else:
            task_indices = {k: v.copy() for k, v in self.task_indices.items()}

        # Create balanced batches
        indices = []
        task_ptrs = {k: 0 for k in range(self.n_tasks)}

        while True:
            batch = []
            for task_id in range(self.n_tasks):
                ptr = task_ptrs[task_id]
                task_idx = task_indices[task_id]

                # Get samples from this task
                end_ptr = min(ptr + self.samples_per_task, len(task_idx))
                batch.extend(task_idx[ptr:end_ptr].tolist())
                task_ptrs[task_id] = end_ptr

                # Wrap around if needed
                if task_ptrs[task_id] >= len(task_idx):
                    task_ptrs[task_id] = 0
                    if self.shuffle:
                        task_indices[task_id] = np.random.permutation(
                            self.task_indices[task_id]
                        )

            if len(batch) == 0:
                break

            indices.extend(batch)

            # Check if we've seen all samples at least once
            total_seen = sum(task_ptrs.values())
            total_samples = sum(len(v) for v in self.task_indices.values())
            if total_seen >= total_samples:
                break

        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def collate_mtl_graphs(batch):
    """
    Collate function for MTL dataset.
    Returns batched graphs with task_ids.
    """
    import dgl

    g_list, label_list, task_id_list = map(list, zip(*batch))

    g_batch = dgl.batch(g_list)
    labels = torch.FloatTensor(np.vstack(label_list))
    task_ids = torch.LongTensor(task_id_list)

    return g_batch, g_batch.ndata['node_attr'], g_batch.edata['edge_attr'], labels, task_ids
