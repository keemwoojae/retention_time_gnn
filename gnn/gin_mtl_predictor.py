# -*- coding: utf-8 -*-
"""
GIN-based Multi-Task Learning Predictor

Hard Parameter Sharing architecture:
- Shared GIN Encoder across all tasks
- Task-specific prediction heads
"""

import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set

from .gin import GIN

__all__ = ['GINMTLPredictor']


class GINMTLPredictor(nn.Module):
    """
    GIN-based Multi-Task Learning model with Hard Parameter Sharing.

    Architecture:
        Input Graphs
            |
        Shared GIN Encoder
            |
        Graph-level Readout (pooling)
            |
        Graph Embeddings
            |
        +---+---+---+---+
        |   |   |   |   |
      Head0 Head1 ... HeadN  (Task-specific heads)
        |   |   |   |   |
        y0  y1  ... yN

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    n_tasks : int
        Number of tasks (prediction heads).
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.1.
    readout : str
        Readout method: 'sum', 'mean', 'max', 'attention', or 'set2set'. Default to 'mean'.
    predictor_hidden_feats : int
        Hidden dimension for prediction heads. Default to 256.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 n_tasks,
                 num_layers=5,
                 emb_dim=300,
                 dropout=0.1,
                 readout='mean',
                 predictor_hidden_feats=256):
        super(GINMTLPredictor, self).__init__()

        self.n_tasks = n_tasks
        self.emb_dim = emb_dim

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        # Shared GNN Encoder
        self.gnn = GIN(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            num_layers=num_layers,
            emb_dim=emb_dim,
            dropout=dropout
        )

        # Graph-level Readout
        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            self.readout = GlobalAttentionPooling(
                gate_nn=nn.Linear(emb_dim, 1)
            )
        elif readout == 'set2set':
            self.readout = Set2Set(emb_dim, n_iters=3, n_layers=1)
            emb_dim = emb_dim * 2  # Set2Set doubles the output dimension
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max', 'attention' or 'set2set', got {}".format(readout))

        # Task-specific Prediction Heads
        self.heads = nn.ModuleList()
        for _ in range(n_tasks):
            head = nn.Sequential(
                nn.Linear(emb_dim, predictor_hidden_feats),
                nn.ReLU(),
                nn.Linear(predictor_hidden_feats, 1)
            )
            self.heads.append(head)

        # For compatibility with existing single-task predict interface
        self.predict = self.heads[0]  # Default head

    def forward(self, g, node_feats, edge_feats, task_ids=None):
        """
        Multi-task prediction.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        task_ids : LongTensor of shape (B,), optional
            Task IDs for each graph in the batch.
            If None, returns predictions from all heads.

        Returns
        -------
        If task_ids is None:
            FloatTensor of shape (B, n_tasks)
                Predictions from all task heads
        If task_ids is provided:
            FloatTensor of shape (B, 1)
                Predictions from the corresponding task head for each sample
        """
        # Shared encoder
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)

        if task_ids is None:
            # Return predictions from all heads
            outputs = []
            for head in self.heads:
                outputs.append(head(graph_feats))
            return torch.cat(outputs, dim=1)  # (B, n_tasks)
        else:
            # Return predictions only from the corresponding head
            batch_size = graph_feats.size(0)
            outputs = torch.zeros(batch_size, 1, device=graph_feats.device)

            # Get unique task IDs in this batch
            unique_tasks = torch.unique(task_ids)

            for task_id in unique_tasks:
                mask = task_ids == task_id
                if mask.any():
                    task_feats = graph_feats[mask]
                    task_preds = self.heads[task_id.item()](task_feats)
                    outputs[mask] = task_preds

            return outputs

    def forward_all_heads(self, g, node_feats, edge_feats):
        """
        Get predictions from all task heads.

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Predictions from all task heads
        """
        return self.forward(g, node_feats, edge_feats, task_ids=None)

    def get_graph_embedding(self, g, node_feats, edge_feats):
        """
        Get graph-level embeddings (before prediction heads).

        Returns
        -------
        FloatTensor of shape (B, emb_dim)
            Graph embeddings
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return graph_feats

    def reset_heads(self):
        """Reset all prediction heads."""
        for head in self.heads:
            for layer in head:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def freeze_encoder(self):
        """Freeze the shared GNN encoder."""
        for param in self.gnn.parameters():
            param.requires_grad = False
        # Also freeze readout if it has parameters
        for param in self.readout.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the shared GNN encoder."""
        for param in self.gnn.parameters():
            param.requires_grad = True
        for param in self.readout.parameters():
            param.requires_grad = True

    def get_encoder_params(self):
        """Get encoder parameters (for separate learning rate)."""
        params = list(self.gnn.parameters())
        params.extend(list(self.readout.parameters()))
        return params

    def get_head_params(self):
        """Get all head parameters."""
        params = []
        for head in self.heads:
            params.extend(list(head.parameters()))
        return params

    def load_pretrained_encoder(self, pretrained_model_path):
        """
        Load pre-trained encoder weights from a single-task model.

        Parameters
        ----------
        pretrained_model_path : str
            Path to the pre-trained model checkpoint
        """
        pretrained_state = torch.load(pretrained_model_path, map_location='cpu')

        # Load GNN weights
        gnn_state = {k.replace('gnn.', ''): v for k, v in pretrained_state.items()
                     if k.startswith('gnn.')}
        self.gnn.load_state_dict(gnn_state)

        # Load readout weights if available
        readout_state = {k.replace('readout.', ''): v for k, v in pretrained_state.items()
                         if k.startswith('readout.')}
        if readout_state:
            self.readout.load_state_dict(readout_state, strict=False)

        print(f"[GINMTLPredictor] Loaded pre-trained encoder from {pretrained_model_path}")
