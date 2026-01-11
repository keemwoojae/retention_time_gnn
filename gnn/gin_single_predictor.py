# -*- coding: utf-8 -*-
"""
GIN-based Single-Head Predictor with Categorical Features

Single head architecture with system/chromatography type as categorical inputs:
- Shared GIN Encoder
- Categorical features (system_id, chrom_type) concatenated to graph embedding
- Single prediction head
"""

import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set

from .gin import GIN

__all__ = ['GINSinglePredictor']


class GINSinglePredictor(nn.Module):
    """
    GIN-based Single-Head model with Categorical Features.

    Architecture:
        Input Graphs
            |
        GIN Encoder
            |
        Graph-level Readout (pooling)
            |
        Graph Embeddings  +  Categorical Features (system_id, chrom_type)
            |                      |
            +----------+-----------+
                       |
                   Concat
                       |
              Single Prediction Head
                       |
                       y

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    cat_feat_dim : int
        Dimension of categorical features (system_id one-hot + chrom_type one-hot).
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.1.
    readout : str
        Readout method: 'sum', 'mean', 'max', 'attention', or 'set2set'. Default to 'mean'.
    predictor_hidden_feats : int
        Hidden dimension for prediction head. Default to 256.
    cat_embedding_dim : int
        If > 0, project categorical features to this dimension before concat.
        If 0, use one-hot directly. Default to 0.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 cat_feat_dim,
                 num_layers=5,
                 emb_dim=300,
                 dropout=0.1,
                 readout='mean',
                 predictor_hidden_feats=256,
                 cat_embedding_dim=0):
        super(GINSinglePredictor, self).__init__()

        self.emb_dim = emb_dim
        self.cat_feat_dim = cat_feat_dim
        self.cat_embedding_dim = cat_embedding_dim

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        # GNN Encoder
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

        # Categorical feature projection (optional)
        if cat_embedding_dim > 0 and cat_feat_dim > 0:
            self.cat_proj = nn.Sequential(
                nn.Linear(cat_feat_dim, cat_embedding_dim),
                nn.ReLU(),
            )
            concat_dim = emb_dim + cat_embedding_dim
        else:
            self.cat_proj = None
            concat_dim = emb_dim + cat_feat_dim

        # Single Prediction Head
        self.predict = nn.Sequential(
            nn.Linear(concat_dim, predictor_hidden_feats),
            nn.ReLU(),
            nn.Linear(predictor_hidden_feats, 1)
        )

    def forward(self, g, node_feats, edge_feats, cat_feats=None):
        """
        Single-head prediction with categorical features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        cat_feats : float32 tensor of shape (B, cat_feat_dim), optional
            Categorical features (system_id one-hot, chrom_type one-hot).

        Returns
        -------
        FloatTensor of shape (B, 1)
            Predictions
        """
        # Encode graph
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)

        # Concatenate categorical features
        if cat_feats is not None and self.cat_feat_dim > 0:
            if self.cat_proj is not None:
                cat_feats = self.cat_proj(cat_feats)
            graph_feats = torch.cat([graph_feats, cat_feats], dim=1)

        # Predict
        return self.predict(graph_feats)

    def get_graph_embedding(self, g, node_feats, edge_feats):
        """
        Get graph-level embeddings (before categorical concat and prediction).

        Returns
        -------
        FloatTensor of shape (B, emb_dim)
            Graph embeddings
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return graph_feats

    def get_combined_embedding(self, g, node_feats, edge_feats, cat_feats=None):
        """
        Get combined embeddings (graph embedding + categorical features).

        Returns
        -------
        FloatTensor of shape (B, emb_dim + cat_dim)
            Combined embeddings
        """
        graph_feats = self.get_graph_embedding(g, node_feats, edge_feats)

        if cat_feats is not None and self.cat_feat_dim > 0:
            if self.cat_proj is not None:
                cat_feats = self.cat_proj(cat_feats)
            graph_feats = torch.cat([graph_feats, cat_feats], dim=1)

        return graph_feats

    def reset_head(self):
        """Reset the prediction head."""
        for layer in self.predict:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if self.cat_proj is not None:
            for layer in self.cat_proj:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def freeze_encoder(self):
        """Freeze the GNN encoder."""
        for param in self.gnn.parameters():
            param.requires_grad = False
        for param in self.readout.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the GNN encoder."""
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
        """Get head parameters (includes cat_proj if exists)."""
        params = list(self.predict.parameters())
        if self.cat_proj is not None:
            params.extend(list(self.cat_proj.parameters()))
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

        print(f"[GINSinglePredictor] Loaded pre-trained encoder from {pretrained_model_path}")
