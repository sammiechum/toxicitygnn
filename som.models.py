"""
som.models.py

GNN model factory and state-loading utilities for node-level Site of Metabolism
(SOM) prediction. Provides a ChebConv-only model builder used by the SOM prediction
pipeline, and a compatibility shim for loading state dicts saved with pre-2.0.0
versions of PyTorch Geometric.
"""

import torch.nn as nn
import torch_geometric.nn as gnn


def createSomGnn(convName, width, depth, featureCount, forEmbedding):
    """Build a sequential Chebyshev GNN for node-level SOM prediction.

    Args:
        convName: Convolution operator name. Supported values:
                  'mf0'–'mf10'  MFConv with max_degree 0/1/2/5/10
                  'gcn'         GCNConv (improved=True)
                  'cheb1k'–'cheb15k'  ChebConv with K=1/2/3/4/5/10/15
                  'gat'         GATv2Conv
                  'arma'        ARMAConv (5 stacks, 1 layer)
                  'gin'/'gin2'/'gin3'/'ginte'/'gin2te'  GINConv variants
        width: Hidden layer width.
        depth: Number of graph convolutional layers.
        featureCount: Number of input node features.
        forEmbedding: If True, omit the final scalar output layer (used for
                      embedding extraction rather than prediction).

    Returns:
        A torch_geometric.nn.Sequential model that takes (x, edge_index) and
        outputs one scalar per node (or node embeddings if forEmbedding=True).
    """
    if convName == 'mf0':
        conv = lambda a, b: gnn.MFConv(a, b, max_degree=0)
    elif convName == 'mf1':
        conv = lambda a, b: gnn.MFConv(a, b, max_degree=1)
    elif convName == 'mf2':
        conv = lambda a, b: gnn.MFConv(a, b, max_degree=2)
    elif convName == 'mf5':
        conv = lambda a, b: gnn.MFConv(a, b, max_degree=5)
    elif convName == 'mf10':
        conv = lambda a, b: gnn.MFConv(a, b, max_degree=10)
    elif convName == 'gcn':
        conv = lambda a, b: gnn.GCNConv(a, b, improved=True)
    elif convName == 'cheb1k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=1)
    elif convName == 'cheb2k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=2)
    elif convName == 'cheb3k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=3)
    elif convName == 'cheb4k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=4)
    elif convName == 'cheb':
        conv = lambda a, b: gnn.ChebConv(a, b, K=5)
    elif convName == 'cheb10k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=10)
    elif convName == 'cheb15k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=15)
    elif convName == 'gat':
        conv = lambda a, b: gnn.GATv2Conv(a, b)
    elif convName == 'arma':
        conv = lambda a, b: gnn.ARMAConv(a, b, num_stacks=5, num_layers=1)
    elif convName == 'gin':
        conv = lambda a, b: gnn.GINConv(nn.Linear(a, b))
    elif convName == 'gin2':
        conv = lambda a, b: gnn.GINConv(nn.Sequential(nn.Linear(a, a), nn.Linear(a, b)))
    elif convName == 'ginte':
        conv = lambda a, b: gnn.GINConv(nn.Linear(a, b), train_eps=True)
    elif convName == 'gin2te':
        conv = lambda a, b: gnn.GINConv(nn.Sequential(nn.Linear(a, a), nn.Linear(a, b)), train_eps=True)
    elif convName == 'gin3':
        conv = lambda a, b: gnn.GINConv(nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b), nn.ReLU(), nn.BatchNorm1d(b)), train_eps=False)
    else:
        raise Exception('Unknown convolution operator ' + convName)

    layerSizes = [featureCount] + [width] * depth
    if not forEmbedding:
        layerSizes += [1]

    modules = []
    for i in range(len(layerSizes) - 1):
        modules.append((conv(layerSizes[i], layerSizes[i + 1]), 'x, edge_index -> x'))
        if forEmbedding or i != len(layerSizes) - 2:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.5))
    return gnn.Sequential('x, edge_index', modules)
