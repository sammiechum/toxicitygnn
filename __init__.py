"""
toxicitygnn package

Provides GNN model factory and state-loading utilities for SOM (Site of Metabolism)
prediction using Chebyshev spectral graph convolutions.
"""

import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric import __version__ as pygVersion


def createGnnSom(convName, width, depth, featureCount):
    """Build a sequential Chebyshev GNN for node-level SOM prediction.

    Args:
        convName: Convolution operator name ('cheb', 'cheb10k', or 'cheb15k').
        width: Hidden layer width.
        depth: Number of graph convolutional layers.
        featureCount: Number of input node features.

    Returns:
        A torch_geometric.nn.Sequential model outputting one scalar per node.
    """
    if convName == 'cheb':
        conv = lambda a, b: gnn.ChebConv(a, b, K=5)
    elif convName == 'cheb10k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=10)
    elif convName == 'cheb15k':
        conv = lambda a, b: gnn.ChebConv(a, b, K=15)
    else:
        raise Exception('Unknown convolution operator ' + convName)

    layerSizes = [featureCount] + [width] * depth + [1]
    modules = []
    for i in range(len(layerSizes) - 1):
        modules.append((conv(layerSizes[i], layerSizes[i + 1]), 'x, edge_index -> x'))
        if i != len(layerSizes) - 2:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.5))
    return gnn.Sequential('x, edge_index', modules)


def loadGnnSomState(model, state):
    """Load a saved state dict into a GNN model, handling pre-2.0.0 PyG format.

    PyTorch Geometric 2.0.0 changed the internal parameter naming convention for
    ChebConv. This function converts older state dicts to the current format before
    calling load_state_dict.

    Args:
        model: The GNN model to load weights into.
        state: A state dict, either in current or pre-2.0.0 PyG format.
    """
    if pygVersion.startswith('2.'):
        newState = {}
        for name, value in state.items():
            nns, index, weightOrBias = name.split('.')
            if nns != 'nns':
                raise Exception('Unexpected state parameter')
            index = int(index)
            if weightOrBias == 'weight':
                for k in range(value.shape[0]):
                    newState['module_%d.lins.%d.weight' % (index, k)] = value[k].t()
            elif weightOrBias == 'bias':
                newState['module_%d.bias' % index] = value
            else:
                raise Exception('Unexpected state parameter')
        state = newState
    model.load_state_dict(state)
