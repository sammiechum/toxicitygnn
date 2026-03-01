#!/usr/bin/env python3
"""
gnn-node.py

Baseline training script for node-level Site of Metabolism (SOM) prediction.
Trains a GNN to classify individual atoms within drug molecules
as metabolic sites or non-sites, using weighted binary cross-entropy to handle
class imbalance. Predictions are saved to a pickle file for further evaluation.

Usage:
    python gnn-node.py [--gpu=N] [--width=N] [--depth=N] [--conv=NAME]
                       [--epochs=N] [--split=N] [--cyp] [--noncyp]
                       [--nokcf] [--noadj] [--save] [--haneen]
"""

import os
import sys
from torch_geometric.nn import global_add_pool
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import numpy as np

gpuIndex = 0

width = 512
depth = 3
conv = 'cheb'
includeKcfTypes = True
includeAdjustments = True
saveModel = False
epochs = 400
split = 0
modelType = 'SomGnnNode'

considerCyp = False
considerNonCyp = False
for arg in sys.argv:
    if arg.startswith('--gpu'):
        gpuIndex = int(arg.split('=', 1)[1])
    elif arg.startswith('--width='):
        width = int(arg.split('=', 1)[1])
    elif arg.startswith('--depth='):
        depth = int(arg.split('=', 1)[1])
    elif arg.startswith('--conv='):
        conv = arg.split('=', 1)[1]
    elif arg.startswith('--epochs='):
        epochs = int(arg.split('=', 1)[1])
    elif arg.startswith('--split='):
        split = int(arg.split('=', 1)[1])
    elif arg.startswith('--cyp'):
        considerCyp = True
    elif arg.startswith('--noncyp'):
        considerNonCyp = True
    elif arg == '--nokcf':
        includeKcfTypes = False
    elif arg == '--noadj':
        includeAdjustments = False
    elif arg == '--save':
        saveModel = True
    elif arg == '--haneen':
        modelType += 'H'


import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric import __version__ as pygVersion


def createSomGnn(convName, width, depth, featureCount, forEmbedding):
    """Build a sequential GNN for node-level SOM prediction.

    Args:
        convName: Graph convolution operator name (e.g. 'cheb', 'gat', 'gin').
        width: Hidden layer width.
        depth: Number of graph convolutional layers.
        featureCount: Number of input node features.
        forEmbedding: If True, omit the final output layer (used for embedding extraction).

    Returns:
        A torch_geometric.nn.Sequential model outputting one scalar per node (or
        embeddings if forEmbedding=True), with global_add_pool applied at the end.
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
    modules.append((global_add_pool, 'x, batch -> x'))
    return gnn.Sequential('x, edge_index, batch', modules)


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

###

datasetName = 'dataset2_addprop_encode'

modelName = '%s-%s%dx%d-%depochs' % (modelType, conv, width, depth, epochs)
if considerCyp != considerNonCyp:
    if considerCyp:
        modelName += '-cyp'
    else:
        modelName += '-noncyp'
if not includeKcfTypes:
    datasetName += '-nokcf'
    modelName += '-nokcf'
if not includeAdjustments:
    datasetName += '-noadj'
    modelName += '-noadj'
modelName += '-%dsplit' % split

predFilename = 'data/pred-%s.pkl' % modelName
modelFilename = 'data/model-%s.pt' % modelName

if os.path.isfile(predFilename):
    print('File with predictions already exists')
    exit()

import datetime as dt
import pickle
import random

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader, Batch
from sklearn.metrics import roc_auc_score

random.seed(123)
torch.manual_seed(123)


def convertMolRecordsToData(mols):
    """Convert molecule records from the dataset pickle into PyG Data objects.

    Args:
        mols: List of molecule dicts from the dataset, each containing 'node' data.

    Returns:
        Tuple of (dataList, batchList) where dataList is a list of PyG Data objects
        and batchList maps each atom to its molecule index.
    """
    batchList = list()
    dataList = []
    for i, mol in enumerate(mols):
        if considerCyp != considerNonCyp:
            if considerCyp and not mol['isCyp']:
                continue
            elif considerNonCyp and not mol['isNonCyp']:
                continue

        molData = mol['node']
        x = (molData['x'])
        y = torch.unsqueeze(torch.FloatTensor(molData['y']), 0)

        atomCount = (molData['x'].size()[0])

        for j in range(int(atomCount)):
            batchList.append(i)

        edgeNum = len(molData['edges'])
        edgeIndex = torch.zeros((2, edgeNum * 2), dtype=torch.int64)
        for j, (a1, a2) in enumerate(molData['edges']):
            edgeIndex[0][j * 2] = a1
            edgeIndex[1][j * 2] = a2
            edgeIndex[0][j * 2 + 1] = a2
            edgeIndex[1][j * 2 + 1] = a1

        dataList.append(Data(x=x, y=y, edge_index=edgeIndex, datasetIndex=i))
    return dataList, batchList


with open('data/%s.pkl' % datasetName, 'rb') as f:
    records = pickle.load(f)

trainMolecules, batchListT = convertMolRecordsToData([records['mols'][i] for i in records['splits'][split]['train']])
validMolecules, batchListV = convertMolRecordsToData([records['mols'][i] for i in records['splits'][split]['valid']])

featureCount = trainMolecules[0].x.shape[1]
device = torch.device('cuda:%d' % gpuIndex if torch.cuda.is_available() else 'cpu')
model = createSomGnn(conv, width, depth, featureCount, forEmbedding=False).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

positives = 0
negatives = 0

for mol in trainMolecules:
    p = mol.y.sum()
    n = len(mol.y) - p
    positives += p
    negatives += n
print('Found %d positive and %d negative SOM examples (total %d)' % (positives, negatives, positives + negatives))
posWeight = torch.FloatTensor([negatives / positives]).to(device)

num_batches = 4
trainLoader = DataLoader(trainMolecules, batch_size=round(len(trainMolecules) / num_batches))
validLoader = DataLoader(validMolecules, batch_size=len(validMolecules))

batchListTSize = round(len(trainMolecules) / num_batches)
batchListTList = list()

pos = 0
for _ in range(num_batches - 1):
    subBatch = list()
    unique = 0
    while unique != batchListTSize + 1:
        if batchListT[pos] not in subBatch:
            unique += 1
        if unique != batchListTSize + 1:
            subBatch.append(batchListT[pos])
            pos += 1
    batchListTList.append(subBatch)

batchListTList.append(batchListT[pos:])

batchListTList = [torch.tensor(sublist) for sublist in batchListTList]
batchListV = torch.tensor(batchListV)

BOLD = '\u001b[1m'
RED = '\u001b[31m'
BLUE = '\u001b[34m'
RESET = '\u001b[0m'
MARKER_TRAIN = BOLD + BLUE + 'o' + RESET
MARKER_VALID = BOLD + RED + 'x' + RESET

print('Time     Epoch    Train AUC (1)  Valid AUC (2)  |50       |60       |70       |80       |90       |100%'.
      replace('(1)', '(' + MARKER_TRAIN + ')').replace('(2)', '(' + MARKER_VALID + ')'))

for epoch in range(epochs):
    model.train()
    for k, moleculesBatch in enumerate(trainLoader):
        batch = moleculesBatch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batchListTList[k])
        pred = pred[batchListTSize * k:]

        loss = nn.functional.binary_cross_entropy_with_logits(pred, batch.y, pos_weight=posWeight)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == (epochs - 1):
        model.eval()
        print(dt.datetime.now().time().strftime('%H:%M:%S ') + ('%5d    ' % epoch), end='', flush=True)
        bar = list('.         ' * 5 + '.')
        for marker, loader in [(MARKER_TRAIN, trainLoader), (MARKER_VALID, validLoader)]:
            predictions = []
            trueLabels = []
            with torch.no_grad():
                for l, moleculesBatch in enumerate(loader):
                    batch = moleculesBatch.to(device)
                    if loader == trainLoader:
                        pred = model(batch.x, batch.edge_index, batchListTList[l])
                        pred = pred[batchListTSize * l:]
                    else:
                        pred = model(batch.x, batch.edge_index, batchListV)

                    threshold = 0.5

                    predictions.extend(torch.sigmoid(pred).cpu().numpy())
                    trueLabels.extend(moleculesBatch.y.cpu().numpy())

                    adjpredictions = [1 if value >= threshold else 0 for value in predictions]
                    trueLabels = [int(value) for value in trueLabels]

            score = roc_auc_score(trueLabels, predictions)
            acc = balanced_accuracy_score(trueLabels, adjpredictions)

            print('%-15f' % acc, end='', flush=True)
            x = round((acc - 0.5) / 0.01)
            if 0 <= x < len(bar):
                bar[x] = marker
        print(''.join(bar))

if saveModel:
    torch.save(model.state_dict(), modelFilename)

model.eval()


def makePredictions(molecules, batchlist):
    """Run inference on a set of molecules and return per-atom predictions.

    Args:
        molecules: List of PyG Data objects.
        batchlist: Tensor mapping each atom to its molecule index.

    Returns:
        Dict mapping dataset molecule index to list of predicted SOM scores.
    """
    preds = {}
    with torch.no_grad():
        for molecule in molecules:
            pred = torch.sigmoid(model(molecule.x.to(device), molecule.edge_index.to(device), batchlist.to(device))).cpu()
            preds[molecule.datasetIndex] = torch.squeeze(pred).tolist()
    return preds


predictions = {
    'type': 'node',
    'split': split,
    'valid': makePredictions(validMolecules, batchListV),
}
with open(predFilename, 'wb') as f:
    pickle.dump(predictions, f)
