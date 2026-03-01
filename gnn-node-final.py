#!/usr/bin/env python3
"""
gnn-node-final.py

Extended training script for node-level Site of Metabolism (SOM) prediction.
Extends gnn-node.py with support for configurable graph pooling strategies,
edge attributes, and optional QikProp ADMET molecular descriptors appended to
node features. Predictions are saved to a pickle file for downstream evaluation.

Usage:
    python gnn-node-final.py --data=DATASET [--gpu=N] [--width=N] [--depth=N]
                             [--conv=NAME] [--pool=NAME] [--epochs=N] [--split=N]
                             [--cyp] [--noncyp] [--nokcf] [--noadj] [--save] [--haneen]
"""

import os
import sys
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import balanced_accuracy_score
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
    if arg.startswith('--data='):
        datasetName = arg.split('=', 1)[1]
    elif arg.startswith('--gpu'):
        gpuIndex = int(arg.split('=', 1)[1])
    elif arg.startswith('--width='):
        width = int(arg.split('=', 1)[1])
    elif arg.startswith('--depth='):
        depth = int(arg.split('=', 1)[1])
    elif arg.startswith('--conv='):
        conv = arg.split('=', 1)[1]
    elif arg.startswith('--pool='):
        pool = arg.split('=', 1)[1]
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

###

import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric import __version__ as pygVersion


class CustomQPLayer(nn.Module):
    """Linear layer that concatenates QikProp descriptors to node embeddings before projection.

    Args:
        input_dim: Combined dimension of node embedding + QikProp descriptor vector.
        output_dim: Output feature dimension.
    """

    def __init__(self, input_dim, output_dim):
        super(CustomQPLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, qp_vector):
        full_x = torch.concat((x, qp_vector), dim=1)
        return self.linear(full_x)


def createSomGnn(convName, poolName, width, depth, featureCount, forEmbedding,
                 edge_featureCount=32, edge_attr_exist=False, use_qp=False):
    """Build a sequential GNN for node-level SOM prediction with configurable convolution and pooling.

    Args:
        convName: Graph convolution operator name (e.g. 'gat', 'cheb', 'gin').
        poolName: Graph pooling operator name ('mean', 'max', or 'add').
        width: Hidden layer width.
        depth: Number of graph convolutional layers.
        featureCount: Number of input node features.
        forEmbedding: If True, omit the final output layer (used for embedding extraction).
        edge_featureCount: Number of edge feature dimensions (used for edge-aware convolutions).
        edge_attr_exist: Whether the dataset includes edge attributes.
        use_qp: Whether to append QikProp descriptors via a CustomQPLayer at the end.

    Returns:
        Tuple of (model, used_edge_dim) where model is a torch_geometric.nn.Sequential
        and used_edge_dim indicates whether edge attributes are consumed by the convolution.
    """
    used_edge_dim = False
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
        if edge_attr_exist:
            conv = lambda a, b: gnn.GATv2Conv(a, b, edge_dim=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.GATv2Conv(a, b)
    elif convName == 'resgat':
        if edge_attr_exist:
            conv = lambda a, b: gnn.ResGatedGraphConv(a, b, edge_dim=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.ResGatedGraphConv(a, b)
    elif convName == 'transform':
        if edge_attr_exist:
            conv = lambda a, b: gnn.TransformerConv(a, b, edge_dim=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.TransformerConv(a, b)
    elif convName == 'gine':
        if edge_attr_exist:
            conv = lambda a, b: gnn.GINEConv(nn.Linear(a, b), edge_dim=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.GINEConv(nn.Linear(a, b))
    elif convName == 'crystal':
        if edge_attr_exist:
            conv = lambda a, b: gnn.CGConv((a, b), dim=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.CGConv((a, b))
    elif convName == 'gen':
        if edge_attr_exist:
            conv = lambda a, b: gnn.GENConv(a, b, edge_dim=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.GENConv(a, b)
    elif convName == 'path':
        if edge_attr_exist:
            conv = lambda a, b: gnn.PDNConv(a, b, edge_dim=edge_featureCount, hidden_channels=64)
            used_edge_dim = True
        else:
            raise Exception('Invalid convolution to use without edge attributes')
    elif convName == 'general':
        if edge_attr_exist:
            conv = lambda a, b: gnn.GeneralConv(a, b, in_edge_channels=edge_featureCount)
            used_edge_dim = True
        else:
            conv = lambda a, b: gnn.GeneralConv(a, b)
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
        if not use_qp:
            layerSizes += [1]
        else:
            layerSizes += [77]

    modules = []
    for i in range(len(layerSizes) - 1):
        if not used_edge_dim:
            modules.append((conv(layerSizes[i], layerSizes[i + 1]), 'x, edge_index -> x'))
        else:
            modules.append((conv(layerSizes[i], layerSizes[i + 1]), 'x, edge_index, edge_attr -> x'))
        if forEmbedding or i != len(layerSizes) - 2:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.5))

    if poolName == 'add':
        modules.append((global_add_pool, 'x, batch -> x'))
    elif poolName == 'mean':
        modules.append((global_mean_pool, 'x, batch -> x'))
    elif poolName == 'max':
        modules.append((global_max_pool, 'x, batch -> x'))
    else:
        raise Exception('Unknown pooling operator ' + poolName)

    if use_qp:
        modules.append((CustomQPLayer(128, 1), 'x, qp_vector -> x'))
        if used_edge_dim:
            return gnn.Sequential('x, edge_index, edge_attr, qp_vector, batch', modules), used_edge_dim
        else:
            return gnn.Sequential('x, edge_index, qp_vector, batch', modules), used_edge_dim
    else:
        if used_edge_dim:
            return gnn.Sequential('x, edge_index, edge_attr, batch', modules), used_edge_dim
        else:
            return gnn.Sequential('x, edge_index, batch', modules), used_edge_dim


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

modelName = '%s-%s-%s%dx%d-%depochs' % (modelType, conv, pool, width, depth, epochs)
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

    Automatically detects whether the dataset includes QikProp descriptors by
    checking if node feature count exceeds 140. QikProp features occupy the last
    51 columns of the feature tensor and are stored separately as qp_vector.

    Args:
        mols: List of molecule dicts from the dataset, each containing 'node' data.

    Returns:
        Tuple of (dataList, batchList, edge_attr, used_qp) where dataList is a list
        of PyG Data objects, batchList maps each atom to its molecule index,
        edge_attr indicates whether edge attributes are present, and used_qp
        indicates whether QikProp features were detected.
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
        if molData['x'].shape[1] > 140:  # dataset includes QikProp descriptors
            x_plus_qp = mol['node']['x']
            x = x_plus_qp[:, :-51]
            qp_vector = x_plus_qp[0:1, -51:]
            used_qp = True
        else:
            x = (molData['x'])
            qp_vector = None
            used_qp = False

        y = torch.unsqueeze(torch.FloatTensor(molData['y']), 0)

        atomCount = (molData['x'].size()[0])

        for j in range(int(atomCount)):
            batchList.append(i)

        edgeNum = len(molData['edges'])
        edgeIndex = torch.zeros((2, edgeNum), dtype=torch.int64)
        for j, (a1, a2) in enumerate(molData['edges'][:(len(molData['edges']) // 2)]):
            edgeIndex[0][j * 2] = a1
            edgeIndex[1][j * 2] = a2
            edgeIndex[0][j * 2 + 1] = a2
            edgeIndex[1][j * 2 + 1] = a1

        if 'edge_attr' in molData:
            edge_attributes = molData['edge_attr']
            edge_attr = True
            if used_qp:
                dataList.append(Data(x=x, y=y, edge_index=edgeIndex, edge_attr=edge_attributes, qp_vector=qp_vector, datasetIndex=i))
            else:
                dataList.append(Data(x=x, y=y, edge_index=edgeIndex, edge_attr=edge_attributes, datasetIndex=i))
        else:
            edge_attr = False
            if used_qp:
                dataList.append(Data(x=x, y=y, edge_index=edgeIndex, qp_vector=qp_vector, datasetIndex=i))
            else:
                dataList.append(Data(x=x, y=y, edge_index=edgeIndex, datasetIndex=i))
    return dataList, batchList, edge_attr, used_qp


with open('data/%s.pkl' % datasetName, 'rb') as f:
    records = pickle.load(f)

trainMolecules, batchListT, edge_attr_exist, used_qp = convertMolRecordsToData([records['mols'][i] for i in records['splits'][split]['train'] if i < len(records['mols'])])
validMolecules, batchListV, edge_attr_exist, used_qp = convertMolRecordsToData([records['mols'][i] for i in records['splits'][split]['valid'] if i < len(records['mols'])])

featureCount = trainMolecules[0].x.shape[1]
device = torch.device('cuda:%d' % gpuIndex if torch.cuda.is_available() else 'cpu')
if not edge_attr_exist:
    if not used_qp:
        model, used_edge = createSomGnn(conv, pool, width, depth, featureCount, forEmbedding=False, use_qp=False)
    else:
        model, used_edge = createSomGnn(conv, pool, width, depth, featureCount, forEmbedding=False, use_qp=True)
else:
    edge_featureCount = trainMolecules[0].edge_attr.shape[1]
    if not used_qp:
        model, used_edge = createSomGnn(conv, pool, width, depth, featureCount, forEmbedding=False, edge_featureCount=edge_featureCount, edge_attr_exist=True, use_qp=False)
    else:
        model, used_edge = createSomGnn(conv, pool, width, depth, featureCount, forEmbedding=False, edge_featureCount=edge_featureCount, edge_attr_exist=True, use_qp=True)

model = model.to(device)

num_batches = 16
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

positives = 0
negatives = 0

for mol in trainMolecules:
    p = mol.y.sum()
    n = len(mol.y) - p
    positives += p
    negatives += n
print('Found %d positive and %d negative SOM examples (total %d)' % (positives, negatives, positives + negatives))
posWeight = torch.FloatTensor([negatives / positives]).to(device)

trainLoader = DataLoader(trainMolecules, batch_size=round(len(trainMolecules) // num_batches) + 1)
validLoader = DataLoader(validMolecules, batch_size=len(validMolecules))

batchListTSize = round(len(trainMolecules) // num_batches) + 1
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

batchListTList = [torch.tensor(sublist).to(device) for sublist in batchListTList]
batchListV = torch.tensor(batchListV).to(device)

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
    qp_vector_historical_record = torch.zeros(0, 51).to(device)
    for k, moleculesBatch in enumerate(trainLoader):
        batch = moleculesBatch.to(device)
        optimizer.zero_grad()

        if used_edge:
            if used_qp:
                qp_vector_historical_record = torch.concat((qp_vector_historical_record, batch.qp_vector), dim=0)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, qp_vector_historical_record, batchListTList[k])
            else:
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batchListTList[k])
        else:
            if used_qp:
                qp_vector_historical_record = torch.concat((qp_vector_historical_record, batch.qp_vector), dim=0)
                pred = model(batch.x, batch.edge_index, qp_vector_historical_record, batchListTList[k])
            else:
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
                qp_vector_historical_record = torch.zeros(0, 51).to(device)
                for l, moleculesBatch in enumerate(loader):
                    batch = moleculesBatch.to(device)
                    if loader == trainLoader:
                        if used_edge:
                            if used_qp:
                                qp_vector_historical_record = torch.concat((qp_vector_historical_record, batch.qp_vector), dim=0)
                                pred = model(batch.x, batch.edge_index, batch.edge_attr, qp_vector_historical_record, batchListTList[l])
                            else:
                                pred = model(batch.x, batch.edge_index, batch.edge_attr, batchListTList[l])
                        else:
                            if used_qp:
                                qp_vector_historical_record = torch.concat((qp_vector_historical_record, batch.qp_vector), dim=0)
                                pred = model(batch.x, batch.edge_index, qp_vector_historical_record, batchListTList[l])
                            else:
                                pred = model(batch.x, batch.edge_index, batchListTList[l])
                        pred = pred[batchListTSize * l:]
                    else:
                        if used_edge:
                            if used_qp:
                                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.qp_vector, batchListV)
                            else:
                                pred = model(batch.x, batch.edge_index, batch.edge_attr, batchListV)
                        else:
                            if used_qp:
                                pred = model(batch.x, batch.edge_index, batch.qp_vector, batchListV)
                            else:
                                pred = model(batch.x, batch.edge_index, batchListV)

                    threshold = 0.5

                    predictions.extend(torch.sigmoid(pred).cpu().numpy())
                    trueLabels.extend(moleculesBatch.y.cpu().numpy())

                    adjpredictions = [1 if value >= threshold else 0 for value in predictions]
                    trueLabels = [int(value) for value in trueLabels]

            score = roc_auc_score(trueLabels, predictions)
            acc = balanced_accuracy_score(trueLabels, adjpredictions)

            print('%-15f' % score, end='', flush=True)
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
            if used_edge:
                if used_qp:
                    pred = torch.sigmoid(model(molecule.x.to(device), molecule.edge_index.to(device), molecule.edge_attr.to(device), molecule.qp_vector.to(device), batchlist.to(device))).cpu()
                else:
                    pred = torch.sigmoid(model(molecule.x.to(device), molecule.edge_index.to(device), molecule.edge_attr.to(device), batchlist.to(device))).cpu()
            else:
                if used_qp:
                    pred = torch.sigmoid(model(molecule.x.to(device), molecule.edge_index.to(device), molecule.qp_vector.to(device), batchlist.to(device))).cpu()
                else:
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
