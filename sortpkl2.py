#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:08:52 2024

@author: sammiechum
"""

import pickle
import random
from rdkit import Chem
import torch
from torch_geometric.data import Data
import kcfconvoy as kcf

remove_drug = list()
elementList = ['*', 'As', 'B', 'Br', 'C', 'Cl', 'Co', 'F', 'Fe', 'Hg', 'I', 'Mg', 'Mn', 'Mo', 'N', 'Ni', 'O', 'P', 'S', 'Se'] #20
kcfList = ['C0', 'C1a', 'C1b', 'C1c', 'C1d', 'C1x', 'C1y', 'C1z', 'C2a', 'C2b', 'C2c', 'C2x', 'C2y', 'C3a', 'C3b', 'C4a', 'C5a', 'C5x', 'C6a', 'C7a', 'C7x', 'C8x', 'C8y', 'N0', 'N1a', 'N1b', 'N1c', 'N1d', 'N1x', 'N1y', 'N2a', 'N2b', 'N2x', 'N2y', 'N3a', 'N4x', 'N4y', 'N5x', 'N5y', 'O0', 'O1a', 'O1b', 'O1c1', 'O1d', 'O2a', 'O2b', 'O2c', 'O2x', 'O3a', 'O3b', 'O3c', 'O4a', 'O5a', 'O5x', 'O6a1', 'O6a2', 'O7a1', 'O7a2', 'O7x1', 'P1a', 'P1b', 'R', 'S0', 'S1a', 'S2a', 'S2x', 'S3a', 'S3x', 'S4a', 'X', 'Z'] #71

def organizeData():
    tox = 0
    non_tox = 0
    toxicity = list()
    ids = list()
    all_smiles = list()

    with open('/Users/sammiechum/Downloads/test.txt', 'r') as f:
        for line in f:
            line = line.split()
            smiles = line[0]
            identifier = line[1]
            tox = (line[2])
            toxicity.append(tox)
            ids.append(identifier)
            all_smiles.append(smiles)


    #Create node for each compound
    #tracking = id
    #node = x, y 

    mols = list()
    for i in range (len(all_smiles)):
        drug = Chem.MolFromSmiles(all_smiles[i])
        
        if drug is None:
            remove_drug.append(i)
            pass
        else: 
            #Atoms
            elements = [atom.GetSymbol() for atom in drug.GetAtoms()]
            types = list()
            edges = list()
            kcf_drug = kcf.KCFvec()
            kcf_drug.input_smiles(all_smiles[i])
            atom_labels = (kcf_drug.kegg_atom_label)
            for atom in atom_labels:
                kcf_type = atom_labels[atom]['kegg_atom']
                types.append(kcf_type)
            
            #List of 1s if tox, 0s if nontox
            numFeatures = 91
            # x = [[0] * numFeatures for i in range(drug.GetNumAtoms())]
            # for atom in drug.GetAtoms():
            #     el= atom.GetSymbol()
            #     if el in elementList:
            #         elPos = elementList.index(el)
            #         x[atom.GetIdx()][elPos] = 1
            #     kcft = types[atom.GetIdx()]
            #     if kcft in kcfList:
            #         kcfPos = kcfList.index(kcft)+20
            #         x[atom.GetIdx()][kcfPos] = 1
            # x_list = torch.tensor((x), dtype=torch.float32)
            
            # y_list = [int(toxicity[i])]*len(elements)
            x_list = torch.zeros((drug.GetNumAtoms(), numFeatures), dtype=torch.float32)
            for atom in drug.GetAtoms():
                el= atom.GetSymbol()
                if el in elementList:
                    elPos = elementList.index(el)
                    x_list[atom.GetIdx()][elPos] = 1
                kcft = types[atom.GetIdx()]
                if kcft in kcfList:
                    kcfPos = kcfList.index(kcft)+20
                    x_list[atom.GetIdx()][kcfPos] = 1
            
            y_list = torch.tensor([float(toxicity[i])], dtype=torch.float32)
            # print(x_list.size(), y_list.size())
            
            #Edges
            edgeIndex = torch.zeros((2, drug.GetNumBonds() * 2), dtype=torch.int64)
            for bond in drug.GetBonds():
                i = bond.GetIdx()
                edgeIndex[0][i * 2] = bond.GetBeginAtomIdx()
                edgeIndex[1][i * 2] = bond.GetEndAtomIdx()
                edgeIndex[0][i * 2 + 1] = bond.GetEndAtomIdx()
                edgeIndex[1][i * 2 + 1] = bond.GetBeginAtomIdx()
                flat_list = edgeIndex.flatten().tolist()
                edges = [(flat_list[i], flat_list[i + 1]) for i in range(0, len(flat_list), 2)]
            
            node = {
                "elements": elements,
                "types": types,
                "edges": edges,
                "x": x_list,
                "y": y_list
            }
            
            mol = {
                "tracking": ids[i],
                "node": node
            }
            
            mols.append(mol)
    print(len(remove_drug))
    return mols
        
def createSplits(size, removeDrug):
    numbers = list(range(0, size-1))
    random.shuffle(numbers)
    
    validSize = int(1.0*size)
    # validSize = int((size - trainSize))

    valid = numbers[:validSize]
    # valid = numbers[trainSize: validSize+trainSize]
    test = []
    
    #Remove drugs with invalid mols (from rdkit)
    for position in removeDrug:
        if position in valid:
            valid.remove(position)
        # elif position in valid:
        #     valid.remove(position)
    
    return valid
    
with open('/Users/sammiechum/Downloads/gnntox/data/dataset.pkl', 'rb') as f:
    records = pickle.load(f)
    del records['features']['enzyme']
    for i in range(10):
        valid = createSplits(4682, remove_drug)
        # records['splits'][i]['train'] = train
        records['splits'][i]['valid'] = valid
    mols = organizeData()
    records['mols'] = mols
        
with open('/Users/sammiechum/Downloads/gnntox/data/dataset_test.pkl', "wb") as file:
    # Dump the dictionary into the pickle file
    pickle.dump(records, file)
    # print((records['mols'][7]['node']['x']).size())
    # print((records['mols'][7]['node']['y']).size())
    
