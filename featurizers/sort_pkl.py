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
import deepchem as dc
import numpy as np

remove_drug = list()
elementList = ['*', 'As', 'B', 'Br', 'C', 'Ca', 'Cl', 'Co', 'F', 'Fe', 'Hg', 'I', 'Mg', 'Mn', 'Mo', 'N', 'Ni', 'O', 'P', 'S', 'Se', 'other'] #21
kcfList = ['C0', 'C1a', 'C1b', 'C1c', 'C1d', 'C1x', 'C1y', 'C1z', 'C2a', 'C2b', 'C2c', 'C2x', 'C2y', 'C3a', 'C3b', 'C4a', 'C5a', 'C5x', 'C6a', 'C7a', 'C7x', 'C8x', 'C8y', 'N0', 'N1a', 'N1b', 'N1c', 'N1d', 'N1x', 'N1y', 'N2a', 'N2b', 'N2x', 'N2y', 'N3a', 'N4x', 'N4y', 'N5x', 'N5y', 'O0', 'O1a', 'O1b', 'O1c1', 'O1d', 'O2a', 'O2b', 'O2c', 'O2x', 'O3a', 'O3b', 'O3c', 'O4a', 'O5a', 'O5x', 'O6a1', 'O6a2', 'O7a1', 'O7a2', 'O7x1', 'P1a', 'P1b', 'R', 'S0', 'S1a', 'S2a', 'S2x', 'S3a', 'S3x', 'S4a', 'X', 'Z', 'other'] #72

def organizeData():
    tox = 0
    non_tox = 0
    toxicity = list()
    ids = list()
    all_smiles = list()

    with open('/Users/sammiechum/Downloads/complete_tox_dataset.txt', 'r') as f:
        for line in f:
            line = line.split()
            smiles = line[0]
            identifier = line[1]
            tox = (line[2])
            
            if not smiles in all_smiles: 
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
            #Atom Features Extraction
            x_arr = [] #Whole molecule feature array
            types = [] #Stores kcf types of atoms of the drug
            
            #Use kcfConvoy to get kcf types of atoms in drug
            kcf_drug = kcf.KCFvec()
            kcf_drug.input_smiles(all_smiles[i])
            atom_labels = (kcf_drug.kegg_atom_label)
            for atom in atom_labels:
                kcf_type = atom_labels[atom]['kegg_atom']
                types.append(kcf_type)
            
            # For each atom of the drug get one hot encoding of element, and kcf
            # type as an array 
            for index, atom in enumerate(drug.GetAtoms()):
                atom_arr = dc.feat.graph_features.one_of_k_encoding_unk(atom.GetSymbol(), elementList) + \
                dc.feat.graph_features.one_of_k_encoding_unk(types[index], kcfList) + \
                dc.feat.graph_features.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +\
                [atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetIsAromatic()] +\
                [float(atom.IsInRing())] +\
                [atom.GetExplicitValence(), atom.GetMass(), atom.GetChiralTag(), atom.GetTotalNumHs()]
                  # Append each atom feature array to the feature array for the whole drug
                x_arr.append(atom_arr)
            
            
         
            #Turn x array into tensor 
            x_list = torch.tensor(x_arr)
            x_list = x_list.float()
            elements = [atom.GetSymbol() for atom in drug.GetAtoms()]
            edges = list()
            
            #List of 1s if tox, 0s if nontox
            y_list = torch.tensor([float(toxicity[i])], dtype=torch.float32)
            # print(x_list.size(), y_list.size())
            
            #Edges
            edgeIndex = torch.zeros((2, drug.GetNumBonds() * 2), dtype=torch.float32)
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
    return mols
        
def createSplits(size, removeDrug):
    numbers = list(range(0, size-1))
    random.shuffle(numbers)
    
    trainSize = int(0.8*(size)) 

    train = numbers[:trainSize]
    valid = numbers[trainSize:]
    test = valid
    
    # trainSize = int(0.7*size)
    # validSize = int((size - trainSize) / 2)
    # testSize = size - (trainSize+validSize+1)

    # train = numbers[:trainSize]
    # valid = numbers[trainSize: validSize+trainSize]
    # test = numbers[validSize+trainSize:]
    
    #Remove drugs with invalid mols (from rdkit)
    for position in removeDrug:
        if position in train:
            train.remove(position)
        elif position in valid:
            valid.remove(position)
        elif position in test:
            test.remove(position)
    
    return train, valid, test
    
with open('/Users/sammiechum/Downloads/gnntox/data/dataset_addprop_encode.pkl', 'rb') as f:
    records = pickle.load(f)
    # del records['features']['enzyme']
    for i in range(10):
        train, valid, test = createSplits(7230, remove_drug)
        records['splits'][i]['train'] = train
        records['splits'][i]['valid'] = valid
        records['splits'][i]['test'] = test
    records['features']['element'] = elementList
    records['features']['kcfType'] = kcfList
    records['features']['degree'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #Add in rest of features if needed like above format
    mols = organizeData()
    records['mols'] = mols
        
with open('/Users/sammiechum/Downloads/gnntox/data/dataset2_addprop_encode.pkl', "wb") as file:
    # Dump the dictionary into the pickle file
    pickle.dump(records, file)
    

