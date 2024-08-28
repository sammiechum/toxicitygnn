
import pickle
import csv
import numpy as np
import json
import FeatureLoaders
from FeaturePreprocessor import FeaturePreprocessor
from AddFeatures import AddFeatures

# Creates a molecule to index mapping needed to load the qikprop data
def create_qikprop_id_map(original_pkl, qikprop_data, qikprop_id_map, modified_qikprop_data):
    tracking_id = list()
    first_column = list()
    
    # Get the list of tracking IDs of all molecules in the pkl dataset
    with open(original_pkl, 'rb') as file:
        data = pickle.load(file)
        
        for i in range (len(data['mols'])):
            tracking_id.append((data['mols'][i]['tracking']))
    
    
    index_list = list()
    count = -1
    prev = ""
    
    # Get the list of tracking IDs of the qikprop dataset
    with open(qikprop_data, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                if not (row[0].split('-')[0] == prev):
                    count+=1
                first_column.append((row[0].split('-')[0]))
                index_list.append(count)
                prev = (row[0].split('-')[0])
                
    # Create a dictionary to match the IDs of each molecule to the index in the qikprop dataset
    dict = {}
    
    for i in range (len(first_column)):
        dict[first_column[i]] = index_list[i]
    
    with open(qikprop_id_map, 'w') as json_file:
        json.dump(dict, json_file, indent=4)
        
    # Replace the IDs in the qikprop dataset with indices
    with open(qikprop_data, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)
    
    for row in data:
        identifier = row[0].split('-')[0]
        if identifier in dict:
            row[0] = dict[identifier]
    
    # Write the modified data back to a CSV file
    with open(modified_qikprop_data, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)
        
    
# Load qikprop features into pkl file
def loadQikProp(original_pkl, modified_qikprop_data, qikprop_id_map, output_file):
    qikPropLoad = FeatureLoaders.QikPropLoader(
        original_pkl,  
        modified_qikprop_data, 
        qikprop_id_map) 
    
    qikPropFeat = FeaturePreprocessor(qikPropLoad)
    feats = qikPropFeat.getFeatures()
    
    #Print analysis of features
    print(f"There are {len(feats)} features")
    print(feats)
    print(f"{len(qikPropFeat.getIncompleteFeatures()) / len(qikPropFeat.getFeatures()) * 100:.1f}% of features are missing at least one observation")
    print(f"{len(qikPropFeat.getIncompleteRows()) / len(qikPropFeat.getDescriptors()) * 100:.1f}% of observations are missing at least one feature")
    qikPropFeat.getDistribution("#stars", visual=True, savePath="stars.png")
    
    #Normalize data/features
    qikPropFeat.normalize("QikProp_tox_means.csv", "QikProp_tox_stds.csv")
    
    #Fill in missing features with average values
    qikPropFeat.avgFill()
    
    #Add qikprop features to to the original dataset pkl file (reformatted so x is a list)
    adderQikProp = AddFeatures(original_pkl)
    adderQikProp.append(qikPropFeat.getDescriptors(), "QikProp")
    #Save to a new pkl file
    adderQikProp.save(output_file)



# An example of how to use the two functions to add qikprop features to a clintox dataset
original_pkl = '../data/clintox/addAtom_addEdge_clintox.pkl' #Dataset before adding qikprop properties
qikprop_data = "../data/clintox/modified_clintox_qikprop.CSV" #Dataset of qikprop properties
qikprop_id_map = "../data/clintox/clintox_qikprop_mapping.CSV" #Generated id mapping from function above
modified_qikprop_data = "../data/clintox/postmodified_clintox_qikprop.CSV" #Dataset of qikprop properties after editing indices
output_file = "../data/clintox/addAtom_addEdge_qikprop_clintox.pkl" #Final featurized dataset with qikprop

create_qikprop_id_map(original_pkl, qikprop_data, qikprop_id_map)
loadQikProp(original_pkl, modified_qikprop_data, qikprop_id_map, output_file) 
