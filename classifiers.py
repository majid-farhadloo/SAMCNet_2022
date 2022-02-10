from __future__ import print_function
from pyexpat import features

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import torch


path = 'D:\\UMN\Research\\MC-DGCNN\\SpatialPathology\\models\\MC-DGCNN\\'



attention_4_1 = pickle.load( open( path + '\\Results\\Tumor_core\\features\\att4_train_1.p', "rb"))
attention_4_2 = pickle.load( open( path + '\\Results\\Tumor_core\\features\\att4_train_2.p', "rb"))

stack_1 = pickle.load(open(path + '\\Results\\Tumor_core\\features\\stack_add_train_1.p', "rb"))
stack_2 = pickle.load(open(path + '\\Results\\Tumor_core\\features\\stack_add_train_2.p', "rb"))

mini_batch_1 = pickle.load(open(path + '\\Results\\Tumor_core\\features\\mini_batches_1.p', "rb"))
mini_batch_2 = pickle.load(open(path + '\\Results\\Tumor_core\\features\\mini_batches_2.p', "rb"))


attention_4 = {}
stack_add = {}
mini_batch = {}

for key in attention_4_1.keys():
    attention_4[key] = attention_4_1.get(key)
    stack_add[key] = stack_1.get(key)
    mini_batch[key] = mini_batch_1.get(key)
  
for key in attention_4_2.keys():
    attention_4[key+4] = attention_4_2.get(key)
    stack_add[key+4] = stack_2.get(key)
    mini_batch[key+4] = mini_batch_2.get(key)

print(sorted(attention_4.keys()), sorted(stack_add.keys()), sorted(mini_batch.keys()))
features = {}
sample_feat = {}
sample_id = 1

for key1, key2 in zip(stack_add.keys(), attention_4.keys()):
    batch_add = stack_add.get(key1)
    batch_att = attention_4.get(key2)
    for idx in range(batch_add.shape[1]):
        sample_add = batch_add[:,idx,:]
        sample_att = batch_att[idx,:]
        count = 0
        sample_id +=1
        point_id = 1 
        for pts in range(sample_add.shape[1]):
            pairs_add = sample_add[:,pts,:]
            pairs_att = sample_att[:,pts,:]
            print("----------------------------------------------------")
            point_id +=1
            neighbor_features = []
            for neighbor in range(6):
                print("center-", pairs_add[1,neighbor], "-- target-", pairs_add[0,neighbor], "=" " att-", torch.mean(pairs_att[:,neighbor]))
                neighbor_features.append(torch.mean(pairs_att[:,neighbor]).item())
            sample_feat[point_id] = np.mean(neighbor_features)
        features[sample_id] = sample_feat

save_path = 'D:\\UMN\Research\\MC-DGCNN\\SpatialPathology\\models\\MC-DGCNN\\Results\\Tumor_core\\features\\'
pickle.dump(features, open(save_path + "attention_level_4.p", "wb" ))

# features = pickle.load( open( save_path + 'attention_level_4.p', "rb"))
# print()