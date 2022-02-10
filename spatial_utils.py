# from models.PRNet.UMichPathology.file_operation import *
import scipy.spatial as sci_spatial
import numpy as np
import torch
from PRNet.helper.sampling_helper import *
from PRNet.helper.spatial_representation import *
from PRNet.UMichPathology.file_operation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_2d_neighborhood_pairs(raw_data, lengths, neighborhood_distance):
    '''
    Gets all 2d point-pairs within the specified neighborhood distance.
    '''
    t_res=[]
    for b in range(raw_data.shape[1]):
        res = []
        tree_index = sci_spatial.KDTree(raw_data[:lengths[b], b, :2].cpu())
        for cur_idx, n_idxs in enumerate(tree_index.query_ball_tree(tree_index, neighborhood_distance)):
            res.extend([[cur_idx, n] for n in n_idxs])
        t_res.append(torch.LongTensor(res).to(device))
    return t_res

def build_distance_matrix(sample_ptype1, sample_ptype2, d=None):
    '''
    Finds the Euclidean distance between each point of phenotype 1 and each point of phenotype 2.

    Input:
        sample_ptype1:  DataFrame consisting of all points of phenotype 1 in the sample region.
        sample_ptype2:  DataFrame consisting of all points of phenotype 2 in the sample region.
        d:              integer neighborhood distance threshold.

    Output:
        dist:           ndarray containing Euclidean distances between all pairs of points.
    '''
    
    dist = sci_spatial.distance.cdist(sample_ptype1,sample_ptype2)

    # Exclude points outside the distance threshold
    if d != None:
        dist[dist > d] = np.nan

    return dist

def calculate_instances(source_ptype, target_ptypes, d=50):
    '''
    Calculates all instances of phenotype 2 in phenotype 1's neighborhood.

    Input: 
        source_ptype:   DataFrame consisting of all points of source phenotype in the sample region.
        target_ptypes:  list of DataFrames consisting of all points of target phenotypes in the sample region.
        d:              integer neighborhood distance threshold.

    Output:
        instances:      integer number of instances of phenotype 2 in phenotype 1's neighborhood.
    '''

    dists=[]
    for target_ptype in target_ptypes:
        if len(source_ptype) == 0 or len(target_ptype) == 0:
            return 0

        dists.append(build_distance_matrix(np.array((source_ptype.X, source_ptype.invertY)).T, np.array((target_ptype.X, target_ptype.invertY)).T, d))

    # Count all source ptypes points which have target ptype in its neighborhood
    instances=np.ones((len(source_ptype)))
    for dist in dists:
        instances=np.logical_and(instances, (dist>0).any(axis=1))
    
    return np.sum(instances)

def get_neighborhood_representation(data, lengths, pointpairs,
                                    min_grid_scale, max_grid_scale, grid_scale_count,
                                    feature_type_count, sampling_ratio):
    '''
    Gets the Position Embedding of the pointpairs in the dataset
    '''
    t_tensors = []
    t_core_point_pairs = []
    for b in range(data.shape[1]):
        len = lengths[b]
        b_data=data[:len, b]
        b_pointpairs=pointpairs[b]
        core_point_idxs, selected_pointpair_idxs = sampling_proc(b_data, b_pointpairs, feature_type_count, sampling_ratio)
        neighborhood_representation = get_pointpairs_representation(b_data, b_pointpairs[selected_pointpair_idxs],
                                                                    min_grid_scale, max_grid_scale, grid_scale_count)

        n_tensor = get_neighborhood_tensor(b_data, b_pointpairs[selected_pointpair_idxs],
                                    neighborhood_representation, grid_scale_count)
        t_tensors.append(n_tensor)
        t_core_point_pairs.append(core_point_idxs)
    return t_tensors, t_core_point_pairs