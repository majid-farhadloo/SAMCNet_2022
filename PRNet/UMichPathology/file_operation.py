import os
import os.path as opath
import pandas as pd
import torch
import numpy as np
import random
from PRNet.helper.spatial_representation import get_pointpairs_representation, get_2d_neighborhood_pairs, \
    get_2d_neighborhood_pairs_from_cache
from PRNet.helper.sampling_helper import sampling_proc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def file_system_scrawl(root_dir, ext=None):
    '''
    Iteratively yields all files in root dir (or the root file)
    '''
    if opath.isfile(root_dir+ext):
        yield root_dir+ext
    elif opath.isdir(root_dir):
        for sub_path in sorted(os.listdir(root_dir)):
            p = opath.join(root_dir, sub_path)
            if opath.isdir(p):
                yield from file_system_scrawl(p)
            elif opath.isfile(p):
                if ext is None or (len(opath.splitext(sub_path)) == 2 and opath.splitext(sub_path)[1] == ext):
                    yield p


def read_file(file_path):
    '''
    Reads a UMich Pathology dataset file
    '''
    input_data = []
    raw_df = pd.read_csv(file_path, sep='\t')
    types = raw_df.columns.tolist()
    types.remove('Cell.X.Position')
    types.remove('Cell.Y.Position')
    types.remove('Other')
    for _, row in raw_df.iterrows():
        for c_idx, col in enumerate(types):
            if row[col] == 'pos':
                entry = [0] * 3
                entry[0] = int(row['Cell.X.Position'])
                entry[1] = int(row['Cell.Y.Position'])
                entry[2] = c_idx
                input_data.append(entry)
    return torch.IntTensor(input_data)


# human_features = np.genfromtxt(r'/content/drive/MyDrive/Colab Notebooks/prs.csv', delimiter=',',
#                               dtype=float)
# human_features = human_features / np.max(human_features)

# file_list = []
# for group_name in ['Anon_Group1', 'Anon_Group2']:
#     for fp in file_system_scrawl(
#             '/content/drive/MyDrive/Research/Current/UMichCancer/Data/{0}/'.format(group_name), '.txt'):
#         file_list.append(fp)


def read_human_feature(file_path):
    return human_features[file_list.index(file_path)]


def split_train_valid(file_paths, file_labels, train_ratio=0.8):
    '''
    Generates a train-val split according to train_ratio
    '''
    xy_pairs = set((p, l) for p, l in zip(file_paths, file_labels))
    train_xys = random.sample(xy_pairs, int(len(file_paths) * train_ratio))
    xy_pairs.difference_update(train_xys)
    return [x for x, _ in train_xys], [x for x, _ in xy_pairs], \
           [y for _, y in train_xys], [y for _, y in xy_pairs]


def get_pointpairs(file_path, raw_data, partial_idx, neighborhood_distance):
    '''
    Gets all pointpairs within the neighborhood distance
    '''
    if opath.exists(file_path + '.neighbor_rep_{}'.format(partial_idx)):
        pointpairs = get_2d_neighborhood_pairs_from_cache(
            file_path + '.neighbor_rep_{}'.format(partial_idx))
    else:
        pointpairs = get_2d_neighborhood_pairs(raw_data, neighborhood_distance,
                                               file_path + '.neighbor_rep_{}'.format(partial_idx))

    return pointpairs


def get_neighborhood_representation(raw_data, pointpairs,
                                    min_grid_scale, max_grid_scale, grid_scale_count,
                                    feature_type_count, sampling_ratio):
    core_point_idxs, selected_pointpair_idxs = sampling_proc(raw_data, pointpairs, feature_type_count, sampling_ratio)
    '''
    Gets the Position Embedding of the pointpairs in the dataset
    '''
    neighborhood_representation = get_pointpairs_representation(raw_data, pointpairs[selected_pointpair_idxs],
                                                                min_grid_scale, max_grid_scale, grid_scale_count)

    return get_neighborhood_tensor(raw_data, pointpairs[selected_pointpair_idxs],
                                   neighborhood_representation, grid_scale_count), core_point_idxs


def get_neighborhood_tensor(raw_data, pointpairs, neighborhood_representation, grid_scale_count):
    '''
    Gets a tensor containing PE
    '''
    return torch.sparse_coo_tensor(pointpairs.T,
                                neighborhood_representation,
                                torch.Size([raw_data.shape[0], raw_data.shape[0], 6 * grid_scale_count]))
