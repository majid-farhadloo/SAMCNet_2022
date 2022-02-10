import torch
import math
import scipy.spatial as sci_spatial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_2d_neighborhood_pairs_from_cache(cache_file_path):
    '''
    Gets all 2d point-pairs from the cache
    '''
    with open(cache_file_path) as cache_file:
        res = [[ori_idx, int(dest_idx)] for ori_idx, line in enumerate(cache_file) for dest_idx in line.split(',')]
    return torch.LongTensor(res)


def get_2d_neighborhood_pairs(raw_data, neighborhood_distance):
    '''
    Gets all 2d point-pairs within the specified neighborhood distance.
    '''
    tree_index = sci_spatial.KDTree(raw_data[:, :2])
    res = []
    for cur_idx, n_idxs in enumerate(tree_index.query_ball_tree(tree_index, neighborhood_distance)):
        res.extend([[cur_idx, n] for n in n_idxs])
    return torch.LongTensor(res)


def get_pointpairs_representation(raw_data, pointpairs,
                                  min_grid_scale, max_grid_scale, grid_scale_count):
    '''
    Calculates Position Embeddings for the point-pairs.
    '''
    xs = (raw_data[pointpairs[:, 1], :2] - raw_data[pointpairs[:, 0], :2]).to(device)
    a = torch.tensor([[1, 0], [-0.5, -math.sqrt(3) / 2], [-0.5, math.sqrt(3) / 2]]).to(device)
    scales = torch.tensor([min_grid_scale * (max_grid_scale / min_grid_scale) ** (s / (grid_scale_count - 1))
                        for s in range(grid_scale_count)]).to(device)
    scaled_proj = torch.einsum('qr, p->qrp', torch.matmul(xs.float(), a.T), 1 / scales)
    return torch.stack((torch.cos(scaled_proj), torch.sin(scaled_proj)), dim=3).reshape((scaled_proj.shape[0], -1))
