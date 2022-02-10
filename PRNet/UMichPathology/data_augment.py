import torch
import math


def get_partial_data(raw_data, n_partitions=5):
    '''
    Gets n_partitions subsets of the data.
    '''
    x_min = torch.min(raw_data[:, 0]).item()
    x_max = torch.max(raw_data[:, 0]).item() + 1
    y_min = torch.min(raw_data[:, 1]).item()
    y_max = torch.max(raw_data[:, 1]).item() + 1

    distance_to_x_max = (x_max - x_min) / n_partitions
    for lo_x in range(x_min, int(x_min + distance_to_x_max) + 1, int(distance_to_x_max)):
        hi_x = lo_x + distance_to_x_max * n_partitions-1
        yield get_partial_data_proc(raw_data, lo_x, hi_x, y_min, y_max)


def get_partial_data_proc(raw_data, lo_x, hi_x, lo_y, hi_y):
    '''
    Gets a partition of the data at the specified location.
    '''
    partial_data = raw_data[torch.all(
        torch.stack((
            raw_data[:, 0] >= lo_x,
            raw_data[:, 0] <= hi_x,
            raw_data[:, 1] >= lo_y,
            raw_data[:, 1] <= hi_y
        ), dim=1), dim=1
    )]
    partial_data[:, 0] -= lo_x
    partial_data[:, 1] -= lo_y
    return partial_data


def get_rotate_data(raw_data, rotation_count=4):
    '''
    Computes n-way rotation on the data.
    '''
    x_avg = torch.mean(raw_data[:, 0].float())
    y_avg = torch.mean(raw_data[:, 1].float())

    x = raw_data[:, 0] - x_avg
    y = raw_data[:, 1] - y_avg

    ss = torch.sin(torch.arange(0, 2 * math.pi - 2 * math.pi / rotation_count + 1e-4, 2 * math.pi / rotation_count))
    cs = torch.cos(torch.arange(0, 2 * math.pi - 2 * math.pi / rotation_count + 1e-4, 2 * math.pi / rotation_count))

    x_new = torch.outer(x, cs) - torch.outer(y, ss) + x_avg
    y_new = torch.outer(x, ss) + torch.outer(y, cs) + y_avg

    for r_idx in range(rotation_count):
        res = torch.IntTensor(raw_data)
        res[:, 0] = x_new[:, r_idx] - torch.min(x_new[:, r_idx])
        res[:, 1] = y_new[:, r_idx] - torch.min(y_new[:, r_idx])
        yield res