import torch
import torch.nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BinaryContextLayer(torch.nn.Module):
    """
    docstring
    """

    def __init__(self, feature_type_count):
        super(BinaryContextLayer, self).__init__()
        self.feature_type_count = feature_type_count

    def forward(self, raw_data, neighborhood_tensor,
                core_point_idxs):
        """
        docstring
        """

        def sparse_dense_mul(s_tensor_3d, d_tensor_2d):
            idxs = s_tensor_3d._indices()
            vals = s_tensor_3d._values()
            d_vals = d_tensor_2d[idxs[0, :], idxs[1, :]]  # get values from relevant entries of dense matrix
            return torch.sparse_coo_tensor(idxs, vals * d_vals.view(-1, 1), s_tensor_3d.shape, device=device)

        def sparse_max(s_tensor_3d):
            idxs = s_tensor_3d._indices()
            vals = s_tensor_3d._values()
            origins = torch.unique(idxs[0])
            max_res = torch.zeros(torch.Size([raw_data.shape[0], vals.shape[1]]), device=device)
            for o in origins:
                max_res[o] = torch.max(vals[idxs[0] == o], dim=0)[0]
            return max_res

        res = torch.zeros(torch.Size([
            torch.sum(core_point_idxs),
            self.feature_type_count,
            neighborhood_tensor.shape[2]]), device=device)

        for feature_idx in range(self.feature_type_count):
            adj_mat = torch.zeros(torch.Size([raw_data.shape[0], raw_data.shape[0]]), device=device)
            adj_mat[:, raw_data == feature_idx] = 1
            res[:, feature_idx, :] = sparse_max(sparse_dense_mul(neighborhood_tensor,
                                                                 adj_mat))[core_point_idxs]

        return res
