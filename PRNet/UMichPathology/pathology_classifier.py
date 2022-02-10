import torch
import torch.nn
from PRNet.pr_net.pr_net import PRNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PathologyClassifier(torch.nn.Module):
    """
    docstring
    """

    def __init__(self, feature_type_count, grid_scale_count, output_classes,
                 pr_representation_dim=32, pp_representation_dim=128,
                 apply_attention=True,
                 leaky_relu_slope=0.2):
        super(PathologyClassifier, self).__init__()

        pr_representation_dim = grid_scale_count if pr_representation_dim is None else pr_representation_dim
        self.pr_net = PRNet(feature_type_count,
                            grid_scale_count,
                            pr_representation_dim,
                            pp_representation_dim,
                            apply_attention,
                            leaky_relu_slope)
        self.pr_representation_dim = pr_representation_dim
        self.classify_linear1 = torch.nn.Linear(feature_type_count * feature_type_count * pr_representation_dim, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.dp1 = torch.nn.Dropout(p=0.5)
        self.classify_linear2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.dp2 = torch.nn.Dropout(p=0.5)
        self.classify_linear3 = torch.nn.Linear(256, output_classes)
        self.classify_activate = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.final = torch.nn.Softmax(dim=0)

    def forward(self, raw_data,
                neighborhood_tensor,
                core_point_idxs):
        """
        docstring
        """
        pr_representations = self.pr_net(raw_data, neighborhood_tensor, core_point_idxs)

        x = self.classify_activate(self.classify_linear1(pr_representations))
        x = self.dp1(x)
        x = self.classify_activate(self.classify_linear2(x))
        x = self.dp2(x)
        # return self.final(self.classify_linear3(x)), pr_representations
        return self.classify_linear3(x), pr_representations


