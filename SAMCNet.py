import math
from sklearn import neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k, self_neighbor=False):
    '''
    Gets the indices of the top K nearest neighbors of x
    '''
    inner = -2*torch.matmul(x.transpose(2, 1), x) # torch.Size([8, 4096, 4096])
    xx = torch.sum(x**2, dim=1, keepdim=True) # torch.Size([8, 1, 4096])
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # torch.Size([8, 4096, 4096])
    if self_neighbor:
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k) : torch.Size([8, 4096, 10])
    else:
        idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]   # (batch_size, num_points, k) : torch.Size([8, 4096, 10])
    return idx
    
device = torch.device('cuda')
class SpatialDGCNN(nn.Module):
    '''
    First layer gets a position embedding of the center vertex, and concats with neighbor's relative distance
    For each layer, attention is applied to the pointpairs based on phenotype
    '''
    def __init__(self, args, num_classes, output_channels):
        super(SpatialDGCNN, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.k = args.k
        self.use_pe = args.use_pe
        self.self_neighbor = args.self_neighbor # to consider self as the first knn
        self.neighbor = args.neighbor # to only consider neighbor and no self-att or no neighbor att and only self-att
        self.num_heads = args.num_heads
        self.positional_encoding = args.PE_dim
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.positional_encoding, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(64), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.feature_pair_attentions1 = [PointpairAttentionLayer(num_classes=num_classes, in_features=64, negative_slope=0.2, self_neighbor = self.self_neighbor, neighbor = self.neighbor) for _ in range(args.num_heads)]
        
        for i, attention in enumerate(self.feature_pair_attentions1): self.add_module('attention1_{}'.format(i), attention)

        if self.num_heads:
            self.conv2 = nn.Sequential(nn.Conv2d(64*2*args.num_heads, 64, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(64), 
                                    nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(64), 
                                    nn.LeakyReLU(negative_slope=0.2))   
             
        self.feature_pair_attentions2 = [PointpairAttentionLayer(num_classes=num_classes, in_features=64, negative_slope=0.2, self_neighbor = self.self_neighbor, neighbor = self.neighbor) for _ in range(args.num_heads)]
        for i, attention in enumerate(self.feature_pair_attentions2): self.add_module('attention2_{}'.format(i), attention)


        
        if self.num_heads:
            self.conv3 = nn.Sequential(nn.Conv2d(64*2*args.num_heads, 128, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(128), 
                                    nn.LeakyReLU(negative_slope=0.2))
        else: 
            self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(128), 
                                    nn.LeakyReLU(negative_slope=0.2))  
            
            
        self.feature_pair_attentions3 = [PointpairAttentionLayer(num_classes=num_classes, in_features=128, negative_slope=0.2, self_neighbor = self.self_neighbor, neighbor = self.neighbor) for _ in range(args.num_heads)]
        for i, attention in enumerate(self.feature_pair_attentions3): self.add_module('attention3_{}'.format(i), attention)

        if self.num_heads:
            self.conv4 = nn.Sequential(nn.Conv2d(128*2*args.num_heads, 256, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(256), 
                                    nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(256), 
                                    nn.LeakyReLU(negative_slope=0.2))
            
        self.feature_pair_attentions4 = [PointpairAttentionLayer(num_classes=num_classes, in_features=256, negative_slope=0.2, self_neighbor = self.self_neighbor, neighbor = self.neighbor) for _ in range(args.num_heads)]
        for i, attention in enumerate(self.feature_pair_attentions4): self.add_module('attention4_{}'.format(i), attention)

        if self.num_heads:
            self.conv5 = nn.Sequential(nn.Conv1d(64*args.num_heads+64*args.num_heads+128*args.num_heads+256*args.num_heads, args.emb_dims, kernel_size=1, bias=False), 
                                    nn.BatchNorm1d(args.emb_dims), 
                                    nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv5 = nn.Sequential(nn.Conv1d(64+64+128+256, args.emb_dims, kernel_size=1, bias=False), 
                                    nn.BatchNorm1d(args.emb_dims), 
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
       
        core_types = x[:,2,:]

        target_features, target_types = self.get_knn_features(x, core_types, k=self.k, spatial_dims=2, self_neighbor = self.self_neighbor)
        x = self.get_spatial_graph_features(x, target_features, k=self.k, spatial_dims=2, use_pe=self.use_pe)
        x = self.conv1(x)

        batch_size, _ , num_points, k  = x.size()
        
        stack = torch.stack((target_types, core_types.view(batch_size, num_points, 1).repeat(1, 1, k)))
        stack = torch.sort(stack, dim=0)[0]
        
        a_x_1 = torch.empty(0, device=device)
        co_efficients_1 = torch.empty(0, device=device)
        for att in self.feature_pair_attentions1:
           attention, co_efficient = att(x, core_types, target_types)   
           a_x_1 = torch.cat((a_x_1,attention), dim= 1)
           co_efficients_1 = torch.cat((co_efficients_1,co_efficient), dim=3)
        if self.num_heads:
            x1 = a_x_1.mean(dim=-1, keepdim=False)
        else:
            x1 = x.mean(dim=-1, keepdim=False)


        target_features, target_types = self.get_knn_features(x1, core_types, k=self.k, self_neighbor = self.self_neighbor)
        x = self.get_spatial_graph_features(x1, target_features, k=self.k)
        x = self.conv2(x)
        
        a_x_2 = torch.empty(0, device=device)
        co_efficients_2 = torch.empty(0, device=device)
        for att in self.feature_pair_attentions2:
           attention, co_efficient = att(x, core_types, target_types)   
           a_x_2 = torch.cat((a_x_2,attention), dim= 1)
           co_efficients_2 = torch.cat((co_efficients_2,co_efficient), dim=3)
        
        if self.num_heads:
            x2 = a_x_2.mean(dim=-1, keepdim=False) 
        else:
            x2 = x.mean(dim=-1, keepdim=False)

        target_features, target_types = self.get_knn_features(x2, core_types, k=self.k, self_neighbor = self.self_neighbor)
        x = self.get_spatial_graph_features(x2, target_features, k=self.k)
        x = self.conv3(x)
        
        a_x_3 = torch.empty(0, device=device)
        co_efficients_3 = torch.empty(0, device=device)
        for att in self.feature_pair_attentions3:
           attention, co_efficient = att(x, core_types, target_types)   
           a_x_3 = torch.cat((a_x_3,attention), dim= 1)
           co_efficients_3 = torch.cat((co_efficients_3,co_efficient), dim=3)
        if self.num_heads:
            x3 = a_x_3.mean(dim=-1, keepdim=False)
        else:
            x3 = x.mean(dim=-1, keepdim=False)
        target_features, target_types = self.get_knn_features(x3, core_types, k=self.k, self_neighbor = self.self_neighbor)
        x = self.get_spatial_graph_features(x3, target_features, k=self.k)
        x = self.conv4(x)
        
        a_x_4 = torch.empty(0, device=device)
        co_efficients_4 = torch.empty(0, device=device)
        for att in self.feature_pair_attentions4:
           attention, co_efficient = att(x, core_types, target_types)   
           a_x_4 = torch.cat((a_x_4,attention), dim= 1)
           co_efficients_4 = torch.cat((co_efficients_4,co_efficient), dim=3)
        if self.num_heads:   
            x4 = a_x_4.mean(dim=-1, keepdim=False)
        else:
            x4 = x.mean(dim=-1, keepdim=False)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_5 = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x_5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x_5, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, a_x_4, stack

    def get_knn_features(self, x, core_types, k=20, spatial_dims=None, idx=None, self_neighbor=False):
        '''
        Gets the features of the top K nearest neighbors of x
        '''
        batch_size, num_dims, num_points  = x.size()
        x = x.view(batch_size, -1, num_points)

        if spatial_dims is None:
            spatial_dims = spatial_dims

        if idx is None:
            idx = knn(x[:,:spatial_dims], k=k, self_neighbor = self_neighbor)   # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        target_features = x.view(batch_size*num_points, -1)[idx, :]
        target_types = core_types.reshape(batch_size*num_points)[idx]

        target_features = target_features.view(batch_size, num_points, k, num_dims) # (batch_size, num_points, k, num_dims)
        target_types = target_types.view(batch_size, num_points, k)
        
        return target_features, target_types

    def get_spatial_graph_features(self, x, knn_features, k=20, spatial_dims=None, use_pe=False):
        '''
        Gets the graph features based on pairwise distance (not mine)
        '''    
        x = x.permute(0, 2, 1)
        batch_size, num_points, num_dims = x.size()
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # (batch_size, num_points, k, num_dims)
        if spatial_dims is not None:
            num_dims = spatial_dims
            x, knn_features = x[:,:,:,:num_dims], knn_features[:,:,:,:num_dims]

        if use_pe:
            # pe_features = (knn_features-x)
            # x_features = self.get_neighborhood_representation(x)
            center_node = self.get_neighborhood_representation(x)
            target_node = self.get_neighborhood_representation(knn_features)
            pe_features = (target_node-center_node)

            features = torch.cat((pe_features, x), dim=3).permute(0, 3, 1, 2).contiguous()
        else:
            features = torch.cat((knn_features-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return features

    def get_neighborhood_representation(self, distances, min_grid_scale=1, max_grid_scale=100, grid_scale_count=5):
        '''
        Gets the Position Embedding of the pointpairs in the dataset based on relative distance
        '''
        a = torch.tensor([[1, 0], [-0.5, -math.sqrt(3) / 2], [-0.5, math.sqrt(3) / 2]], device=device)
        scales = torch.tensor([min_grid_scale * (max_grid_scale / min_grid_scale) ** (s / (grid_scale_count - 1)) for s in range(grid_scale_count)], device=device)
        scaled_proj = torch.einsum('bnkr, s -> bnkrs', torch.matmul(distances.float(), a.T), 1 / scales)
        neighborhood_representation = torch.stack((torch.cos(scaled_proj), torch.sin(scaled_proj)), dim=4)
        neighborhood_representation = neighborhood_representation.reshape((scaled_proj.shape[0], scaled_proj.shape[1], scaled_proj.shape[2], -1))

        return neighborhood_representation


class PointpairAttentionLayer(nn.Module):
    """
    Attention layer, where each pointpair or self gets its own attention coefficient
    """
    def __init__(self, num_classes, in_features, negative_slope, self_neighbor, neighbor):
        super(PointpairAttentionLayer, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.negative_slope = negative_slope
        self.self_neighbor = self_neighbor
        self.neighbor = neighbor
        
        self.W = nn.Parameter(torch.empty(size=(in_features, in_features), device=device)) # Linear transformation matrix
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.n_perms = sum([self.num_classes-x for x in range(self.num_classes)]) # Number of order-invariant pointpair permutations
        
        if self.neighbor:
            self.a_pair = nn.Parameter(torch.empty(size=(self.n_perms, in_features), device=device))
            nn.init.xavier_uniform_(self.a_pair, gain=1.414)
            
        if self.self_neighbor:
            self.a_self = nn.Parameter(torch.empty(size=(self.num_classes, in_features), device=device)) # Self-attn
            nn.init.xavier_uniform_(self.a_self, gain=1.414)

        self.linear = nn.Linear(in_features, 1) ## for visualization purposes

        self.leakyrelu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x, core_types, target_types, self_neighbor = False, neighbor = True):
        batch_size, in_features, num_points, k = x.size()

        # Linear transformation to help apply attention across an alternative feature space
        Wh = torch.matmul(x.permute(0, 2, 3, 1), self.W)

        # Stack each pointpair by phenotype codes and sort each pair for permutation invariance
        stack = torch.stack((target_types, core_types.view(batch_size, num_points, 1).repeat(1, 1, k)))
        stack = torch.sort(stack, dim=0)[0]

        # a = torch.ones(size=(batch_size, num_points, k, in_features), device=device) * self.a_pad # Default pad coefficients
        
        a = torch.ones(size=(batch_size, num_points, k, in_features), device=device) 
        n=0
        for i in range(self.num_classes):
            if self.self_neighbor:
                a[:,:,:1][core_types == i] = self.a_self[i] # Assign self-attn coefficients
            if self.neighbor:
                for j in range(i, self.num_classes):
                    if self_neighbor:
                        a[:,:,1:][(stack[0,:,:,1:] == i) & (stack[1,:,:,1:] == j)] = self.a_pair[n] # Assign pointpair coefficients
                    else:
                        a[(stack[0,:,:,:] == i) & (stack[1,:,:,:] == j)] = self.a_pair[n] # Assign pointpair coefficients
                    n+=1

        e = self.leakyrelu(Wh * a) # Create pointpair attention multipliers between nodes
        
        att_viz = self.linear(e)
        attention_viz = F.softmax(att_viz, dim=2) # Puts attention values across pointpairs into a similar [0, 1] reference frame
        
        attention = F.softmax(e, dim=2) # Puts attention values across pointpairs into a similar [0, 1] reference frame
        
        
        h_prime = attention * Wh # Broadcast pointpair attention values for each pointpair edge
        return F.elu(h_prime.permute(0, 3, 1, 2)), attention # Permute back into the convolvable shape and return w/ elu