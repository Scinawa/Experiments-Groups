import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
import numpy as np

from layers import GCN, HGPSLPool


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv1_ = torch.nn.Conv1d(1, self.nhid, kernel_size=4)

        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    # def forward(self, data):
    #     x, edge_index, batch, skew = data.x, data.edge_index, data.batch, data.skew
    #     edge_attr = None

    #     import pdb
    #     pdb.set_trace()

    #     added_layer = self.conv1_((torch.from_numpy(np.array(skew))))

    #     x = F.relu(torch.cat([self.conv1(x, edge_index, edge_attr), added_layer], dim=1) )
    #     x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
    #     x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

    #     x = F.relu(self.conv2(x, edge_index, edge_attr))
    #     x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
    #     x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

    #     x = F.relu(self.conv3(x, edge_index, edge_attr))
    #     x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

    #     x = F.relu(x1) + F.relu(x2) + F.relu(x3)

    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=self.dropout_ratio, training=self.training)
    #     x = F.relu(self.lin2(x))
    #     x = F.dropout(x, p=self.dropout_ratio, training=self.training)
    #     x = F.log_softmax(self.lin3(x), dim=-1)

    #     return x


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        #import IPython
        #IPython.embed()

        import pdb
        pdb.set_trace()

        # Process the additional vector input using 1D convolution
        vector_input = torch.from_numpy(np.array(data.skew))  # Assuming skew is your additional input
        vector_out = F.relu(self.conv1_(vector_input.unsqueeze(1)))  # Reshape for Conv1d

        # Flatten or reshape vector_out to match the dimensions of x1
        vector_out = vector_out.view(vector_out.size(0), -1)


        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Now you can concatenate x1 and vector_out
        x1 = torch.cat([x1, vector_out], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #import pdb
        #pdb.set_trace()

        # Concatenate graph-based features and the processed vector
        #combined = torch.cat([x1, x2, x3, vector_out], dim=1)

        # Continue with your existing layers
        #x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x3))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x





