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
        self.initial_nodes_skew = args.initial_nodes_skew

        self.conv1 = GCNConv(self.num_features, self.nhid)
        #self.conv1_ = torch.nn.Conv1d(1, self.nhid, kernel_size=4)
        
        self.skewLin1 = torch.nn.Linear(self.initial_nodes_skew, 1*self.nhid)  


        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 3, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin2b = torch.nn.Linear(self.nhid // 2, self.nhid // 2)

        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None


        try:
            # it works on PROTEINS
            vector_input = torch.from_numpy(np.array(data.skew))  # Assuming skew is your additional input
            print("try ok")
            import pdb
            pdb.set_trace()
        except:
            print("try not ok")
            # this is needed for ENZYMES
            vector_input = torch.from_numpy(np.array(data.skew)[0])  # Assuming skew is your additional input
            import pdb
            pdb.set_trace()
        
        vector_input = vector_input.unsqueeze(1)
        xskew = F.relu(self.skewLin1(vector_input.T))
        xskew = xskew.view(xskew.size(0), -1)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
 

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)



        #Continue with your existing layers
        x = F.relu(x1) + F.relu(x2) + F.relu(x3)  
        # don't attach the skew here because there is a reason why relu is only positive numbers
        # and the skew might also have negative numbers. sooo.... 

        x = torch.cat([x, xskew], dim=1) # attach the skew here



        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = F.relu(self.lin2b(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)


        x = F.log_softmax(self.lin3(x), dim=-1)

        return x




