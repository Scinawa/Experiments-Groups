import argparse
import glob
import os
import time
import sys
import pickle

import torch
import torch.nn.functional as F
from extended_models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils 

#sys.path.append('/Users/scinawa/workspace/grouptheoretical/new-experiments/multi-orbit-bispectrum')
sys.path.append('/Users/scinawa/workspace/grouptheoretical/multi-orbit-bispectrum-main')
from spectrum_utils import * 
from utils import *

import random

import warnings
warnings.filterwarnings("ignore")


import networkx as nx
import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cpu:0', help='specify cuda devices')
parser.add_argument('--correlation', type=int, default=3, help='which of the k-correlations do we want to use')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--read_dataset', type=bool, default=False, help='Use precomputed k-reduced-skew spectrum')
parser.add_argument('--save_precomputed_skew', type=bool, default=True, help='Save the precomputed k-reduced-skew spectrum')
parser.add_argument('--multi_orbits', type=bool, default=False, help='Add node features to the sksp')
parser.add_argument('--deterministic', type=bool, default=False, help='Make the training deterministic')
parser.add_argument('--model', type=int, default=0, help='Pick a different NN')
parser.add_argument('--a', type=int, default=3, help='Smallest graph we consider')
parser.add_argument('--b', type=int, default=25, help='Biggest graph we consider')


args = parser.parse_args()




def create_dataset(dataset, correlation, args):
    max_n = 0
    lista_cose_belle = []
    real_dataset = []
    len("Original dataset length: {}".format(dataset))

    for i, current_g in enumerate(dataset):

        print("\n\n\n Graph id: {} - ".format(i), end="")
        

        nxgraph = nx.to_numpy_array(torch_geometric.utils.to_networkx(current_g) )
        if nxgraph.shape[0] > max_n:
            max_n = nxgraph.shape[0]

        if (nxgraph.shape[0] > args.a) and (nxgraph.shape[0] <= args.b):
            lista_cose_belle.append(i)


            funzione = FuncOnGroup(nxgraph)


            if args.multi_orbits:
                funzione.add_orbit(np.array([nxgraph[i][i] for i in range(nxgraph.shape[0])]))
                for feature_number in range(dataset[i].x.shape[1]):  
                    funzione.add_orbit(dataset[i].x[:, feature_number])


            skew = reduced_k_correlation(funzione, k=correlation, method="extremedyn", vector=True )

            mezzo = dataset[i].to_dict()
            mezzo['skew']  = skew

            print("- Graph shape:{} - len-skew: {}".format(nxgraph.shape[0], len(skew)), end='')        
            if args.save_precomputed_skew:
                print("Save precomputed skew", end="")
                if args.multi_orbits:
                    print("Multi orbits", end="")
                    path = "{}/{}_{}_mo.pickle".format(args.dataset, correlation, i)
                    if not os.path.isfile(path):
                        with open("{}/{}_{}_mo.pickle".format(args.dataset, correlation, i), 'wb') as handle:
                            print(" dumped-mo ", end="")
                            pickle.dump(skew, handle, protocol=pickle.HIGHEST_PROTOCOL)       
                    else: 
                        print("(file already exist)", end="")           
                else:
                    print("-Single orbit-", end="" )
                    path = "{}/{}_{}.pickle".format(args.dataset, correlation, i)
                    print("-path: {}".format(path), end='')   
                    if not os.path.isfile(path):
                        with open("{}/{}_{}.pickle".format(args.dataset, correlation, i), 'wb') as handle:
                            print(" dumped-so ", end="")
                            pickle.dump(skew, handle, protocol=pickle.HIGHEST_PROTOCOL)       
                    else: 
                        print("(file already exist)", end="")
            else: 
                print("Not save precomputed skew", end="")

            real_dataset.append(torch_geometric.data.Data.from_dict(mezzo))
        else:
            print("(Skipping-{}, shape: {})".format(i, nxgraph.shape[0]), end="", flush=True)
    
    print("\nLen real dataset {}".format(len(real_dataset)))
    print("Biggest matrix in dataset: {}".format(max_n))
    return real_dataset


def read_dataset(dataset, correlation, args):

    lista_cose_belle = []
    max_n = 0 
    real_dataset = []
    len("Original dataset length: {}".format(dataset))

    for i, current_g in enumerate(dataset):
        print(".", end="")

        nxgraph = nx.to_numpy_array(torch_geometric.utils.to_networkx(current_g) )

        if nxgraph.shape[0] > max_n:
            print(max_n)
            print("\n   ")
            max_n = nxgraph.shape[0]

        if (nxgraph.shape[0] > args.a) and (nxgraph.shape[0] <= args.b):
            lista_cose_belle.append(i)

            mezzo = dataset[i].to_dict()
            
            ##
            ## python main PROTEINS
            # Batch(batch=[6674], edge_index=[2, 24804], ptr=[439], x=[6674, 4], y=[438])
            # x.shape = torch.Size([6604, 4])

            # python extended-main PROTEINS
            # Batch(batch=[6634], edge_index=[2, 24652], ptr=[439], skew=[438], x=[6634, 4], y=[438])
            # x.shape = torch.Size([6562, 4])


            # main ENZYMES
            # Batch(batch=[3004], edge_index=[2, 11700], ptr=[166], x=[3004, 21], y=[165])
            # x.shape = torch.Size([3048, 21])


            # extended-main ENZYMES
            # Batch(batch=[3017], edge_index=[2, 11738], ptr=[166], skew=[165], x=[3017, 21], y=[165])
            # x.shape= torch.Size([2987, 21])

            ## 


            if args.multi_orbits:
                with open("{}/{}_{}_mo.pickle".format(args.dataset, correlation, i), 'rb') as handle:
                    skew = pickle.load(handle)
            else: 
                with open("{}/{}_{}.pickle".format(args.dataset, correlation, i), 'rb') as handle:
                    skew = pickle.load(handle)


            mezzo['skew']  = skew

            #print("i: {}, len skew: {}, nxgraph.shape[0]: {}".format(i, len(skew), nxgraph.shape[0]))                                 
            tmp_ = torch_geometric.data.Data.from_dict(mezzo)

            # import IPython
            # IPython.embed()

            real_dataset.append(tmp_)
        else:
            print("(Skipped: {} {})".format(i, nxgraph.shape[0]), end="", flush=True)
   
    print("\nLen of filtered dataset {}".format(len(real_dataset)))
    print("Max element in dataset: {}".format(max_n))
    print("\n")
    return real_dataset



torch.use_deterministic_algorithms(args.deterministic)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
   torch.cuda.manual_seed(args.seed)


old_dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)



if args.read_dataset:
    print("Reading the dataset from file: (args.read_dataset = {})".format(args.read_dataset))
    dataset = read_dataset(old_dataset, args.correlation, args)
else:  
    print("Computing k-reduced-skew spectrum from scratch")
    dataset = create_dataset(old_dataset, args.correlation, args)


#import pdb
#pdb.set_trace()


args.num_classes = old_dataset.num_classes
args.num_features = old_dataset.num_features

args.initial_nodes_skew = len(dataset[0].skew)

print("\n\n\n Initial nodes skew {}".format(args.initial_nodes_skew))

# import IPython
# IPython.embed()
# test passed, all the skew vectors have the same type across different dataset (PROTEINS and NCI1)
# nonok = [i for i,_ in enumerate(dataset) if dataset[i].skew.dtype != "float32"]




num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)





def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth_extended'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth_extended')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth_extended')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training

    
    filez = glob.glob('*.pth_extended')
    for f in filez:
        os.remove(f)
    

    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth_extended'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))

