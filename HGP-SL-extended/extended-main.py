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

sys.path.append('/Users/scinawa/workspace/grouptheoretical/new-experiments/multi-orbit-bispectrum')
from spectrum_utils import * 
from utils import *

import random
#random.seed(3)
#torch.manual_seed(3)

import warnings
warnings.filterwarnings("ignore")
#torch.use_deterministic_algorithms(True)



import networkx as nx
import tqdm




def export_dataset_to_file(dataset, dataset_name, with_node_features=False):

    lista_cose_belle = []
    real_dataset = []
    len("Original dataset length: {}".format(dataset))

    # import pdb
    # pdb.set_trace()

    for i, current_g in enumerate(dataset):

        nxgraph = nx.to_numpy_array(torch_geometric.utils.to_networkx(current_g) )
        if (nxgraph.shape[0] <= 59) and (nxgraph.shape[0] > 2):
            print(".", end="")
            lista_cose_belle.append(i)

            if with_node_features ==False:
                with open("{}/{}.pickle".format(dataset_name, i), 'wb') as handle:
                    pickle.dump(nxgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else: 
                with open("{}/{}.pickle".format(dataset_name, i), 'wb') as handle:
                    pickle.dump((nxgraph, dataset[i].x ), handle, protocol=pickle.HIGHEST_PROTOCOL)


        else:
            print("(S-{}-{})".format(i, nxgraph.shape[0]), end="", flush=True)
    print("\nLen real dataset {}".format(len(lista_cose_belle)))
    print("Exported dataset to file")
    return None








def create_dataset(dataset, correlation, args):

    lista_cose_belle = []
    real_dataset = []
    len("Original dataset length: {}".format(dataset))

    for i, current_g in enumerate(dataset):

        #import pdb
        #pdb.set_trace()

        nxgraph = nx.to_numpy_array(torch_geometric.utils.to_networkx(current_g) )
        if (nxgraph.shape[0] <= 25) and (nxgraph.shape[0] > 2):
            print(".", end="")
            lista_cose_belle.append(i)


            ##### CASE 2 we do everything on a single machine
            #func_orbit = create_func_on_group_from_matrix_1orbit(nxgraph)
            funzione = FuncOnGroup(nxgraph)
            funzione.add_orbit(np.array([nxgraph[i][i] for i in range(nxgraph.shape[0])]))

            for feature_number in range(dataset[i].x.shape[1]):  
                funzione.add_orbit(dataset[i].x[:, feature_number])


            skew = reduced_k_correlation(funzione, k=correlation, method="extremedyn", vector=True )

            mezzo = dataset[i].to_dict()
            
            with open("{}/{}_{}.pickle".format(args.dataset, correlation, i), 'rb') as handle:
                skew = pickle.load(handle)
                mezzo['skew']  = skew


            print("len skew {}, graph shape]:{}".format(len(skew), nxgraph.shape[0]))                                 

            real_dataset.append(torch_geometric.data.Data.from_dict(mezzo))
        else:
            print("(S-{}-{})".format(i, nxgraph.shape[0]), end="", flush=True)
    print("\nLen real dataset {}".format(len(real_dataset)))

    return real_dataset


def read_dataset(dataset, correlation, args):

    lista_cose_belle = []
    real_dataset = []
    len("Original dataset length: {}".format(dataset))

    for i, current_g in enumerate(dataset):


        nxgraph = nx.to_numpy_array(torch_geometric.utils.to_networkx(current_g) )
        if (nxgraph.shape[0] <= 59) and (nxgraph.shape[0] > 2):
            print(".", end="")
            lista_cose_belle.append(i)

            mezzo = dataset[i].to_dict()
            


            if args.multi_orbits:
                with open("{}/{}_{}_mo.pickle".format(args.dataset, correlation, i), 'rb') as handle:
                    skew = pickle.load(handle)
            else: 
                with open("{}/{}_{}.pickle".format(args.dataset, correlation, i), 'rb') as handle:
                    skew = pickle.load(handle)

            mezzo['skew']  = skew

            print("len skew {}, nxgraph.shape[0]:{}".format(len(skew), nxgraph.shape[0]))                                 

            real_dataset.append(torch_geometric.data.Data.from_dict(mezzo))
        else:
            print("(S-{}-{})".format(i, nxgraph.shape[0]), end="", flush=True)
    print("READ DATASET")
    print("\nLen real dataset {}".format(len(real_dataset)))

    return real_dataset





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
parser.add_argument('--export_dataset', type=bool, default=False, help='Export the dataset so you can create the mosksp')
parser.add_argument('--multi_orbits', type=bool, default=False, help='Add node features to the sksp')



args = parser.parse_args()

#torch.manual_seed(args.seed)

#if torch.cuda.is_available():
#    torch.cuda.manual_seed(args.seed)

print("\n\n Working with {}\n\n".format(args.dataset))

old_dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)



if args.read_dataset:
    dataset = read_dataset(old_dataset, args.correlation, args)
else:  
    print("Computing k-reduced-skew spectrum from scratch")
    dataset = create_dataset(old_dataset, args.correlation, args)


    ##### fast sksp on my machines to play with. 
    if args.save_precomputed_skew:  
        with open("TUDataset-{}-{}-skew.pickle".format(args.dataset, args.correlation), 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved dataset")


#### export dataset to be processed on another machine
if args.export_dataset:
    export_dataset_to_file(old_dataset, args.dataset, args.multi_orbits)
    sys.exit(0)



args.num_classes = old_dataset.num_classes
args.num_features = old_dataset.num_features

args.initial_nodes_skew = len(dataset[0].skew)




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

