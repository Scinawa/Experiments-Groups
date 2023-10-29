import sys
import numpy as np

from tqdm import tqdm
import sys
sys.path.append('..')
#from model import Net
#from utils.config import process_config, get_args
#from utils.basis_transform import basis_transform
#from utils.config import get_config_from_json
import random
import time
import pickle

# from IPython import embed; embed()


sys.path.append('/Users/scinawa/workspace/grouptheoretical/new-experiments/multi-orbit-bispectrum')
from spectrum_utils import * 
from utils import *

import networkx as nx




def generate_random_skew():
    skew_spectrums = {}
    graphs = []

    for k in range(2,7):
        skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]


    for _ in tqdm(range(int(sys.argv[1])), desc="Generating skew spectrums of random graphs"):
        nxgraph = nx.fast_gnp_random_graph(int(sys.argv[2]),float(sys.argv[3]))

        graph = nx.to_numpy_array(nxgraph)
        graphs.append(nxgraph)

        for k in range(2,7):
            print("Creating {}-th correlation".format(k))

            func_1o = create_func_on_group_from_matrix_1orbit(graph)
            #func_2o = create_func_on_group_from_matrix_2orbits(np.array(graph))

            skew_spectrums["1orbit-{}-corre-dict".format(k)].append(
                reduced_k_correlation(func_1o, k=k, method="extremedyn", vector=True))



    ############ USE WITH CARE!!!!!! IT WILL OVERWRITE THE PERVIOUSLY COMPUTED FEATURES FILE
    with open("megadump-random-graphs-features-{}-{}-{}.pickle".format(int(sys.argv[1]), sys.argv[2], sys.argv[3]), 'wb') as handle:
        pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_all_undirected():
    skew_spectrums = {}
    graphs = []

    for k in range(2,8):
        skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]


    for nxgraph in tqdm(nx.graph_atlas_g(), desc="Generating skew spectrums of all graphs"):
        #nxgraph = nx.fast_gnp_random_graph(int(sys.argv[2]),float(sys.argv[3]))

        if nxgraph.number_of_nodes() == 7:
            graph = nx.to_numpy_array(nxgraph)
            graphs.append(nxgraph)

            for k in range(2,8):
                print("Creating {}-th correlation".format(k))

                try:
                    func_1o = create_func_on_group_from_matrix_1orbit(graph)
                    #func_2o = create_func_on_group_from_matrix_2orbits(np.array(graph))

                    skew_spectrums["1orbit-{}-corre-dict".format(k)].append(
                        reduced_k_correlation(func_1o, k=k, method="extremedyn", vector=True))
                except Exception as e:
                    print("Exception: {}".format(e))


    ############ USE WITH CARE!!!!!! IT WILL OVERWRITE THE PERVIOUSLY COMPUTED FEATURES FILE
    with open("megadump-atlas.pickle", 'wb') as handle:
        pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)








if __name__ == "__main__":
    print("Generating the skew-spectra from matrices..")
    print("quantity, nodes, p")
    
    
    #generate_random_skew()
    generate_all_undirected()