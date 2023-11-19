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






def graph_from_fileg6(filepath):
    file1 = open(filepath, 'r')
    Lines = file1.readlines()
    for line in Lines:
        yield nx.from_graph6_bytes(line.strip().encode("utf-8"))




# def the_function_para4(outdegree_sequence):
#     small_graphs_so_far = []
#     tmp = dict()
#     for edges in generate_graphs(outdegree_sequence):
#         g = nx.from_edgelist(edges, create_using=nx.DiGraph)
#         indegree_sequence = tuple(sorted(degree for _, degree in g.in_degree()))
#         if indegree_sequence in tmp:
#             if not any(nx.is_isomorphic(g_before, g) for g_before in tmp[indegree_sequence]):
#                 tmp[indegree_sequence].append(g)
#         else:
#             tmp[indegree_sequence] = [g]
#     for graph in tmp.values():
#         small_graphs_so_far.extend(graph)
#     return small_graphs_so_far


# def version_para4(n):
#     graphs_so_far = list()
#     with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            
#             grafetti = p.starmap(the_function_para4, [(outdegree_sequence,) for outdegree_sequence in combinations_with_replacement(range(n), n)])
#             p.close()
#             p.join()

#     for graphs in grafetti:
#         graphs_so_far.extend(graphs)

#     return graphs_so_far


# def generate_undirected_para(filepath):

#     #### TODO THIS IS NOT YET PARALLEL!!! 
#     skew_spectrums = {}
#     graphs = []

#     for k in range(2,8):
#         skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]


#     for nxgraph in tqdm(graph_from_fileg6(filepath), desc="Generating skew spectrums of all graphs"):


#         graph = nx.to_numpy_array(nxgraph)
#         graphs.append(nxgraph)

#         for k in range(2,8):
#             print("Creating {}-th correlation".format(k))

#             try:
#                 func_1o = create_func_on_group_from_matrix_1orbit(graph)
#                 #func_2o = create_func_on_group_from_matrix_2orbits(np.array(graph))

#                 skew_spectrums["1orbit-{}-corre-dict".format(k)].append(
#                     reduced_k_correlation(func_1o, k=k, method="extremedyn", vector=True))
#             except Exception as e:
#                 print("Exception: {}".format(e))


#     ############ USE WITH CARE!!!!!! IT WILL OVERWRITE THE PERVIOUSLY COMPUTED FEATURES FILE
#     with open("{}-skew.pickle".format(filepath), 'wb') as handle:
#         pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)






















def generate_undirected(filepath):
    skew_spectrums = {}
    graphs = []

    for k in range(2,8):
        skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]


    for nxgraph in tqdm(graph_from_fileg6(filepath), desc="Generating skew spectrums of all graphs"):


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
    with open("{}-skew.pickle".format(filepath), 'wb') as handle:
        pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    print("Generating skew spectra from {}".format(sys.argv[1]) )
    
    
    #generate_random_skew()
    generate_undirected(sys.argv[1])