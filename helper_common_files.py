import pandas as pd
import scipy
import scipy.io
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sklearn as sk

import time
from sklearn.decomposition  import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#from spectrum_utils import *
import numpy as np
import warnings
import copy


######## Machine learning stuff


def PCA_everything(dataset, ncomponents):
    pca = PCA(n_components= ncomponents)
    return pca.fit_transform(dataset)


def clean_it(skew_spectra):
    return  np.round(skew_spectra, decimals=6)

# def make_example(kcorrelation, stop_at_length=None):

#     if stop_at_length:
#         indices = list(range(len(kcorrelation[:stop_at_length])))    # limit here the size of the final tree, i.e. the number of grpahs we consider
#     else:
#         indices = list(range(len(kcorrelation)))

#     newlist =  []

#     for i in indices:
#         newlist.append(kcorrelation[i])

#     random.Random(4).shuffle(indices)

#     for i in indices:
#         newlist.append(kcorrelation[i])

#     print("len of newlist", len(newlist))
#     return np.array(newlist)



def pollute_with_isomorphic_graphs(raw_dataset, number_collisions_to_add, kcorre_names):
    """
    Add number_collisions_to_add isomorphic graphs to dataset
    both in the graphs and in the k-r-s-spectra
    """
    dataset_length = len(raw_dataset[0])
    #print(dataset_length)
    indices = random.choices(range(dataset_length), k=number_collisions_to_add)

    for i in indices:
        raw_dataset[0].append(raw_dataset[0][i])
        for kcorre_name in kcorre_names:
            raw_dataset[1][kcorre_name].append(raw_dataset[1][kcorre_name][i])

    return raw_dataset




def create_T_table_2(k_correlations, kcorre_names):
    """
    This function creates the T table for the k_correlations
    Very slow function, use only for small datasets
    """
    T_sets = {}

    for kcorre_name in kcorre_names:
        T_sets[kcorre_name]=[]

        indices = set(range(len(k_correlations[kcorre_name])))
        #print("indices", indices)

        indexlen = len(indices)
        dista = sk.metrics.pairwise_distances(k_correlations[kcorre_name])
        i = 0
        print("--- computed distances ---")
        while True:
            row = random.choice(list(indices))


            #agiowhere = np.where(
            #    np.isclose(dista[row], np.zeros(len(dista[row])))
            #    )
            #print(list(agiowhere[0]))

            #print(  np.where(  np.isclose(dista[row], np.zeros(len(dista[row])) ,  atol=1e-03    ) )    )


            group = set(np.where(
                np.isclose(dista[row], np.zeros(len(dista[row])),  atol=1e-04 )
                )[0]           )
            


            #print("row: {} group: {}, indexlen: {}".format(row, group, len(indices)))
            
            T_sets[kcorre_name].append(group)
            
            indices = indices - group

            if len(indices)==0:
                print("Finished clustering!")
                break
            else:
                i=i+1
                if i > indexlen:
                    print("Warning: numerical error in distances, exiting")
                    break

    return T_sets   


def create_T_table(k_correlations, kcorre_names):
    """
    Same thing as before, but super fast! 

    """
    T_tmp = {}
    T_sets = {}

    for kcorre_name in kcorre_names:

        T_tmp[kcorre_name]={}

        # Iterate through the skew_spectra and populate the dictionary
        for index, vector in enumerate(k_correlations[kcorre_name]):
            #print(index)
            # Convert the vector to a tuple to use it as a dictionary key :) 
            vector_tuple = tuple(vector)

            if vector_tuple in T_tmp[kcorre_name]:
                # If the vector representation exists in the dictionary, append the index
                T_tmp[kcorre_name][vector_tuple].append(index)
            else:
                # Oth.. create a new entry with the index
                T_tmp[kcorre_name][vector_tuple] = [index]

        # Finally, convert the dictionary values to sets
        T_sets[kcorre_name] = [set(indices) for indices in T_tmp[kcorre_name].values()]
    return T_sets



def build_networkx_graph(T, maxk):
    G = nx.DiGraph()
    G.add_node((0,0))

    breakout = False

    for k in range(maxk, 1, -1):
        #print(k)
        for subsetz in T['1orbit-{}-corre-dict'.format(k+1)]:
            for setz in T['1orbit-{}-corre-dict'.format(k)]:
                #print("k:", k, "subset", subsetz, "set:", setz, "k:", k)


                for node in subsetz:                
                    if {node}.issubset(setz):
                        G.add_edge((k, ) + tuple(setz), (k+1,) + tuple(subsetz) )
                    else:
                        pass 
                        #G.add_node( (k+1,) + tuple(subsetz)  )
                        #print("anomaly detected")
            

    for setz in T['1orbit-2-corre-dict']:
        G.add_edge((0,0), (2,) + tuple(setz))
    return G



def graph_subtraction_nodes(G, Gbfs):
    Gclone = G.copy()
    for node in Gbfs:
        Gclone.remove_node(node)

    return Gclone


def graph_subtraction_edges(G, Gbfs):
    Gclone_edges = G.copy()
    for edge in Gbfs.edges():
        Gclone_edges.remove_edge(*edge)
    return Gclone_edges


def count_bifurcations(grafetto, kcorre_names):
    histogram = {kcorre_name : 0 for kcorre_name in kcorre_names}
    histogram['1orbit-0-corre-dict'] = 0
    hits = []
    lottery_tickets = []
    for node in grafetto.nodes():
        if grafetto.out_degree(node)>1:
            histogram['1orbit-{}-corre-dict'.format(node[0])] +=  grafetto.out_degree(node)  # TODO is this right? consider {1,2,3,4} -> {1,2,3}, {4} or {1,2,3,4} -> {1,2}, {3}, 
            hits.append(node)
            if node[0] > 2:      # we can distinguish graphs in this node in the NEXT k of correlation
                lottery_tickets.append(node)
    return histogram, hits, lottery_tickets



def count_collisions(grafetto, kcorre_names):
    histogram = {kcorre_name : 0 for kcorre_name in kcorre_names}
    histogram['1orbit-0-corre-dict'] = 0
    hits = []
    for node in grafetto.nodes():
        #print(grafetto.in_degree(node))
        if grafetto.in_degree(node)>1:
            histogram['1orbit-{}-corre-dict'.format(node[0])] +=  grafetto.in_degree(node)  # TODO is this right? consider {1,2,3,4} -> {1,2,3}, {4} or {1,2,3,4} -> {1,2}, {3}, 
            hits.append(node)
    return histogram, hits

def find_maximum_number(strings):
    max_number = -1  # Initialize with a small value
    for string in strings:
        parts = string.split('-')
        if len(parts) >= 2:
            try:
                number = int(parts[1])
                max_number = max(max_number, number)
            except ValueError:
                print(max_number, parts)  # Ignore strings that don't have a valid number
    return max_number


def find_isomorphic_graphs(raw_graphs):
    isomorphic_graphs = []
    graphs = copy.deepcopy(raw_graphs)


    for i in range(len(graphs)):
        #print(i, end="")
        which_graphs_are_isomorphic_to_current = []
        try:
            which_graphs_are_isomorphic_to_current.extend( [j for j in range(len(graphs)) if nx.is_isomorphic(graphs[i], graphs[j])  ])
            #print(which_graphs_are_isomorphic_to_current, end="")
            for toremove in which_graphs_are_isomorphic_to_current:
                del graphs[toremove]

            isomorphic_graphs.append(which_graphs_are_isomorphic_to_current)
        except Exception as e:
            pass
            #print("finished working on graphs:", e)
            #break

    isomorphic_graphs = [i for i in isomorphic_graphs if len(i) > 1 ]

    return isomorphic_graphs








####### POSITION OF NODES  
    # def position_of_nodes(G):
#     #pos = nx.spring_layout(G, seed=10)

#     total_nodes = list(G.nodes())


#     #pprint.pprint(total_nodes)

#     k = 10
#     for n in total_nodes:
#         cc = nx.bfs_tree(G, n)
 
#         k = k + 1
#         for node in cc:
#             #print(node)
#             pos[node] = np.array([k, node[0]])
#             total_nodes.remove(node)

#     # handle remaining nodes
#     pass
#     #print("\n\n")
#     #pprint.pprint(pos)

#     return pos