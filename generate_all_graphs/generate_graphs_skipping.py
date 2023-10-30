#!/usr/bin/env python
"""
Efficient motif generation.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing

import pickle

from timeit import timeit
from itertools import combinations, product, chain, combinations_with_replacement
from datetime import datetime
import random



def generate_graphs(outdegree_sequence):
    """Generates all directed graphs with a given out-degree sequence."""
    for edges in product(*[generate_edges(node, degree, len(outdegree_sequence)) \
                           for node, degree in enumerate(outdegree_sequence)]):
        yield(list(chain(*edges)))


def generate_edges(node, outdegree, total_nodes):
    """Generates all edges for a given node with a given out-degree and a given graph size."""
    for targets in combinations(set(range(total_nodes)) - {node}, outdegree):
        yield([(node, target) for target in targets])






########## OLD PARALLEL 
#### this paralelize the check if the graph is isomorphic. 

def check_in_parallel(g, tmp_indegree):
    number_of_threads = multiprocessing.cpu_count()
    p = multiprocessing.get_context("fork").Pool(int(number_of_threads/10))
    #print("processes {}".format(p))
    return_pool = p.starmap(nx.is_isomorphic, [(g, g_) for g_ in tmp_indegree])
    #print(len(return_pool))
    return return_pool


def version_para1(n):
    """Creates graphs in batches, where each batch contains graphs with
    the same out-degree sequence.  Within a batch, graphs are sorted
    by in-degree sequence, such that only graphs with the same
    in-degree sequence have to be tested for isomorphism.
    """
    graphs_so_far = list()
    for outdegree_sequence in combinations_with_replacement(range(n), n):
        tmp = dict()
        for edges in generate_graphs(outdegree_sequence):
            g = nx.from_edgelist(edges, create_using=nx.DiGraph)
            indegree_sequence = tuple(sorted(degree for _, degree in g.in_degree()))
            #print(len(tmp))
            if indegree_sequence in tmp:
                list_isomorphic = check_in_parallel(g, tmp[indegree_sequence])    
                if not any(list_isomorphic):
                #if not any(nx.is_isomorphic(g_before, g) for g_before in tmp[indegree_sequence]):
                    tmp[indegree_sequence].append(g)
            else:
                tmp[indegree_sequence] = [g]
        for graphs in tmp.values():
            graphs_so_far.extend(graphs)
    return graphs_so_far

####### NON PARALLEL 

def version_nonpara(n):
    """Creates graphs in batches, where each batch contains graphs with
    the same out-degree sequence.  Within a batch, graphs are sorted
    by in-degree sequence, such that only graphs with the same
    in-degree sequence have to be tested for isomorphism.
    """
    graphs_so_far = list()
    for outdegree_sequence in combinations_with_replacement(range(n), n):
        tmp = dict()
        for edges in generate_graphs(outdegree_sequence):
            g = nx.from_edgelist(edges, create_using=nx.DiGraph)
            indegree_sequence = tuple(sorted(degree for _, degree in g.in_degree()))
            #print(len(tmp))
            if indegree_sequence in tmp:
                #list_isomorphic = check_in_parallel(g, tmp[indegree_sequence])    
                #if not any(list_isomorphic):
                if not any(nx.is_isomorphic(g_before, g) for g_before in tmp[indegree_sequence]):
                    tmp[indegree_sequence].append(g)
            else:
                tmp[indegree_sequence] = [g]
        for graphs in tmp.values():
            graphs_so_far.extend(graphs)
    return graphs_so_far


####### PARALLEL 2 

def the_function_para2(edges, tmp_):
    #print(len(tmp_))
    g = nx.from_edgelist(edges, create_using=nx.DiGraph)
    indegree_sequence = tuple(sorted(degree for _, degree in g.in_degree()))
    if indegree_sequence in tmp_:
        if not any(nx.is_isomorphic(g_before, g) for g_before in tmp_[indegree_sequence]):
            tmp_[indegree_sequence].append(g)
    else:
        tmp_[indegree_sequence] = [g]
    return tmp_


def version_para2(n):
    """Creates graphs in batches, where each batch contains graphs with
    the same out-degree sequence.  Within a batch, graphs are sorted
    by in-degree sequence, such that only graphs with the same
    in-degree sequence have to be tested for isomorphism.
    """
    graphs_so_far = list()
    for outdegree_sequence in combinations_with_replacement(range(n), n):
        
        manager = multiprocessing.Manager()
        tmp = manager.dict()

        #number_of_threads = multiprocessing.cpu_count()
        #p = multiprocessing.get_context("fork").Pool(number_of_threads)
        outputs = []
        with multiprocessing.Pool(int(multiprocessing.cpu_count()/10)) as p:
            outputs = p.starmap(the_function_para2, [(edges, tmp.copy(),) for edges in generate_graphs(outdegree_sequence)])
            p.close()
            p.join()
        for output  in outputs:
            tmp.update(output)

        for graphs in tmp.values():
            graphs_so_far.extend(graphs)

    
    return graphs_so_far


######## PARALLEL 3 
### let's use normal dictionary, as the insert in managed dictioanry is 117 time slower, mmld

def the_function(edges, tmp_):
    #print(len(tmp_))
    g = nx.from_edgelist(edges, create_using=nx.DiGraph)
    indegree_sequence = tuple(sorted(degree for _, degree in g.in_degree()))
    if indegree_sequence in tmp_:
        if not any(nx.is_isomorphic(g_before, g) for g_before in tmp_[indegree_sequence]):
            tmp_[indegree_sequence].append(g)
    else:
        tmp_[indegree_sequence] = [g]
    return None #tmp_


def version_para3(n):
    """Creates graphs in batches, where each batch contains graphs with
    the same out-degree sequence.  Within a batch, graphs are sorted
    by in-degree sequence, such that only graphs with the same
    in-degree sequence have to be tested for isomorphism.
    """
    graphs_so_far = list()
    for outdegree_sequence in combinations_with_replacement(range(n), n):
        
        manager = multiprocessing.Manager()
        tmp = manager.dict()

        #number_of_threads = multiprocessing.cpu_count()
        #p = multiprocessing.get_context("fork").Pool(number_of_threads)
        with multiprocessing.Pool(int(multiprocessing.cpu_count()/10)) as p:
            p.starmap(the_function, [(edges, tmp,) for edges in generate_graphs(outdegree_sequence)])
            p.close()
            p.join()


        for graphs in tmp.values():
            graphs_so_far.extend(graphs)

    
    return graphs_so_far



############ PARA 4

def the_function_para4(outdegree_sequence):
    small_graphs_so_far = []
    tmp = dict()
    a = random.random()
    b = random.random()
    if a > 0.8: 
        for edges in generate_graphs(outdegree_sequence):
            if b > 0.8:
                g = nx.from_edgelist(edges, create_using=nx.DiGraph)
                indegree_sequence = tuple(sorted(degree for _, degree in g.in_degree()))
                if indegree_sequence in tmp:
                    if not any(nx.is_isomorphic(g_before, g) for g_before in tmp[indegree_sequence]):
                        tmp[indegree_sequence].append(g)
                else:
                    tmp[indegree_sequence] = [g]

    for graph in tmp.values():
        small_graphs_so_far.extend(graph)
    return small_graphs_so_far


def version_para4(n):
    """Creates graphs in batches, where each batch contains graphs with
    the same out-degree sequence.  Within a batch, graphs are sorted
    by in-degree sequence, such that only graphs with the same
    in-degree sequence have to be tested for isomorphism.
    """
    graphs_so_far = list()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            grafetti = p.starmap(the_function_para4, [(outdegree_sequence,) for outdegree_sequence in combinations_with_replacement(range(n), n)])
            p.close()
            p.join()

    for graphs in grafetti:
        graphs_so_far.extend(graphs)

    
    return graphs_so_far




if __name__ == '__main__':

    now = datetime.now()

    if sys.argv[2] == "para1":
        graphs = version_para1(int(sys.argv[1]))
    if sys.argv[2] == "para2":
        graphs = version_para2(int(sys.argv[1]))
    if sys.argv[2] == "nonpara":
        graphs = version_nonpara(int(sys.argv[1]))
    if sys.argv[2] == "para3":
        graphs = version_para3(int(sys.argv[1]))
    if sys.argv[2] == "para4":
        graphs = version_para4(int(sys.argv[1]))

    later = datetime.now()
    difference = (later - now).total_seconds()

    with open('/home/svu/cqtales/all_non_isomorphic_graphs_{}_{}_skipped.pickle'.format(sys.argv[1], sys.argv[2]), 'wb') as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done {}  with {} in {} seconds".format(sys.argv[1], sys.argv[2], difference))

    #print(len(graphs))  

