import networkx as nx
import numpy as np
import os
import re
from utils import graph_processing

def read_graphfile(datadir, dataname, quiet, max_nodes=None):
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                # index labels as written: Tox21/AIDS are 0-based, most TU sets 1-based. Subtracting 1 turned
                # Tox21's label 0 (O) into index -1, i.e. the last slot, whose position differed between the
                # training and testing files.
                node_labels+=[int(line)]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        if not quiet:
            print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip()
                attrs = [float(attr) for attr in re.split(r"[,\s]+", line)]  
                node_attrs.append(np.array(attrs))
    except IOError:
        if not quiet:
            print('No node attributes')
       
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # raw labels are kept as-is; mapping to normal (0) / anomalous (1) happens in main.prepare_data.
    # (Remapping here by first-seen order made class 0 depend on file order, which differs between
    # Tox21 training and testing files.)
    with open(filename_graphs) as f:
        for line in f:
            graph_labels.append(int(line.strip()))
    graph_labels = np.array(graph_labels)

    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        G=nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i-1]
        for u in graph_processing.node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                graph_processing.node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                graph_processing.node_dict(G)[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        mapping={}
        it=0
        for n in graph_processing.node_iter(G):
            mapping[n]=it
            it+=1
            
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs

