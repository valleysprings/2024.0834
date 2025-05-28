import warnings

warnings.filterwarnings('ignore')

import os
from os import environ

environ['OMP_NUM_THREADS'] = '48'

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from tqdm import tqdm
import pandas as pd
import graph_tool.all as gt
from graph_tool import topology
import networkx as nx
# from karateclub import GraRep, Diff2Vec, NodeSketch

import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import numpy as np
import multiprocessing

# from sklearnex import patch_sklearn
from auxiliary.utils import *

# patch_sklearn()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def generate_embeddings_naive(gt_graph, num_features=10):
    num_nodes = gt_graph.num_vertices()
    degrees = gt_graph.get_out_degrees(gt_graph.get_vertices())
    degrees = np.array(degrees, dtype=np.float32)
    # turn the degrees into a 2Dtensor
    degrees = torch.from_numpy(degrees).unsqueeze(1)
    embeddings = torch.randn(num_nodes, num_features - 1)
    embeddings = torch.cat((embeddings, degrees), dim=1)

    return embeddings

def generate_embeddings_core_number(gt_graph):
    embeddings = topology.kcore_decomposition(gt_graph)
    embeddings = np.array(embeddings.a, dtype=np.float32)
    embeddings /= np.max(embeddings)
    
    return embeddings


def generate_embeddings_node2vec(data):
    model = Node2Vec(
        data.edge_index,
        embedding_dim = 128,
        walk_length = 20,
        context_size = 10,
        walks_per_node = 10,
        num_negative_samples = 1,
        p = 1,
        q = 1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=32)

    model.train()
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
    # print(model.embedding.weight.data.cpu().numpy().shape)

    return model.embedding.weight.data.cpu()


# def generate_embeddings_grarep(nx_graph):
#     model = GraRep(
#         order=3
#     )
#     model.fit(nx_graph)

#     return model.get_embedding()


# def generate_embeddings_Diff2Vec(nx_graph):
#     model = Diff2Vec(
#         workers=multiprocessing.cpu_count() * 2,
#         diffusion_number=1
#     )
#     model.fit(nx_graph)

#     return model.get_embedding()


# def generate_embeddings_NodeSketch(nx_graph):
#     model = NodeSketch()
#     model.fit(nx_graph)

#     return model.get_embedding()


def gt_to_pyg(gt_graph):
    edge_index = torch.tensor(gt_graph.get_edges()).t().contiguous()
    edge_index = edge_index.type(torch.long)
    gt_index = torch.arange(gt_graph.num_vertices())
    data = Data(edge_index=edge_index, gt_index=gt_index)
    # noinspection PyPropertyAccess
    data.num_nodes = gt_graph.num_vertices()

    return data


def add_embeddings(data, graph_gt=None, graph_nx=None, method='node2vec'):
    if method == 'naive':
        data.x = generate_embeddings_naive(graph_gt)
    elif method == 'node2vec':
        data.x = generate_embeddings_node2vec(data)
    # elif method == 'grarep':
    #     data.x = generate_embeddings_grarep(graph_nx)
    # elif method == 'diff2vec':
    #     data.x = generate_embeddings_Diff2Vec(graph_nx)
    # elif method == 'nodesketch':
    #     data.x = generate_embeddings_NodeSketch(graph_nx)
    else:
        print('method not supported')
        return

    print('data.x.shape: ', data.x.shape)

    if data.x.dtype != torch.float32:
        data.x = torch.tensor(data.x, dtype=torch.float)
    data.num_features = data.x.shape[1]

    return data


def save_graph_pyg(graph_pyg, name, method='node2vec'):
    if not os.path.exists('../preprocess/' + name):
        os.makedirs('../preprocess/' + name)
    torch.save(graph_pyg, '../preprocess/' + name + '/' + method + '_graph_pyg.pt')

    return


def gt_to_nx(gt_graph):
    edge_list = gt_graph.get_edges()
    G = nx.Graph()
    G.add_edges_from(edge_list)

    return G


def save_graph_nx(graph_nx, name):
    if not os.path.exists('../preprocess/' + name):
        os.makedirs('../preprocess/' + name)
    nx.write_edgelist(graph_nx, '../preprocess/' + name + '/graph_nx.edgelist')

    return


def preprocess(dataset, method):
    graph_gt, map_gt = parse_ungraph_gt_fast('../dataset/' + dataset + '/com-' + dataset + '.ungraph.txt')
    graph_nx = gt_to_nx(graph_gt)
    graph_pyg = gt_to_pyg(graph_gt)
    graph_pyg = add_embeddings(graph_pyg, graph_gt=graph_gt, graph_nx=graph_nx, method=method)
    save_graph_pyg(graph_pyg, args.dataset, method=method)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the graph data')
    parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'youtube', 'dblp', 'lj'],
                        help='The dataset to preprocess')
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['naive', 'node2vec'],
                        help='The method to use for generating node embeddings')
    args = parser.parse_args()
    
    print(args)

    preprocess(args.dataset, args.method)
