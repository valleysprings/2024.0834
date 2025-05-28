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
from torch_geometric.utils.convert import from_networkx

import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import numpy as np
from utils import *
import multiprocessing


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



def add_embeddings(data):
    data.x = generate_embeddings_node2vec(data)
    print('data.x.shape: ', data.x.shape)

    if data.x.dtype != torch.float32:
        data.x = torch.tensor(data.x, dtype=torch.float)
    data.num_features = data.x.shape[1]

    return data


def save_graph_pyg(graph_pyg, name):
    if not os.path.exists('../preprocess/' + name):
        os.makedirs('../preprocess/' + name)
    torch.save(graph_pyg, '../preprocess/' + name + '/single_node2vec_graph_pyg.pt')

    return


def preprocess(dataset):  

    graph_nx = nx.read_gml("../dataset/" + dataset + ".gml", destringizer=int)
    print('graph_nx number of nodes: ', graph_nx.number_of_nodes())
    graph_pyg = from_networkx(graph_nx)
    
    if dataset in ['dolphins', 'karate']:
        graph_pyg.y -= 1
    print('graph_pyg: ', graph_pyg.y)
    print('graph_pyg number of nodes: ', graph_pyg.num_nodes)
    print('graph_pyg number of edges: ', graph_pyg.num_edges)
    graph_pyg = add_embeddings(graph_pyg)
    save_graph_pyg(graph_pyg, dataset)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the graph data')
    parser.add_argument('--dataset', type=str, default='dolphins', choices=['dolphins', 'karate', 'eu-core', 'football'],
                        help='The dataset to preprocess')

    args = parser.parse_args()
    
    print(args)

    preprocess(args.dataset)