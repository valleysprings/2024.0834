import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from tqdm import tqdm
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data

from cdlib.benchmark import LFR, PP
from networkx.generators.community import LFR_benchmark_graph, planted_partition_graph

matplotlib.use('Agg')

os.environ['OMP_NUM_THREADS'] = '48'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


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

# python implementation of LFR benchmark graph might fail to generate the graph (so does cdlib), download the graph from https://zenodo.org/records/4450167
def lfr_generate_nx(n, tau1, tau2, mu, average_degree, min_community, max_community):
    G = LFR_benchmark_graph(    
        n, tau1, tau2, mu, average_degree=average_degree, min_community=min_community, max_community=max_community, seed = random.randint(0, 1000000), tol=1e-5, max_iters=10000
    )

    communities = {frozenset([int(i) for i in G.nodes[v]["community"]]) for v in G}
    
    # make communities a dictionary for v -> community
    comms = {}
    for community in communities:
        com_idx = 0
        for v in community:
            comms[v] = com_idx
        com_idx += 1
    
    return G, comms


def lfr_generate_cdlib(n, tau1, tau2, mu, average_degree):
    G, coms = LFR(
        n, tau1, tau2, mu, average_degree=average_degree, max_community=int(0.1 * n), max_degree=int(0.1 * n), seed = random.randint(0, 1000000), tol=1e-5, max_iters=10000
    )
    coms = coms.to_node_community_map()
    coms = {k: v[0] for k, v in coms.items()}

    return G, coms


def pp_generate_cdlib(l, k, p_in, mu):
    # l: number of groups
    # k: number of nodes in each group
    # mu = (l - 1) * p_out / (p_in + (l - 1) * p_out)
    # p_out = mu * p_in / ((l - 1) * (1 - mu))
    p_out = mu * p_in / ((l - 1) * (1 - mu))
    G, coms = PP(
        l, k, p_in, p_out, seed=random.randint(0, 1000000)
    )
    assert len(G.nodes()) == l * k
    
    coms = coms.to_node_community_map()
    coms = {k: v[0] for k, v in coms.items()}

    return G, coms
    

def graph_draw(G, communities, args, idx=None):
    # communities is a dictionary for v -> community
    len_communities = len(set(list(communities.values())))
    colors = plt.cm.rainbow(np.linspace(0, 1, len_communities))

    # color the nodes
    node_colors = [0] * G.number_of_nodes()
    node_colors_dict = {}
    for k, v in communities.items():
        node_colors[k] = colors[v]
        node_colors_dict[k] = colors[v]
        
    # small radius, small edge width, community color, based on networkx
    pos_kk = nx.kamada_kawai_layout(G)
    pos_sl = nx.spring_layout(G) 
    pos_spectral = nx.spectral_layout(G)
    pos_arf = nx.arf_layout(G)
    nx.draw(G, 
            pos=pos_sl,
            node_color=node_colors,
            node_size=10, 
            width=0.1,
            with_labels=False)


    if args.graph_type == 'lfr':
        if idx is None:
            plt.savefig(f'../visualization/simulation/LFR_benchmark_graph_{args.n}_{args.tau1}_{args.tau2}_{args.mu}_{args.average_degree}.pdf', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'../visualization/simulation/LFR_benchmark_graph_{args.n}_{args.tau1}_{args.tau2}_{args.mu}_{args.average_degree}_{idx}.pdf', bbox_inches='tight', dpi=300)
    elif args.graph_type == 'pp':
        plt.savefig(f'../visualization/simulation/Planted_Partition_graph_{args.l}_{args.k}_{args.p_in}_{args.mu}.pdf', bbox_inches='tight', dpi=300)

def save_graph_pyg(graph_pyg, name, method='node2vec'):
    if not os.path.exists('../preprocess/simulation/'):
        os.makedirs('../preprocess/simulation/')
    torch.save(graph_pyg, '../preprocess/simulation/' + name + "-" + method + '_graph_pyg.pt')

    return

def nx_to_pyg(nx_graph):
    edge_index = torch.tensor(list(nx_graph.edges())).t().contiguous()
    # undirected graph
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # delete duplicates
    edge_index = torch.unique(edge_index, dim=1)
    
    edge_index = edge_index.type(torch.long)
    # print(edge_index.shape)
    num_nodes = nx_graph.number_of_nodes()
    # node_index = torch.arange(num_nodes)
    
    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    return data

def preprocess(G, communities, method, args, idx=None):
    # G to pyg
    data = nx_to_pyg(G)
    data.x = generate_embeddings_node2vec(data=data)
    if data.x.dtype != torch.float32:
        data.x = torch.tensor(data.x, dtype=torch.float)
    assert data.x.shape[0] == data.num_nodes
    data.num_features = data.x.shape[1]

    # no overlapping communities, directly assign labels

    data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    for k, v in communities.items():
        data.y[k] = v
    # print(data.y)
    
    # save
    if args.num_graphs == 1:
        if args.graph_type == 'lfr':
            if args.use_downloaded_graph:
                save_graph_pyg(data, f'{args.graph_type}-{args.n}-2-1-{args.mu}-{args.average_degree}', method=method)
            else:
                save_graph_pyg(data, f'{args.graph_type}-{args.n}-{args.tau1}-{args.tau2}-{args.mu}-{args.average_degree}', method=method)
        elif args.graph_type == 'pp':
            save_graph_pyg(data, f'{args.graph_type}-{args.l}-{args.k}-{args.p_in}-{args.mu}', method=method)
    else:
        if args.graph_type == 'lfr':
            save_graph_pyg(data, f'{args.graph_type}-{args.n}-{args.tau1}-{args.tau2}-{args.mu}-{args.average_degree}-{idx}', method=method)
        elif args.graph_type == 'pp':
            save_graph_pyg(data, f'{args.graph_type}-{args.l}-{args.k}-{args.p_in}-{args.mu}-{idx}', method=method)

def fetch_lfr_graph(args):
    file_list = sorted([f for f in os.listdir(f'../dataset/simulation/{args.average_degree}deg/{args.n}n/{int(args.mu * 100)}mu/') if f.startswith('graph_')])[:args.num_graphs]
    
    graphs = []
    for file in file_list:
        G = nx.read_edgelist(f'../dataset/simulation/{args.average_degree}deg/{args.n}n/{int(args.mu * 100)}mu/' + file, nodetype=int)
        communities = dict()
        with open(f'../dataset/simulation/{args.average_degree}deg/{args.n}n/{int(args.mu * 100)}mu/' + file.replace('graph_', 'membership_')) as f:
            for line in f:
                node, com = map(int, line.split())
                communities[node] = com
                
        graphs.append((G, communities))
        
    return graphs

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_type', type=str, default='lfr')
    argparser.add_argument('--use_downloaded_graph', action='store_true', help='use downloaded LFR graphs')
    argparser.add_argument('--num_graphs', type=int, default=1, help='number of graphs to retrieve')
    # lfr
    argparser.add_argument('--n', type=int, default=1000, help='for lfr')
    argparser.add_argument('--tau1', type=float, default=3, help='for lfr')
    argparser.add_argument('--tau2', type=float, default=1.5, help='for lfr')
    argparser.add_argument('--mu', type=float, default=0.1, help='for lfr and pp')
    argparser.add_argument('--average_degree', type=int, default=10, help='for lfr')
    # pp
    argparser.add_argument('--l', type=int, default=50, help='for pp, number of groups')
    argparser.add_argument('--k', type=int, default=20, help='for pp, number of nodes in each group')
    argparser.add_argument('--p_in', type=float, default=0.5, help='for pp, probability of intra-group edges')
    args = argparser.parse_args()
    print(args)
    
    if args.graph_type == 'lfr' and not args.use_downloaded_graph:
        # G, communities = lfr_generate_nx(args.n, args.tau1, args.tau2, args.mu, args.average_degree, args.min_community, args.max_community)
        if args.num_graphs == 1:
            G, communities = lfr_generate_cdlib(args.n, args.tau1, args.tau2, args.mu, args.average_degree)
            graph_draw(G, communities, args, idx=None)
            preprocess(G, communities, method='node2vec', args=args, idx=None)
        else:
            idx = 0
            while idx < args.num_graphs:
                print(f'Generating graph {idx+1} of {args.num_graphs}', flush=True)
                try:
                    G, communities = lfr_generate_cdlib(args.n, args.tau1, args.tau2, args.mu, args.average_degree)
                    print(f'Preprocessing graph {idx+1} of {args.num_graphs}', flush=True)
                    preprocess(G, communities, method='node2vec', args=args, idx=idx)
                    idx += 1
                except Exception as e:
                    print(f'Error: {e}', flush=True)
                    continue

            
    elif args.graph_type == 'pp':
        if args.num_graphs == 1:
            G, communities = pp_generate_cdlib(args.l, args.k, args.p_in, args.mu)
            graph_draw(G, communities, args, idx=None)
            preprocess(G, communities, method='node2vec', args=args, idx=None)
        else:
            idx = 0
            while idx < args.num_graphs:
                print(f'Generating graph {idx+1} of {args.num_graphs}', flush=True)
                try:
                    G, communities = pp_generate_cdlib(args.l, args.k, args.p_in, args.mu)
                    print(f'Preprocessing graph {idx+1} of {args.num_graphs}', flush=True)
                    preprocess(G, communities, method='node2vec', args=args, idx=idx)
                    idx += 1
                except Exception as e:
                    print(f'Error: {e}', flush=True)
                    continue
    else:
        raise ValueError(f'Graph type {args.graph_type} not supported')

