from turtle import pos
import graph_tool.all as gt
from networkx import subgraph

import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import metis
import subprocess
import os
# from karateclub import GraRep, Diff2Vec, NodeSketch
import multiprocessing

# For k-truss and k-core precompute
ktruss_graph = {}
kcore_graph = {}
metis_graph = {}
nx_graph = None
K_MAX = 10
num_partitions = [2, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def precompute(graph_pyg, is_kcore=False, is_ktruss=False, is_metis=False, k=-1, num_partition=None):
    # change graph_pyg to nx_graph
    global nx_graph
    nx_graph = to_networkx(graph_pyg, to_undirected=True)
    # assert nx.is_connected(nx_graph)
    # assert nx_graph.number_of_nodes() == graph_pyg.num_nodes
    # assert nx_graph.number_of_edges() == graph_pyg.num_edges
    # assert nx_graph.number_of_nodes() == max(nx_graph.nodes) + 1
    
    # remove loop
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    
    if is_ktruss:
        if k == -1: # adaptive k
            for k in range(3, K_MAX + 1):
                t = nx.k_truss(nx_graph, k=k)
                # print("t: ", t, flush=True)
                t = nx.connected_components(t)
                # print("t: ", t, flush=True)
                t = [list(j) for j in t]
                # print("t: ", t, flush=True)
                ktruss_graph[k] = t
        else:
            t = nx.k_truss(nx_graph, k=k)
            t = nx.connected_components(t)
            t = [list(j) for j in t]
            ktruss_graph[k] = t
        return ktruss_graph
            
        
    elif is_kcore:
        if k == -1: # adaptive k
            for k in range(1, K_MAX + 1):
                t = nx.k_core(nx_graph, k=k)
                # print("t: ", t, flush=True)
                t = nx.connected_components(t)
                # print("t: ", t, flush=True)
                t = [list(j) for j in t]
                # print("t: ", t, flush=True)
                kcore_graph[k] = t
        else:
            t = nx.k_core(nx_graph, k=k)
            t = nx.connected_components(t)
            t = [list(j) for j in t]
            kcore_graph[k] = t
        return kcore_graph
    
    elif is_metis:
        # print("is_metis", flush=True)
        # print(num_partition, flush=True)
        if num_partition == -1:
            num_node = nx_graph.number_of_nodes()
            # filter num_partitions to be less than num_node // 20
            global num_partitions
            num_partitions = [num for num in num_partitions if num <= num_node // 20]
            if len(num_partitions) == 0:
                num_partitions = [2]
            # print(num_partitions, flush=True)
            for k in num_partitions:
                edgecut, parts = metis.part_graph(nx_graph, k, contig=True)
                # generate subgraph
                # print("parts: ", parts, flush=True)
                subgraphs = []
                for i in range(k):
                    nodes_in_part = [node for node, part in enumerate(parts) if part == i]
                    # assert nx.is_connected(nx_graph.subgraph(nodes_in_part)) == True
                    subgraphs.append(nodes_in_part)
                # print("subgraphs: ", subgraphs, flush=True)
                # make sure the union of subgraphs is the whole graph
                # assert set.union(*[set(subgraph) for subgraph in subgraphs]) == set(range(nx_graph.number_of_nodes()))
                metis_graph[k] = subgraphs
        else:
            edgecut, parts = metis.part_graph(nx_graph, num_partition, contig=True)
            # generate subgraph
            subgraphs = []
            for i in range(num_partition):
                nodes_in_part = [node for node, part in enumerate(parts) if part == i]
                subgraphs.append(nodes_in_part)
            metis_graph[num_partition] = subgraphs
        return metis_graph
    
    raise ValueError('is_kcore or is_ktruss or is_metis or is_metis_cohesive must be True')

def revise_simrank_nx(graph_pyg, pos_nodes, is_simrank=False, is_panther=False, subgraph_max_node=1e3):
    # change graph_pyg to nx_graph
    global nx_graph
    if nx_graph is None:
        nx_graph = to_networkx(graph_pyg, to_undirected=True)
        # remove loop
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    
    if is_simrank:
        # aggregate the pos_nodes
        res = np.zeros(nx_graph.number_of_nodes())
        for pos in pos_nodes:
            sim = nx.simrank_similarity(nx_graph, source=pos)
            # print(sim)
            sim = np.array(list(sim.values()))
            res += sim
        # print(res)
        # fetch the largest indices
        sorted_indices = np.argsort(res)[::-1]
        sorted_indices = sorted_indices.copy()
        subgraph_max_node = min(subgraph_max_node, nx_graph.number_of_nodes())
        subgraph_nodes = sorted_indices[:subgraph_max_node]
        subgraph_nodes = torch.tensor(subgraph_nodes)
    
    elif is_panther:
        # aggregate the pos_nodes
        res = np.zeros(nx_graph.number_of_nodes())
        subgraph_max_node = min(subgraph_max_node, nx_graph.number_of_nodes())
        for pos in pos_nodes:
            sim = nx.panther_similarity(nx_graph, source=pos, k=subgraph_max_node)
            res[pos] += 1 # not include itself
            for key, val in sim.items():
                res[int(key)] += val
            # sim = np.array(list(sim.values()))
            # res += sim
        # print(res)
        # fetch the largest indices
        sorted_indices = np.argsort(res)[::-1]
        sorted_indices = sorted_indices.copy()
        subgraph_nodes = sorted_indices[:subgraph_max_node]
        subgraph_nodes = torch.tensor(subgraph_nodes)
    
    return subgraph_nodes
            
def revise_simrank_cpp(dataset, pos_nodes, subgraph_max_node=1e3, num_nodes=None, epsilon=1e-4):
    # write the pos_nodes to a file
    with open(f"../ablation/ExactSim/query/{dataset}.query", "w") as file:
        for node in pos_nodes:
            file.write(f"{node}\n")

    # see whether we have the precomputed result
    flag = 0
    for i in pos_nodes:
        if not os.path.exists(f"../ablation/ExactSim/results/{dataset}/{epsilon}/{i}.txt"):
            flag = 1
            break
    # print(flag)
        
    if flag == 1:
        # run the cpp program
        try:
            res = subprocess.run(
                ["../ablation/ExactSim/EXSim", 
                "-d", f"../ablation/ExactSim/dataset/{dataset}.txt", 
                "-f", f"{dataset}", 
                "-algo", "ExactSim", 
                "-e", f"{epsilon}", 
                "-qn", f"{len(pos_nodes)}"],
                capture_output=True,
                text=True,
                check=True 
            )
            # print("Standard Output:\n", res.stdout)
        except subprocess.CalledProcessError as e:
            print("An error occurred while running the command:")
            print("Return code:", e.returncode)
            print("Standard Error:\n", e.stderr)
    
    # read the result
    res_dict = {}
    for i in pos_nodes:
        with open(f"../ablation/ExactSim/results/{dataset}/{epsilon}/{i}.txt", "r") as file:
            for line in file:
                node, val = line.split()
                if node not in res_dict:
                    res_dict[int(node)] = float(val)
                else:
                    res_dict[int(node)] += float(val)
            
    
    # sort the res_dict by value
    sorted_res_dict = sorted(res_dict.keys(), key=lambda x: res_dict[x], reverse=True)
    subgraph_max_node = min(subgraph_max_node, len(sorted_res_dict), num_nodes)
    subgraph_nodes = sorted_res_dict[:subgraph_max_node]
    subgraph_nodes = torch.tensor(subgraph_nodes)
    
    return subgraph_nodes

            
def subgraph_candidate_adaptable_new(graph_gt, query, subgraph_min_node=10, subgraph_max_node=1e3, degree_normalized=False, damping=0.95, epsilon=1e-4, log_transform=True, verbose=False):
    personalization = graph_gt.new_vertex_property("double")
    for q in query:
        personalization.a[q] = 1

    personalized_pagerank_scores = gt.pagerank(graph_gt, pers=personalization, damping=damping, epsilon=epsilon)
    values = personalized_pagerank_scores.a
    # normalized by degree
    if degree_normalized:
        values = [val / (graph_gt.vertex(int(idx)).out_degree() + graph_gt.vertex(int(idx)).in_degree()) for idx, val in enumerate(values)]
        values = np.array(values)
    sorted_values_with_index = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    # if verbose:
    #     print(f'line 219 in subgraph_search.py: sorted_values_with_index: {sorted_values_with_index}', flush=True)
    #     print(f'line 220 in subgraph_search.py: values: {values}', flush=True)
        
    subgraph_nodes, ratio = [], []
    max_valid_idx = len(sorted_values_with_index) - 1
    for idx, (node, val) in enumerate(sorted_values_with_index):
        if val < epsilon:
            max_valid_idx = idx
            break
    max_node = min(subgraph_max_node, max_valid_idx)
    subgraph_min_node = min(subgraph_min_node, max_node - 1) - 1
    if verbose:
        print(f'line 230 in subgraph_search.py: max_node: {max_node}', flush=True)
    
    for i in range(max_node - 1):
        if log_transform:
            r = sorted_values_with_index[i][1] / sorted_values_with_index[i + 1][1]
        else:
            r = sorted_values_with_index[i][1] - sorted_values_with_index[i + 1][1]
        res = (i, r)
        ratio.append(res)
    ratio = sorted(ratio, key=lambda x: x[1], reverse=True)
    if verbose:
        print(f'line 241 in subgraph_search.py: ratio: {ratio}', flush=True)

    while len(subgraph_nodes) < subgraph_min_node:
        for idx, val in ratio:
            # print(idx, val, flush=True)
            if idx >= subgraph_min_node:
                subgraph_nodes = [node for node, val in sorted_values_with_index[:idx]]
                break
    subgraph_nodes = torch.tensor(subgraph_nodes)
    if verbose:
        print(f'line 251 in subgraph_search.py: subgraph_nodes: {subgraph_nodes}', flush=True)
        
    return subgraph_nodes, values, sorted_values_with_index


def subgraph_candidate_adaptable_old(graph_gt, query, threshold=0.25, subgraph_min_node=5, subgraph_max_node=1e3, degree_normalized=False):
    personalization = graph_gt.new_vertex_property("double")
    for q in query:
        personalization.a[q] = 1

    personalized_pagerank_scores = gt.pagerank(graph_gt, pers=personalization)
    values = personalized_pagerank_scores.a
    # normalized by degree
    if degree_normalized:
        values = [val / (graph_gt.vertex(int(idx)).out_degree() + graph_gt.vertex(int(idx)).in_degree()) for idx, val in enumerate(values)]
        values = np.array(values)
    sorted_values_with_index = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    subgraph_nodes = []

    while len(subgraph_nodes) <= len(query) * 2:
        temp = None
        temp_list = []
        for idx, (node, val) in enumerate(sorted_values_with_index):
            if temp is None or temp / val <= 1 + threshold:
                pass
            else:
                subgraph_nodes = temp_list.copy()

            temp_list.append(node)
            temp = val
            if idx > subgraph_max_node:
                break
        threshold /= 2
    # print('subgraph_nodes: ', len(subgraph_nodes))
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes, values, sorted_values_with_index

def subgraph_candidate_ppr_fixed(graph_gt, query, subgraph_max_node=400, degree_normalized=False, damping=0.95, epsilon=1e-4):
    personalization = graph_gt.new_vertex_property("double")
    for q in query:
        personalization.a[q] = 1

    personalized_pagerank_scores = gt.pagerank(graph_gt, pers=personalization, damping=damping, epsilon=epsilon)
    values = personalized_pagerank_scores.a
    if degree_normalized:
        values = [val / (graph_gt.vertex(int(idx)).out_degree() + graph_gt.vertex(int(idx)).in_degree()) for idx, val in enumerate(values)]
        values = np.array(values)
    sorted_values_with_index = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # subgraph_nodes = []

    # for idx, (node, val) in enumerate(sorted_values_with_index):
    #     subgraph_nodes.append(node)
    #     if idx > subgraph_max_node:
    #         break
    
    subgraph_nodes = [node for node, _ in sorted_values_with_index[:subgraph_max_node]]

    # print('subgraph_nodes: ', len(subgraph_nodes))
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes, values, sorted_values_with_index

def subgraph_candidiate_bfs_fixed(graph_gt, query, subgraph_max_node=400):
    subgraph_nodes = []
    bfs_queue = [q for q in query]
    while len(subgraph_nodes) < subgraph_max_node and len(bfs_queue) > 0:
        node = bfs_queue.pop(0)
        if node not in subgraph_nodes:
            subgraph_nodes.append(node)
            for neighbor in graph_gt.get_all_neighbors(node):
                if neighbor not in subgraph_nodes and neighbor not in bfs_queue:
                    bfs_queue.append(neighbor)
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes


def revise_subgraph(dataset, full_graph_pyg, gt_graph, pos_nodes, neg_nodes, threshold = 0.25, subgraph_min_node = 10, 
                    subgraph_max_node = 1e3, damping=0.95, epsilon=1e-4, log_transform=True, subgraph_size_fixed=False, 
                    is_bfs=False, is_simrank=False, is_panther=False, is_kcore=False, is_ktruss=False, k_selection=-1, is_metis=False, num_partition=-1, is_metis_cohesive=False, 
                    dynamic_subgraph_method=None, al_method='aggregated', degree_normalized=False, verbose=False):
    # set values
    values = None
    sorted_values_with_index = None
    pos_nodes = [int(node) for node in pos_nodes]
    
    if verbose:
        print(f'line 338 in subgraph_search.py: positive nodes for subgraph search: {pos_nodes}', flush=True)
        
    if is_bfs:
        subgraph_nodes = subgraph_candidiate_bfs_fixed(gt_graph, pos_nodes, subgraph_max_node)
    elif is_kcore:
        subgraph_nodes, k_selected = revise_kcore(full_graph_pyg, pos_nodes, k_selection)
    elif is_ktruss:
        subgraph_nodes, k_selected = revise_ktruss(full_graph_pyg, pos_nodes, k_selection)
    elif is_metis:
        subgraph_nodes, num_partition_selected = revise_metis(full_graph_pyg, pos_nodes, num_partition)
    elif is_metis_cohesive:
        subgraph_nodes, num_partition_selected = revise_metis_cohesive(full_graph_pyg, pos_nodes, num_partition)
    elif is_simrank:
        # networkx version is not used since it is too slow
        # subgraph_nodes = revise_simrank_nx(full_graph_pyg, pos_nodes, is_simrank, is_panther, subgraph_max_node)
        # cpp version is used (ExactSim from SIGMOD 2020)
        subgraph_nodes = revise_simrank_cpp(dataset, pos_nodes, subgraph_max_node, num_nodes = gt_graph.num_vertices())
    elif is_panther:
        subgraph_nodes = revise_simrank_nx(full_graph_pyg, pos_nodes, is_simrank, is_panther, subgraph_max_node)
    elif subgraph_size_fixed:
        subgraph_nodes, values, sorted_values_with_index = subgraph_candidate_ppr_fixed(gt_graph, pos_nodes, subgraph_max_node, degree_normalized, damping, epsilon)
    else:
        subgraph_nodes, values, sorted_values_with_index = subgraph_candidate_adaptable_new(gt_graph, pos_nodes, subgraph_min_node, subgraph_max_node, degree_normalized, damping, epsilon, log_transform, verbose)
        # subgraph_nodes, values, sorted_values_with_index = subgraph_candidate_adaptable(gt_graph, pos_nodes, threshold, subgraph_min_node, subgraph_max_node, degree_normalized)
    # assert pos_nodes are in subgraph_nodes
    # assert len([pos_node for pos_node in pos_nodes if pos_node not in subgraph_nodes]) == 0
    # neg_nodes may not in subgraph_nodes
    # print(neg_nodes, flush=True)
    
    # generate aggregated scores since it uses ppr scores
    if al_method == 'aggregated' and values is None:
        _, values, sorted_values_with_index = subgraph_candidate_ppr_fixed(gt_graph, pos_nodes, subgraph_max_node, degree_normalized, damping, epsilon)
    
    # print(al_method, flush=True)
    # print(type(values), flush=True)
    
    if type(neg_nodes) == int:
        len_neg_nodes = neg_nodes
        # print([full_graph_pyg.y[node] for node in subgraph_nodes], flush=True)
        subgraph_all_neg_nodes = [node for node in subgraph_nodes if full_graph_pyg.y[node] == 0]
        len_neg_nodes = min(len_neg_nodes, len(subgraph_all_neg_nodes))
        # print(len_neg_nodes, flush=True)
        neg_nodes = np.random.choice(subgraph_all_neg_nodes, len_neg_nodes, replace=False).tolist()
    else:
        neg_nodes = [neg_node for neg_node in neg_nodes if neg_node in subgraph_nodes]
        
    # change original_graph_pyg train_mask
    if verbose:
        print(f'line 386 in subgraph_search.py: number of positive nodes: {len(pos_nodes)}', f'number of negative nodes: {len(neg_nodes)}')
        print(f'line 387 in subgraph_search.py: type of positive nodes: {type(pos_nodes)}', f'type of negative nodes: {type(neg_nodes)}')
        print(f'line 388 in subgraph_search.py: pos_nodes: {pos_nodes}', f'neg_nodes: {neg_nodes}', flush=True)
    train_mask = torch.zeros(full_graph_pyg.num_nodes, dtype=torch.bool)
    train_mask[pos_nodes] = True
    train_mask[neg_nodes] = True
    full_graph_pyg.train_mask = torch.tensor(train_mask)
    # get subgraph from original_graph_pyg using subgraph_nodes
    # print(f'full graph x shape: {full_graph_pyg.x.shape}', f'full graph y shape: {full_graph_pyg.y.shape}, edge_index shape: {full_graph_pyg.edge_index.shape}', flush=True)
    subgraph_pyg = full_graph_pyg.subgraph(subgraph_nodes)
    if verbose:
        print(f'line 397 in subgraph_search.py: subgraph_pyg.train_mask: {subgraph_pyg.train_mask}', flush=True)
        print(f'line 398 in subgraph_search.py: subgraph_pyg.y: {subgraph_pyg.y}', flush=True)
    
    if dynamic_subgraph_method is not None:
        subgraph_pyg = dynamic_subgraph_rep(subgraph_pyg, dynamic_subgraph_method, sorted_values_with_index)
    
    additional_info = {}
    if is_kcore:
        additional_info['k_selected'] = k_selected
    elif is_ktruss:
        additional_info['k_selected'] = k_selected
    elif is_metis:
        additional_info['num_partition_selected'] = num_partition_selected
    elif is_metis_cohesive:
        additional_info['num_partition_selected'] = num_partition_selected

    return subgraph_pyg, pos_nodes, neg_nodes, values, sorted_values_with_index, additional_info

def revise_kcore(graph_pyg, pos_nodes, k=-1):
    if not kcore_graph:
        precompute(graph_pyg, is_kcore=True)
    
    subgraph_nodes = None
    # print(kcore_graph[4], flush=True)
    if k == -1:
        for current_k in sorted(kcore_graph.keys(), reverse=True):
            for component in kcore_graph[current_k]:
                if set(pos_nodes).issubset(component):
                    subgraph_nodes = component
                    k = current_k
                    break
            if k != -1:
                break

    else:
        for component in kcore_graph[k]:
            if set(pos_nodes).issubset(component):
                subgraph_nodes = component
                break
    
    # print("kcore: ", k, flush=True)
    if subgraph_nodes is None:
        raise ValueError(f'kcore not found for pos_nodes: {pos_nodes}')
    
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes, k
    
    
def revise_ktruss(graph_pyg, pos_nodes, k=-1):
    if not ktruss_graph:
        precompute(graph_pyg, is_ktruss=True, k=k)
    
    subgraph_nodes = None
    # print(ktruss_graph[4], flush=True)
    if k == -1:
        for current_k in sorted(ktruss_graph.keys(), reverse=True):
            for component in ktruss_graph[current_k]:
                if set(pos_nodes).issubset(component):
                    subgraph_nodes = component
                    k = current_k
                    break
            if k != -1:
                break
    else:
        for component in ktruss_graph[k]:
            if set(pos_nodes).issubset(component):
                subgraph_nodes = component
                break
    
    # print("ktruss: ", k, flush=True)
    if subgraph_nodes is None:
        raise ValueError(f'ktruss not found for pos_nodes: {pos_nodes}')
    
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes, k

def revise_metis(graph_pyg, pos_nodes, num_partition=-1):
    
    if not metis_graph:
        precompute(graph_pyg, is_metis=True, num_partition=num_partition)
        
    # print(metis_graph.keys())
    
    subgraph_nodes = None
    if num_partition == -1:
        for current_num_partition in sorted(metis_graph.keys(), reverse=True):
            # print(current_num_partition, flush=True)
            for subgraph in metis_graph[current_num_partition]:
                if set(pos_nodes).issubset(subgraph):
                    subgraph_nodes = subgraph
                    num_partition = current_num_partition
                    break
    else:
        for subgraph in metis_graph[num_partition]:
            if set(pos_nodes).issubset(subgraph):
                subgraph_nodes = subgraph
                break
    
    if subgraph_nodes is None:
        raise ValueError(f'metis not found for pos_nodes: {pos_nodes}')
    
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes, num_partition


def revise_metis_cohesive(graph_pyg, pos_nodes, num_partition=-1):
    
    if not metis_graph:
        precompute(graph_pyg, is_metis=True, num_partition=num_partition)
        
    subgraph_nodes = None
    if num_partition == -1:
        for current_num_partition in sorted(metis_graph.keys(), reverse=True):
            # print(current_num_partition, flush=True)
            multi_subgraph = set()
            for subgraph in metis_graph[current_num_partition]:
                subgraph_set = set(subgraph)
                for node in set(pos_nodes):
                    if node in subgraph_set:
                        multi_subgraph = multi_subgraph.union(subgraph_set)
                if set(pos_nodes).issubset(multi_subgraph):
                    break
            # determine the subgraph nodes are connected in original graph
            # print(multi_subgraph, flush=True)
            multi_subgraph = list(multi_subgraph)
        
            # if nx.is_connected(nx_graph.subgraph(multi_subgraph)):
            #     subgraph_nodes = multi_subgraph
            #     num_partition = current_num_partition
            #     break
            subgraph_nodes = multi_subgraph
            num_partition = current_num_partition
            break
                
    else:
        multi_subgraph = set()
        for subgraph in metis_graph[num_partition]:
            subgraph_set = set(subgraph)
            for node in set(pos_nodes):
                if node in subgraph_set:
                    multi_subgraph = multi_subgraph.union(subgraph_set)
            if set(pos_nodes).issubset(multi_subgraph):
                break
        # determine the subgraph nodes are connected in original graph
        multi_subgraph = list(multi_subgraph)
        # if nx.is_connected(nx_graph.subgraph(multi_subgraph)):
        #     subgraph_nodes = multi_subgraph
        subgraph_nodes = multi_subgraph
               
    if subgraph_nodes is None:
        raise ValueError(f'metis not found for pos_nodes: {pos_nodes}, \
                         included: {set(pos_nodes).issubset(set(multi_subgraph))}, \
                         connected: {nx.is_connected(nx_graph.subgraph(multi_subgraph))}')
    
    subgraph_nodes = torch.tensor(subgraph_nodes)
    return subgraph_nodes, num_partition

    
def revise_graph(full_graph_pyg, gt_graph, pos_nodes, neg_nodes):
    
    if type(neg_nodes) == int:
        len_neg_nodes = neg_nodes
        all_neg_nodes = [node for node in range(full_graph_pyg.num_nodes) if full_graph_pyg.y[node] == 0]
        len_neg_nodes = min(len_neg_nodes, len(all_neg_nodes))
        neg_nodes = np.random.choice(all_neg_nodes, len_neg_nodes, replace=False)
    # change original_graph_pyg train_mask
    train_mask = torch.zeros(full_graph_pyg.num_nodes, dtype=torch.bool)
    train_mask[pos_nodes] = True
    train_mask[neg_nodes] = True
    full_graph_pyg.train_mask = torch.tensor(train_mask)

    _, values, _ = subgraph_candidate_ppr_fixed(gt_graph, pos_nodes, 1000)

    return full_graph_pyg, pos_nodes, neg_nodes, values


def dynamic_subgraph_rep(subgraph_pyg, method_selection, sorted_values_with_index):
    """
    Generate dynamic subgraph representations in subgraph search stage
    """

    subgraph_nx_graph = to_networkx(subgraph_pyg, to_undirected=True)
    subgraph_nx_graph.remove_edges_from(nx.selfloop_edges(subgraph_nx_graph))
    subgraph_size = subgraph_nx_graph.number_of_nodes()
        
    if method_selection == 'coreness':
        # compute coreness of the subgraph, coclep default
        subgraph_pyg.x = generate_embeddings_core_number(subgraph_nx_graph)
    elif method_selection == 'ppr':
        ppr_scores = [sorted_values_with_index[i][1] for i in range(subgraph_size)]
        ppr_scores = torch.tensor(ppr_scores, dtype=torch.float)
        ppr_scores = ppr_scores / ppr_scores.max()
        # print(ppr_scores, flush=True)
        subgraph_pyg.x = ppr_scores.unsqueeze(1)
    elif method_selection == 'gaussian':
        # ics-gnn default
        subgraph_pyg.x = torch.randn(subgraph_size, 100)
    # elif method_selection == 'nodesketch':
    #     subgraph_pyg.x = generate_embeddings_nodesketch(subgraph_nx_graph)
    # elif method_selection == 'diff2vec':
    #     subgraph_pyg.x = generate_embeddings_diff2vec(subgraph_nx_graph)
    # elif method_selection == 'grarep':
    #     subgraph_pyg.x = generate_embeddings_grarep(subgraph_nx_graph)
    else:
        raise ValueError('Invalid method selection')
    
    if subgraph_pyg.x.dtype != torch.float32:
        subgraph_pyg.x = torch.tensor(subgraph_pyg.x, dtype=torch.float)
    subgraph_pyg.num_features = subgraph_pyg.x.shape[1]
    # print(subgraph_pyg.x.shape, flush=True)

    return subgraph_pyg

# simple subgraph embedding methods
def generate_embeddings_core_number(subgraph_nx_graph):
    core_numbers = nx.core_number(subgraph_nx_graph)
    embeddings = []
    for node in range(subgraph_nx_graph.number_of_nodes()):
        embeddings.append(core_numbers[node])
        
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    if len(embeddings) > 0:
        embeddings = embeddings / embeddings.max()
    
    embeddings = embeddings.unsqueeze(1)
    
    return embeddings


# def generate_embeddings_grarep(subgraph_nx_graph):
#     if subgraph_nx_graph.number_of_nodes() <= 32:
#         dim = subgraph_nx_graph.number_of_nodes() // 2
#     else:
#         dim = 32
        
#     model = GraRep(
#         dimensions=dim,
#         order=3
#     )
#     model.fit(subgraph_nx_graph)
#     embeddings = model.get_embedding()
#     embeddings = torch.tensor(embeddings, dtype=torch.float)
#     # print(embeddings.shape, flush=True)

#     return embeddings


# def generate_embeddings_diff2vec(subgraph_nx_graph):
#     model = Diff2Vec(
#         workers=multiprocessing.cpu_count() * 2,
#         diffusion_number=1
#     )
#     model.fit(subgraph_nx_graph)

#     return model.get_embedding()


# def generate_embeddings_nodesketch(subgraph_nx_graph):
#     model = NodeSketch()
#     model.fit(subgraph_nx_graph)

#     return model.get_embedding()

      