import os
import random
from networkx import nodes
import pandas as pd
import graph_tool.all as gt
from graph_tool.draw import sfdp_layout, radial_tree_layout, arf_layout, fruchterman_reingold_layout
from torch_geometric.utils import subgraph, k_hop_subgraph

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20  # Adjusting the font size for better readability
palettes = plt.cm.tab20([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    print(f"line 31 in utils.py: Random seed set as {seed}")


def parse_ungraph_gt_fast(file_path, directed=False):
    # Use pandas to read the file efficiently
    df = pd.read_csv(file_path, sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'])

    # Create an empty graph using graph_tool
    G = gt.Graph(directed=directed)

    # Assuming that nodes are represented as integers
    vamp = G.add_edge_list(df.values, hashed=True)

    return G, vamp


def find_inv_map(map):
    inv_map = {}
    for i in range(len(map.a)):
        # assert inv_map.get(map.a[i]) is None
        inv_map[map.a[i]] = i
    return inv_map


def read_communities_from_file(file_path):
    communities = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of integers (node IDs)
            community = list(map(int, line.strip().split()))
            if len(community) >= 5:
                communities.append(community)
            else:
                pass
    # remove dupilcates
    communities = list(set([tuple(sorted(comm)) for comm in communities]))
    return communities

def communities_generation_pyg(graph_pyg):
    comms = graph_pyg.y
    # comms to dict
    comms_dict = {}
    for i in range(len(comms)):
        if comms[i].item() in comms_dict:
            comms_dict[comms[i].item()].append(i)
        else:
            comms_dict[comms[i].item()] = [i]
    # comms to list
    comms_list = []
    for i in comms_dict:
        if len(comms_dict[i]) >= 5:
            comms_list.append(comms_dict[i])
    # print(f'line 83 in utils.py: comms_list: {comms_list}', flush=True)
    return comms_list
    
    
def load_graph_pyg(name, method='node2vec'):
    return torch.load('../preprocess/' + name + '/' + method + '_graph_pyg.pt')

def load_graph_pyg_feat(name, method='node2vec', concat=False, original=False):
    if original:
        return torch.load('../preprocess/' + name + '/' + 'original_graph_pyg.pt')
    if concat:
        return torch.load('../preprocess/' + name + '/' + 'concat_' + method + '_graph_pyg.pt')
    else:
        return torch.load('../preprocess/' + name + '/' + 'single_' + method + '_graph_pyg.pt')

def load_graph_pyg_feat_simulation(graph_type='lfr', details=['1000', '4', '2', '0.1', '5'], num_graphs=1, counter=0):
    details = '-'.join(details)
    if num_graphs == 1:
        return torch.load('../preprocess/simulation/' + graph_type + '-' + details + '-node2vec_graph_pyg.pt')
    else:
        return torch.load('../preprocess/simulation/' + graph_type + '-' + details + '-' + str(counter) + '-node2vec_graph_pyg.pt')
    
    
def comms_index_generation(comms, num_queries=100):
    if len(comms) < num_queries:
        comms_index = np.random.choice(np.arange(len(comms)), num_queries, replace=True)
    else:
        comms_index = np.random.choice(np.arange(len(comms)), num_queries, replace=False)
    return comms_index


def trainset_generation(comms, comm_index, inv_map, pos_num=3):
    pos_nodes = comms[comm_index]
    if inv_map is not None:
        pos_nodes = [inv_map[i] for i in pos_nodes]
    else:
        pos_nodes = [i for i in pos_nodes]
    pos_num = min(pos_num, len(pos_nodes))
    pos_nodes = np.random.choice(pos_nodes, pos_num, replace=False)

    return pos_nodes


def label_generation(graph_pyg, comms, comm_index, inv_map):
    y = torch.zeros(graph_pyg.num_nodes, dtype=torch.long)
    if inv_map is None:
        for i in comms[comm_index]:
            y[i] = 1
    else:
        for j in comms[comm_index]:
            y[inv_map[j]] = 1
    graph_pyg.y = torch.tensor(y)
    # print(f'line 132 in utils.py: number of positive nodes: {sum(graph_pyg.y)}', flush=True)

    return graph_pyg


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def model_eval(model, subgraph_pyg, device, origin=False, pos_threshold=0.5, community_size_fixed=False, community_size=5):
    model.eval()
    with torch.no_grad():
        output = model(subgraph_pyg.x.to(device), subgraph_pyg.edge_index.to(device)).squeeze()
        if not origin:
            # choose community size with largest output
            if community_size_fixed:
                output_idx = output.argsort(descending=True)[:community_size]
                output = torch.zeros_like(output)
                output[output_idx] = 1
            # choose community size with output > pos_threshold
            else:
                output = output > pos_threshold
        
    return output


def calculate_stats_tradition(num_nodes, inv_map, y_true, y_pred):
    y_true_binary = np.zeros(num_nodes)
    y_pred_binary = np.zeros(num_nodes)

    if inv_map is None:
        for i in y_true:
            y_true_binary[i] = 1
    else:
        for i in y_true:
            y_true_binary[inv_map[i]] = 1

    for i in y_pred:
        y_pred_binary[i] = 1

    # y_true is community labels, with target community as 1 and others as 0
    # y_pred_binary is predicted labels, with target community as 1 and others as 0

    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    jaccard = jaccard_score(y_true_binary, y_pred_binary)
    nmi = normalized_mutual_info_score(y_true_binary, y_pred_binary)
    
    # NMI score

    return precision, recall, f1, jaccard, nmi


def calculate_stats(full_graph_pyg, subgraph, y_pred):
    y_true = full_graph_pyg.y.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_pred_index = np.where(y_pred == 1)[0]
    y_pred_pos_index = subgraph.gt_index[y_pred_index]
    y_pred = np.zeros(full_graph_pyg.num_nodes)
    # print('y_pred_pos_index: ', y_pred_pos_index)
    y_pred[y_pred_pos_index] = 1

    # y_true is community labels, with target community as 1 and others as 0
    # y_pred is predicted labels, with target community as 1 and others as 0
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # print('precision: ', precision)
    # print('recall: ', recall)
    # print('f1: ', f1)

    return precision, recall, f1, jaccard, nmi

def attributes_sims(subgraph, method):
    # Attributes similarity
    # 1. Cosine similarity
    # 2. Pearson correlation (standardized)
    
    if method == 'cosine':
        sims = cosine_similarity(subgraph.x.cpu().numpy())
    elif method == 'pearson':
        sims = np.corrcoef(subgraph.x.cpu().numpy())
    else:
        raise ValueError('Invalid method')
    
    return sims

def visualize_graph(full_graph_pyg=None, subgraph_pyg=None, y_pred=None, y_true=None, pos_nodes = None, neg_nodes = None, output_name='ground_truth.png'):
    G = gt.Graph(directed=False)
    G_aug = gt.Graph(directed=False)
    prefix = '../visualization/case_study_pics/'
    pos_nodes_suffix = '-'.join([str(i) for i in pos_nodes])
    output_name = output_name.replace('.pdf', '') + '_' + pos_nodes_suffix + '.pdf'

    # if full_graph_pyg is not None and y_true is not None and y_pred is None:
    #     # demonstration of ground truth community
    #     subgraph_pyg = full_graph_pyg.subgraph(y_true)
    #     # find one-hop neighbors
    #     # subset, edge_index, mapping, edge_mask = k_hop_subgraph(y_true, 1, full_graph_pyg.edge_index)
    #     # subgraph_pyg_one_hop_edge_index = edge_index
    #     # print('subgraph_pyg: ', subgraph_pyg.edge_index)
    #     G.add_edge_list(subgraph_pyg.edge_index.t().cpu().numpy())
    #     # G_aug.add_edge_list(subgraph_pyg_one_hop_edge_index.t().cpu().numpy())
    #     pos = radial_tree_layout(G)
    #     # pos_aug = sfdp_layout(G_aug)
    #     deg = G.degree_property_map('total')
    #     deg.a = 2 * (np.sqrt(deg.a) * 10 + 0.5)
    #     gt.graph_draw(G, pos=pos, vertex_size=deg, vertex_fill_color="#9DC3E6", edge_pen_width=10, output_size=(500, 500), output= prefix + output_name)
    #     # gt.graph_draw(G_aug, pos=pos_aug, vertex_size=deg_aug, output_size=(500, 500), output= prefix + 'one_hop_' + output_name)

    #     return
    
    if full_graph_pyg is not None and subgraph_pyg is not None and y_pred is not None and y_true is not None:
        # demonstration of predicted community
        # print('y_pred: ', y_pred)
        # print('y_true: ', y_true)
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        
        subgraph_pyg_nodes = subgraph_pyg.gt_index
        subgraph_pyg_nodes = torch.tensor(subgraph_pyg_nodes)
        
        
        # one-hop neighbors
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(y_pred, 1, full_graph_pyg.edge_index)
        y_pred_one_hop = full_graph_pyg.subgraph(subset)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(y_true, 1, full_graph_pyg.edge_index)
        y_true_one_hop = full_graph_pyg.subgraph(subset)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(subgraph_pyg_nodes, 1, full_graph_pyg.edge_index)
        subgraph_pyg_one_hop = full_graph_pyg.subgraph(subset)
        
        
        
        y_true_one_hop_nodes = y_true_one_hop.gt_index
        y_pred_one_hop_nodes = y_pred_one_hop.gt_index
        subgraph_pyg_one_hop_nodes = subgraph_pyg_one_hop.gt_index
        
        y_total = y_pred.tolist() + y_true.tolist()
        y_total = set(y_total)
        y_total = np.array(list(y_total))
        # print('y_total: ', y_total)
        y_total_tensor = torch.tensor(y_total)
        subgraph_pyg_rec = full_graph_pyg.subgraph(y_total_tensor)
        
        # subgraph_pyg_rec = full_graph_pyg
        # y_total = [i for i in range(subgraph_pyg_rec.num_nodes)]
        
        # print(subgraph_pyg_rec.edge_index.t().cpu().numpy())
        v_map = {i: subgraph_pyg_rec.gt_index[i].item() for i in range(len(y_total))}
        inv_map = {v: k for k, v in v_map.items()}
        # print('subgraph_pyg_rec.gt_index', subgraph_pyg_rec.gt_index)
        # print('inv_map', inv_map)
        # print('edge_index', subgraph_pyg_rec.edge_index.t().cpu().numpy())
        G.add_vertex(len(y_total))
        G.add_edge_list(subgraph_pyg_rec.edge_index.t().cpu().numpy())
        # print('vertices', [i for i in G.vertices()])
        # pos = sfdp_layout(G)
        pos = sfdp_layout(G, C=0.2, p=1.2, K=1.0, gamma=0.5, max_iter=1000)
        # pos = fruchterman_reingold_layout(G)
        color_left, color_right = {}, {}
        for i in y_total:
            if i in y_pred and i in y_true:
                color_right[i] = palettes[0] # True positive
            elif i in y_pred and i not in y_true:
                color_right[i] = palettes[1] # False positive
            elif i not in y_pred and i in y_true:
                color_right[i] = palettes[2] # False negative
            else:
                color_right[i] = palettes[3] # True negative
            
            if i in pos_nodes:
                color_right[i] = palettes[4]
            elif i in neg_nodes:
                color_right[i] = palettes[5]
        
        for i in y_total:
            if i in pos_nodes:
                color_left[i] = palettes[4]
            elif i in subgraph_pyg_nodes:
                color_left[i] = palettes[0]
            else:
                color_left[i] = palettes[1]
        
        assert len(color_left.keys()) == len(y_total), 'color length not equal to y_total length'
        assert len(color_right.keys()) == len(y_total), 'color length not equal to y_total length'
        
        # legend
        legend_labels_left = ['Positive query', 'Subgraph nodes', 'Other nodes']
        legend_handles_left = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[4], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[0], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[1], markersize=10)]
        
        legend_labels_right = ['True Positive', 'False Positive', 'False Negative', 'True Negative', 'Positive query', 'Negative query']
        legend_handles_right = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[0], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[1], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[2], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[3], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[4], markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palettes[5], markersize=10)]

        # print('color: ', color, flush=True)
        color_map_left = G.new_vertex_property('vector<double>')
        color_map_right = G.new_vertex_property('vector<double>')
        for i in y_total:
            color_map_left[G.vertex(inv_map[i])] = color_left[i]
            color_map_right[G.vertex(inv_map[i])] = color_right[i]
        print('output_name: ', prefix + output_name, flush=True)
        
        v_prop = G.new_vertex_property("string")
        for i in range(len(y_total)):
            v_prop[G.vertex(i)] = str(v_map[i])
        
        # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # a = gt.graph_draw(G, pos=pos, vertex_fill_color = color_map_left, vertex_text=v_prop, vertex_text_position = 1.0, edge_pen_width = 0.01, mplfig=axes[0])
        # a.fit_view()
        # axes[0].legend(legend_handles_left, legend_labels_left, loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)
        
        # a = gt.graph_draw(G, pos=pos, vertex_fill_color = color_map_right, vertex_text=v_prop, vertex_text_position = 1.0, edge_pen_width = 0.01, mplfig=axes[1])
        # a.fit_view()
        # axes[1].legend(legend_handles_right, legend_labels_right, loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)
        
        # remove coordinates
        # axes[0].set_axis_off()
        # axes[1].set_axis_off()
        # axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # plt.tight_layout()
        
        # single graph
        plt.switch_backend("cairo")
        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        a = gt.graph_draw(G, pos=pos, vertex_fill_color = color_map_right, vertex_text=v_prop, vertex_text_position = 1.0, edge_pen_width = 0.002, mplfig=fig)
        a.fit_view()
        plt.legend(legend_handles_right, legend_labels_right, fontsize=15)
        fig.savefig(prefix + 'single_' + output_name, bbox_inches='tight')
        
        return 
    
    else:
        # demonstration of subgraph
        G.add_edge_list(subgraph_pyg.edge_index.t().cpu().numpy())
        pos = sfdp_layout(G)
        gt.graph_draw(G, pos=pos, output_size=(500, 500), output= prefix + output_name)

        return

def visualize_ppr(ppr, y_true, dataset, comms_num):
    prefix = '../visualization/case_study_pics/'
    output_path = prefix + dataset + '_ppr_' +  str(comms_num) + '.pdf' 

    # Create a figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Subplot a: Visualize the ratio
    # ratio = []
    # log_ppr = np.log(ppr)
    # for i in range(len(ppr)):
    #     ratio.append((i, ppr[i]))
    # nodes_a, values_a = zip(*ratio[:100])
    # colors_a = ['red' if node == y_true else 'blue' for node in nodes_a]
    # ax1.bar(range(100), values_a, color=colors_a)
    # ax1.set_title('Top 100 Nodes by Ratio')
    # ax1.set_xlabel('Nodes')
    # ax1.set_ylabel('Ratio Value')
    # # ax1.axvline(nodes_a.index(y_true), color='green', linestyle='--', label=f'Ground Truth Node: {y_true}')
    # ax1.legend()

    # Subplot b: Visualize the ppr
    # nodes_b, values_b = zip(*ppr[:100])
    # colors_b = ['red' if node == y_true else 'blue' for node in nodes_b]
    # ax2.bar(range(100), values_b, color=colors_b)
    # ax2.set_yscale('log')
    # ax2.set_title('Top 100 Nodes by PPR')
    # ax2.set_xlabel('Nodes')
    # ax2.set_ylabel('PPR Value (log scale)')
    # # ax2.axvline(nodes_b.index(y_true), color='green', linestyle='--', label=f'Ground Truth Node: {y_true}')
    # ax2.legend()

    nodes, values = zip(*ppr[:100])
    print('nodes: ', nodes)
    print('y_true: ', y_true)
    colors = [palettes[0] if node in y_true else palettes[1] for node in nodes]
    labels = ['PPR for Ground Truth' if node in y_true else 'PPR for Others' for node in nodes]
    ax.bar(range(100), values, color=colors, label=labels)
    ax.set_yscale('log')
    ax.set_title(f'Top 100 PPR score from a community of {dataset}')
    ax.set_xlabel('Nodes')
    ax.set_ylabel('PPR Value (log scale)')

    ax2 = ax.twinx()
    ppr_ratio = np.log(values)
    ppr_ratio = - np.diff(ppr_ratio)
    ax2.plot(range(99), ppr_ratio, color=palettes[4], label='Log difference', linewidth=4)
    max_index = np.argmax(ppr_ratio) 

    ax2.plot(max_index, ppr_ratio[max_index], 'o', color=palettes[4], markersize=10, label='Highest difference') 
    ax2.set_ylabel('Negative Log difference')
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    handles = handles1 + handles2 
    labels = labels1 + labels2

    unique_handles_labels = dict(zip(labels, handles))
    ax.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper right')
    # plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    return