import os
import sys
import time
import warnings
from dataclasses import dataclass, field
import os.path as osp
from typing import List, Optional, Literal, Dict

import torch
from tqdm import tqdm
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# Local imports
from auxiliary.utils import *
from auxiliary.preprocess_nofeat import *
from ics.models import *
from ics.active_learning import *
from ics.subgraph_search import *
from ics.config import parse_args

# Environment settings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '48'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
counter = 0 # for simulation

@dataclass
class PPR_ICS:
    # Data parameters
    dataset: str = 'amazon'
    embedding: Literal['node2vec', 'grarep', 'diff2vec', 'nodesketch'] = 'node2vec'
    original: bool = False
    simulation: bool = False
    simulation_graph_type: Literal['lfr', 'pp'] = None
    simulation_details: List[str] = field(default_factory=lambda: [])
    num_graphs: int = 1
    directed_using_undirected_embedding: bool = False
    directed_using_undirected_PPR: bool = False
    directed_using_undirected_GNN: bool = False
    
    # Label parameters
    pos_num: int = 3  # Initial positive labels per community
    neg_num: int = 3  # Initial negative labels per community
    num_queries: int = 100  # Number of CS queries
    
    # Subgraph parameters
    is_subgraph: bool = True  # True for subgraph search, False for graph search
    subgraph_size_fixed: bool = False
    subgraph_max_node: float = 1e3
    subgraph_min_node: int = 5
    is_bfs: bool = False
    is_simrank: bool = False
    is_panther: bool = False
    is_kcore: bool = False
    is_ktruss: bool = False
    is_metis: bool = False
    is_metis_cohesive: bool = False
    num_partition: int = -1  # -1 means adaptive number of partitions
    k_selection: int = -1 # -1 means find the largest k for k-core and k-truss
    
    # PPR parameters
    ppr_threshold: float = 0.25 # obsolete
    degree_normalized: bool = False # PPR is degree normalized
    damping: float = 0.95 # alpha in PPR
    epsilon: float = 1e-4 # epsilon in PPR
    log_transform: bool = True
    
    # Active learning parameters
    al_method: Literal['random', 'uncertainty', 'entropy', 'ppr', 'subgraph_degree', 
                      'global_degree', 'aggregated'] = 'uncertainty'
    deterministic: bool = False
    incremental_num: int = 3  # Nodes to add per round
    al_round: int = 1
    temperature_1: float = 1.0
    temperature_2: float = 1.0
    percentile_normalize: bool = False
    time_decay: bool = True
    simple_aggregation: bool = False
    beta_sampled_aggregation: bool = False
    
    # GNN parameters
    model: Literal['GCN', 'GraphSAGE', 'GAT'] = 'GCN'
    epochs: int = 100
    learning_rate: float = 0.01
    hidden_layers: List[int] = field(default_factory=lambda: [128])
    dropout: float = 0.5
    gnn_threshold: float = 0.5
    gnn_learned_embeddings: bool = False
    gnn_embed_size: int = 128
    dynamic_subgraph_method: Literal['coreness', 'ppr', 'gaussian', 'nodesketch'] = None
    use_ELPH_modified: bool = False
    reset_parameters: bool = not gnn_learned_embeddings # reset parameters for GNN (not for GNN-learned-embeddings)
    
    # Community parameters
    community_size_fixed: bool = False
    community_size: int = 5
    sim1_threshold: float = 0.9
    sim2_threshold: float = 1.0
    loss_ma_threshold: float = 1e-2
    
    # Other parameters
    seed: int = 42
    verbose: bool = False
    draw: bool = False
    
    # Runtime attributes
    graph_gt: gt.Graph = field(init=False)
    graph_pyg: Data = field(init=False)
    inv_map: Optional[Dict] = field(init=False)
    comms: List = field(init=False)

    def __post_init__(self):
        self.graph_gt, self.graph_pyg, self.inv_map, self.comms = self.load_data(self.dataset)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PPR_ICS':
        params = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                params[field_name] = getattr(args, field_name)
        
        if hasattr(args, 'not_subgraph'):
            params['is_subgraph'] = not args.not_subgraph
        
        if hasattr(args, 'not_log_transform'):
            params['log_transform'] = not args.not_log_transform
        
        if hasattr(args, 'not_reset_parameters'):
            params['reset_parameters'] = not args.not_reset_parameters
        
        if hasattr(args, 'not_time_decay'):
            params['time_decay'] = not args.not_time_decay
        
        return cls(**params)

    def load_data(self, dataset):
        if dataset in ['dolphins', 'karate', 'eu-core', 'football', 'amazon_cn'] and not self.simulation:
            graph_pyg = load_graph_pyg_feat(dataset, method=self.embedding, concat=False, original=self.original)
            graph_pyg.gt_index = torch.arange(graph_pyg.num_nodes)
            graph_gt = gt.Graph(directed=False)
            graph_gt.add_edge_list(graph_pyg.edge_index.t())
            seed_torch(self.seed)
            if dataset in ['amazon_cn']:
                comms = graph_pyg.comms
            else:
                comms = communities_generation_pyg(graph_pyg)
            return graph_gt, graph_pyg, None, comms
        
        elif dataset in ['eu-core_directed', 'eu-core_undirected']:
            graph_pyg = load_graph_pyg(dataset, method='node2vec')
            graph_pyg.gt_index = torch.arange(graph_pyg.num_nodes)
            
            if dataset == 'eu-core_undirected':
                graph_gt = gt.Graph(directed=False)
                graph_gt.add_edge_list(graph_pyg.edge_index.t())
            else:
                graph_gt = gt.Graph(directed=True)
                
                if self.directed_using_undirected_embedding:
                    graph_pyg_emb = load_graph_pyg('eu-core_undirected', method='node2vec')
                    graph_pyg.x = graph_pyg_emb.x
                    if self.verbose:
                        print(f'line 170 in main.py: change embedding to undirected', flush=True)
                if self.directed_using_undirected_GNN:
                    graph_pyg_gnn = load_graph_pyg('eu-core_undirected', method='node2vec')
                    graph_pyg.edge_index = graph_pyg_gnn.edge_index
                    if self.verbose:
                        print(f'line 175 in main.py: change GNN to undirected', flush=True)
                if self.directed_using_undirected_PPR:
                    graph_pyg_ppr = load_graph_pyg('eu-core_undirected', method='node2vec')
                    graph_gt.add_edge_list(graph_pyg_ppr.edge_index.t())
                    if self.verbose:
                        print(f'line 180 in main.py: change PPR to undirected', flush=True)
                else:
                    graph_pyg_directed = load_graph_pyg('eu-core_directed', method='node2vec')
                    graph_gt.add_edge_list(graph_pyg_directed.edge_index.t())  
            comms = communities_generation_pyg(graph_pyg)
            if self.verbose:
                seed_torch(self.seed)
            
            return graph_gt, graph_pyg, None, comms
        elif self.simulation_graph_type in ['lfr', 'pp'] and self.simulation:
            graph_pyg = load_graph_pyg_feat_simulation(graph_type=self.simulation_graph_type, details=self.simulation_details, num_graphs=self.num_graphs, counter=counter)
            graph_pyg.gt_index = torch.arange(graph_pyg.num_nodes)
            graph_gt = gt.Graph(directed=False)
            graph_gt.add_edge_list(graph_pyg.edge_index.t())
            comms = communities_generation_pyg(graph_pyg)
            return graph_gt, graph_pyg, None, comms
        else:
            seed_torch(self.seed)
            graph_gt, map_gt = parse_ungraph_gt_fast(
                osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset', dataset,
                         'com-' + dataset + '.ungraph.txt'))
            inv_map = find_inv_map(map_gt)
            comms = read_communities_from_file(
                osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset', dataset,
                         'com-' + dataset + '.top5000.cmty.txt'))
            graph_pyg = load_graph_pyg(dataset, method=self.embedding)
            return graph_gt, graph_pyg, inv_map, comms

    def binary_cls_revise_active_learning(self):
        f1_list, recall_list, precision_list, jaccard_list, nmi_list = {}, {}, {}, {}, {}
        f1_sub_list, recall_sub_list, precision_sub_list, jaccard_sub_list, nmi_subgraph_list = {}, {}, {}, {}, {}
        t_sub_list, t_gnn_list, t_al_list, t_eval_list = {}, {}, {}, {}
        size_sub, size_comm = {}, {}
        subgraph_ablation_para_track_list = {}
        comms_index = comms_index_generation(self.comms, num_queries=self.num_queries)

        for i, comm_index in tqdm(enumerate(comms_index), total=len(comms_index), desc="Processing communities", ncols=100, disable=True):

            f1_list[comm_index], recall_list[comm_index], precision_list[comm_index], jaccard_list[comm_index], nmi_list[comm_index] = [], [], [], [], []
            f1_sub_list[comm_index], recall_sub_list[comm_index], precision_sub_list[comm_index], jaccard_sub_list[comm_index], nmi_subgraph_list[comm_index] = [], [], [], [], []
            t_sub_list[comm_index], t_gnn_list[comm_index], t_al_list[comm_index], t_eval_list[comm_index] = [], [], [], []
            size_sub[comm_index], size_comm[comm_index] = [], []
            subgraph_ablation_para_track_list[comm_index] = []

            pos_nodes = trainset_generation(self.comms, comm_index, self.inv_map, pos_num=self.pos_num)
            self.graph_pyg = label_generation(self.graph_pyg, self.comms, comm_index, self.inv_map)
            

            t_sub_start = time.time()
            if self.is_subgraph:
                subgraph_pyg, pos_nodes, neg_nodes, values, sorted_values_with_index, additional_info = revise_subgraph(
                    dataset=self.dataset, full_graph_pyg=self.graph_pyg, gt_graph=self.graph_gt, pos_nodes=pos_nodes, neg_nodes=self.neg_num, threshold=self.ppr_threshold, 
                    subgraph_min_node=self.subgraph_min_node, subgraph_max_node=self.subgraph_max_node, damping=self.damping, epsilon=self.epsilon,
                    log_transform=self.log_transform, subgraph_size_fixed=self.subgraph_size_fixed, is_bfs=self.is_bfs, is_simrank=self.is_simrank, is_panther=self.is_panther, is_kcore=self.is_kcore, is_ktruss=self.is_ktruss, k_selection=self.k_selection, 
                    is_metis=self.is_metis, num_partition=self.num_partition, is_metis_cohesive=self.is_metis_cohesive, dynamic_subgraph_method=self.dynamic_subgraph_method,
                    al_method=self.al_method, degree_normalized=self.degree_normalized, verbose=self.verbose)
            else:
                subgraph_pyg, pos_nodes, neg_nodes, values = revise_graph(full_graph_pyg=self.graph_pyg, original_graph_gt=self.graph_gt, pos_nodes=pos_nodes, neg_num=self.neg_num)

            t_sub_end = time.time()
            size_sub[comm_index].append(subgraph_pyg.num_nodes)
            
            if self.is_kcore or self.is_ktruss:
                subgraph_ablation_para_track_list[comm_index].append(additional_info['k_selected'])
            elif self.is_metis or self.is_metis_cohesive:
                subgraph_ablation_para_track_list[comm_index].append(additional_info['num_partition_selected'])
                
            if self.verbose:
                print(f'line 248 in main.py: Initial subgraph for community {comm_index} with {subgraph_pyg.num_nodes} nodes and {subgraph_pyg.num_edges} edges', flush=True)

            t_sub_list[comm_index].append(t_sub_end - t_sub_start)
            
            # instantiate model
            if self.reset_parameters and not self.gnn_learned_embeddings and not self.use_ELPH_modified:
                model_gnn = GNN(in_channels=subgraph_pyg.num_features, 
                                hidden_layers=self.hidden_layers, 
                                out_channels=1, 
                                layer_type=self.model, 
                                dropout=self.dropout)

                optimizer_adam = torch.optim.Adam(model_gnn.parameters(), lr=self.learning_rate, weight_decay=5e-4)
                model_gnn = model_gnn.to(device)
                model_gnn.train()

                
            for i in range(self.al_round + 1):
                # instantiate model here
                if self.gnn_learned_embeddings:
                    model_gnn = GNN_learned_embeddings(subgraph_size=subgraph_pyg.num_nodes, 
                                                    in_channels=self.gnn_embed_size, 
                                                    hidden_layers=self.hidden_layers, 
                                                    out_channels=1, 
                                                    layer_type=self.model, 
                                                    dropout=self.dropout,
                                                    device=device)
                
                elif self.use_ELPH_modified:
                    model_gnn = ELPH_modified(subgraph_size=subgraph_pyg.num_nodes, 
                                            in_channels=self.gnn_embed_size, 
                                            hidden_layers=self.hidden_layers, 
                                            out_channels=1, 
                                            dropout=self.dropout,
                                            device=device)
                    
                elif not self.reset_parameters:
                    model_gnn = GNN(in_channels=subgraph_pyg.num_features, 
                                    hidden_layers=self.hidden_layers, 
                                    out_channels=1, 
                                    layer_type=self.model, 
                                    dropout=self.dropout)

                if self.reset_parameters or not self.gnn_learned_embeddings or self.use_ELPH_modified:
                    optimizer_adam = torch.optim.Adam(model_gnn.parameters(), lr=self.learning_rate, weight_decay=5e-4)
                    model_gnn = model_gnn.to(device)
                    model_gnn.train()

                t_gnn_start = time.time()

                loss_seq = []
                for epoch in range(self.epochs):
                    optimizer_adam.zero_grad()
                    output = model_gnn(subgraph_pyg.x.to(device), subgraph_pyg.edge_index.to(device)).squeeze()
                    loss = F.binary_cross_entropy(output[subgraph_pyg.train_mask],
                                                  subgraph_pyg.y[subgraph_pyg.train_mask].float().to(device))
                    loss_seq.append(loss.item())
                    loss.backward()
                    optimizer_adam.step()

                    # early stopping
                    if epoch % 20 == 0 and epoch > 0:
                        loss_gap = np.array(loss_seq[-20:]) - loss.item()
                        if self.verbose:
                            print(f'line 312 in main.py: loss_gap: {loss_gap.mean()}')
                        if loss_gap.mean() < self.loss_ma_threshold:
                            break

                t_gnn_end = time.time()
                t_gnn_list[comm_index].append(t_gnn_end - t_gnn_start)

                t_eval_start = time.time()
                y_pred = model_eval(model_gnn, subgraph_pyg, device, origin=False, 
                                    pos_threshold=self.gnn_threshold, 
                                    community_size_fixed=self.community_size_fixed, 
                                    community_size=self.community_size)
                
                t_eval_end = time.time()
                t_eval_list[comm_index].append(t_eval_end - t_eval_start)
                size_comm[comm_index].append(y_pred.sum().item())

                precision, recall, f1, jaccard, nmi = calculate_stats(self.graph_pyg, subgraph_pyg, y_pred)
                y_pred_subgraph = torch.ones(subgraph_pyg.num_nodes, dtype=torch.long)
                precision_subgraph, recall_subgraph, f1_subgraph, jaccard_subgraph, nmi_subgraph = calculate_stats(self.graph_pyg, subgraph_pyg, y_pred_subgraph)
                
                if self.draw:
                    gt = self.comms[comm_index]
                    if self.inv_map is not None:
                        gt = [self.inv_map[i] for i in gt]

                    y_pred = y_pred.cpu().numpy()
                    y_res = subgraph_pyg.gt_index[y_pred == 1]
                    visualize_graph(full_graph_pyg=self.graph_pyg, subgraph_pyg=subgraph_pyg, y_true=gt, y_pred=y_res,
                                    pos_nodes=pos_nodes, neg_nodes=neg_nodes,
                                    output_name=f'{self.dataset}/gt_{comm_index}_{i}.pdf')

                f1_list[comm_index].append(f1)
                recall_list[comm_index].append(recall)
                precision_list[comm_index].append(precision)
                jaccard_list[comm_index].append(jaccard)
                nmi_list[comm_index].append(nmi)
                
                f1_sub_list[comm_index].append(f1_subgraph)
                recall_sub_list[comm_index].append(recall_subgraph)
                precision_sub_list[comm_index].append(precision_subgraph)
                jaccard_sub_list[comm_index].append(jaccard_subgraph)
                nmi_subgraph_list[comm_index].append(nmi_subgraph)
                
                    
                if y_pred.sum() > self.sim1_threshold * len(y_pred) or len(filter_indices(subgraph_pyg)) <= (1 - self.sim2_threshold) * len(y_pred):
                    if self.verbose:
                        print(f'line 359 in main.py: Early stop for community {comm_index} at round {i} with {subgraph_pyg.num_nodes} nodes and {subgraph_pyg.num_edges} edges', flush=True)
                    break

                
                if i == self.al_round:
                    if self.verbose:
                        print(f'line 365 in main.py: Total time for community {comm_index}: {sum(t_sub_list[comm_index]) + sum(t_gnn_list[comm_index]) + sum(t_al_list[comm_index]) + sum(t_eval_list[comm_index]):.4f}')
                    break

                t_al_start = time.time()
                pos, neg = active_learning(model_gnn, subgraph_pyg, device=device, method=self.al_method, deterministic=self.deterministic, incremental_num=self.incremental_num,
                                           original_pyg=self.graph_pyg, ppr_vec=values, temperature_1=self.temperature_1, temperature_2=self.temperature_2, current_active_learning_round=i, 
                                           max_active_learning_round=self.al_round, percentile_normalize=self.percentile_normalize, 
                                           time_decay=self.time_decay, simple_aggregation=self.simple_aggregation, beta_sampled_aggregation=self.beta_sampled_aggregation, verbose=self.verbose)
                t_al_end = time.time()
                t_al_list[comm_index].append(t_al_end - t_al_start)

                if len(pos) + len(neg) > 0:
                    # if instantiate model in each round, reset_parameters is not needed
                    if self.reset_parameters and not self.gnn_learned_embeddings and not self.use_ELPH_modified:
                        reset_parameters(model_gnn)
                    
                    pos_nodes = np.concatenate((pos_nodes, pos))
                    neg_nodes = np.concatenate((neg_nodes, neg))
                    
                    if self.verbose:
                        print(f'line 385 in main.py: For next round, number of positive nodes: {len(pos_nodes)}', f'number of negative nodes: {len(neg_nodes)}')
                        print(f'line 386 in main.py: For next round, positive nodes: {pos_nodes}', f'negative nodes: {neg_nodes}')
                    
                    t_sub_start = time.time()
                    if self.is_subgraph:
                        subgraph_pyg, pos_nodes, neg_nodes, values, sorted_values_with_index, additional_info = revise_subgraph(
                            dataset=self.dataset, full_graph_pyg=self.graph_pyg, gt_graph=self.graph_gt, pos_nodes=pos_nodes, neg_nodes=neg_nodes, threshold=self.ppr_threshold, 
                            subgraph_min_node=self.subgraph_min_node, subgraph_max_node=self.subgraph_max_node, damping=self.damping, epsilon=self.epsilon,
                            log_transform=self.log_transform, subgraph_size_fixed=self.subgraph_size_fixed, is_bfs=self.is_bfs, is_simrank=self.is_simrank, is_panther=self.is_panther, is_kcore=self.is_kcore, is_ktruss=self.is_ktruss, k_selection=self.k_selection, 
                            is_metis=self.is_metis, num_partition=self.num_partition, is_metis_cohesive=self.is_metis_cohesive, dynamic_subgraph_method=self.dynamic_subgraph_method,
                            al_method=self.al_method, degree_normalized=self.degree_normalized, verbose=self.verbose)
                        size_sub[comm_index].append(subgraph_pyg.num_nodes)

                    else:
                        subgraph_pyg, pos_nodes, neg_nodes, values = revise_graph(full_graph_pyg=self.graph_pyg, original_graph_gt=self.graph_gt, pos_nodes=pos_nodes, neg_nodes=neg_nodes)
                    t_sub_end = time.time()
                    t_sub_list[comm_index].append(t_sub_end - t_sub_start)
                    
                    if self.is_kcore or self.is_ktruss:
                        subgraph_ablation_para_track_list[comm_index].append(additional_info['k_selected'])
                    elif self.is_metis or self.is_metis_cohesive:
                        subgraph_ablation_para_track_list[comm_index].append(additional_info['num_partition_selected'])
                else:
                    break

        if self.simulation and self.num_graphs > 1:
            global counter
            counter += 1

        return comms_index, f1_list, recall_list, precision_list, jaccard_list, nmi_list, \
            t_sub_list, t_gnn_list, t_al_list, t_eval_list, \
            f1_sub_list, recall_sub_list, precision_sub_list, jaccard_sub_list, nmi_subgraph_list, \
            size_sub, size_comm, subgraph_ablation_para_track_list


if __name__ == '__main__':
    args = parse_args()
    
    print('-' * 200)
    print('-' * 200)

    if args.num_graphs > 1:
        # interactive
        f1_list_sim, recall_list_sim, precision_list_sim, jaccard_list_sim, nmi_list_sim = [], [], [], [], []
        f1_sub_list_sim, recall_sub_list_sim, precision_sub_list_sim, jaccard_sub_list_sim, nmi_subgraph_list_sim = [], [], [], [], []
        size_sub_sim, size_comm_sim = [], []
    
        # default
        f1_list_sim_default, recall_list_sim_default, precision_list_sim_default, jaccard_list_sim_default, nmi_list_sim_default = [], [], [], [], []
        f1_sub_list_sim_default, recall_sub_list_sim_default, precision_sub_list_sim_default, jaccard_sub_list_sim_default, nmi_subgraph_list_sim_default = [], [], [], [], []
        size_sub_sim_default, size_comm_sim_default = [], []
        
        for i in range(args.num_graphs):
            comms_index, f1_list, recall_list, precision_list, jaccard_list, nmi_list, \
                t_sub_list, t_gnn_list, t_al_list, t_eval_list, \
                f1_sub_list, recall_sub_list, precision_sub_list, jaccard_sub_list, nmi_subgraph_list, \
                size_sub, size_comm, subgraph_ablation_para_track_list = \
                PPR_ICS.from_args(args).binary_cls_revise_active_learning()

            # padding the results for each community
            for comm_index in comms_index:
                if len(f1_list[comm_index]) < args.al_round + 1:
                    f1_list[comm_index] += [f1_list[comm_index][-1]] * (args.al_round + 1 - len(f1_list[comm_index]))
                    recall_list[comm_index] += [recall_list[comm_index][-1]] * (args.al_round + 1 - len(recall_list[comm_index]))
                    precision_list[comm_index] += [precision_list[comm_index][-1]] * (args.al_round + 1 - len(precision_list[comm_index]))
                    jaccard_list[comm_index] += [jaccard_list[comm_index][-1]] * (args.al_round + 1 - len(jaccard_list[comm_index]))
                    nmi_list[comm_index] += [nmi_list[comm_index][-1]] * (args.al_round + 1 - len(nmi_list[comm_index]))
                    
                    f1_sub_list[comm_index] += [f1_sub_list[comm_index][-1]] * (args.al_round + 1 - len(f1_sub_list[comm_index]))
                    recall_sub_list[comm_index] += [recall_sub_list[comm_index][-1]] * (args.al_round + 1 - len(recall_sub_list[comm_index]))
                    precision_sub_list[comm_index] += [precision_sub_list[comm_index][-1]] * (args.al_round + 1 - len(precision_sub_list[comm_index]))
                    jaccard_sub_list[comm_index] += [jaccard_sub_list[comm_index][-1]] * (args.al_round + 1 - len(jaccard_sub_list[comm_index]))
                    nmi_subgraph_list[comm_index] += [nmi_subgraph_list[comm_index][-1]] * (args.al_round + 1 - len(nmi_subgraph_list[comm_index]))
                    
                    size_sub[comm_index] += [size_sub[comm_index][-1]] * (args.al_round + 1 - len(size_sub[comm_index]))
                    size_comm[comm_index] += [size_comm[comm_index][-1]] * (args.al_round + 1 - len(size_comm[comm_index]))
                    if args.is_kcore or args.is_ktruss or args.is_metis or args.is_metis_cohesive:
                        subgraph_ablation_para_track_list[comm_index] += [subgraph_ablation_para_track_list[comm_index][-1]] * (args.al_round + 1 - len(subgraph_ablation_para_track_list[comm_index]))
                # assert len(f1_list[comm_index]) == args.al_round + 1, print(len(f1_list[comm_index]))
                # assert len(recall_list[comm_index]) == args.al_round + 1, print(len(recall_list[comm_index]))
                # assert len(precision_list[comm_index]) == args.al_round + 1, print(len(precision_list[comm_index]))
            
            f1_np = np.array([f1 for comm_index, f1 in f1_list.items()])
            precision_np = np.array([precision for comm_index, precision in precision_list.items()])
            recall_np = np.array([recall for comm_index, recall in recall_list.items()])
            jaccard_np = np.array([jaccard for comm_index, jaccard in jaccard_list.items()])
            nmi_np = np.array([nmi for comm_index, nmi in nmi_list.items()])
            
            f1_sub_np = np.array([f1 for comm_index, f1 in f1_sub_list.items()])
            precision_sub_np = np.array([precision for comm_index, precision in precision_sub_list.items()])
            recall_sub_np = np.array([recall for comm_index, recall in recall_sub_list.items()])
            jaccard_sub_np = np.array([jaccard for comm_index, jaccard in jaccard_sub_list.items()])
            nmi_subgraph_np = np.array([nmi for comm_index, nmi in nmi_subgraph_list.items()])
            
            size_sub_np = np.array([size for comm_index, size in size_sub.items()])
            size_comm_np = np.array([size for comm_index, size in size_comm.items()])
            
            f1_list_sim.append(f1_np[:, args.al_round].mean() * 100)
            recall_list_sim.append(recall_np[:, args.al_round].mean() * 100)
            precision_list_sim.append(precision_np[:, args.al_round].mean() * 100)
            jaccard_list_sim.append(jaccard_np[:, args.al_round].mean() * 100)
            nmi_list_sim.append(nmi_np[:, args.al_round].mean() * 100)
            
            f1_sub_list_sim.append(f1_sub_np[:, args.al_round].mean() * 100)
            recall_sub_list_sim.append(recall_sub_np[:, args.al_round].mean() * 100)
            precision_sub_list_sim.append(precision_sub_np[:, args.al_round].mean() * 100)
            jaccard_sub_list_sim.append(jaccard_sub_np[:, args.al_round].mean() * 100)
            nmi_subgraph_list_sim.append(nmi_subgraph_np[:, args.al_round].mean() * 100)
            
            size_sub_sim.append(size_sub_np[:, args.al_round].mean())
            size_comm_sim.append(size_comm_np[:, args.al_round].mean())
            
            f1_list_sim_default.append(f1_np[:, 0].mean() * 100)
            recall_list_sim_default.append(recall_np[:, 0].mean() * 100)
            precision_list_sim_default.append(precision_np[:, 0].mean() * 100)
            jaccard_list_sim_default.append(jaccard_np[:, 0].mean() * 100)
            nmi_list_sim_default.append(nmi_np[:, 0].mean() * 100)
            
            f1_sub_list_sim_default.append(f1_sub_np[:, 0].mean() * 100)
            recall_sub_list_sim_default.append(recall_sub_np[:, 0].mean() * 100)
            precision_sub_list_sim_default.append(precision_sub_np[:, 0].mean() * 100)
            jaccard_sub_list_sim_default.append(jaccard_sub_np[:, 0].mean() * 100)
            nmi_subgraph_list_sim_default.append(nmi_subgraph_np[:, 0].mean() * 100)
            
            size_sub_sim_default.append(size_sub_np[:, 0].mean())
            size_comm_sim_default.append(size_comm_np[:, 0].mean())
        
        # print all arguments
        print('-' * 200)
        print('-' * 200)
        print('Arguments:')
        for arg in vars(args):
            print(arg, getattr(args, arg))

        print('-' * 200)
        
        print('Multiple Graphs Simulation')
        print("Default")
        print(f'Average Non-AL F1: {np.mean(f1_list_sim_default):.2f} ± {np.std(f1_list_sim_default):.2f}')
        print(f'Average Non-AL Recall: {np.mean(recall_list_sim_default):.2f} ± {np.std(recall_list_sim_default):.2f}')
        print(f'Average Non-AL Precision: {np.mean(precision_list_sim_default):.2f} ± {np.std(precision_list_sim_default):.2f}')
        print(f'Average Non-AL Jaccard: {np.mean(jaccard_list_sim_default):.2f} ± {np.std(jaccard_list_sim_default):.2f}')
        print(f'Average Non-AL NMI: {np.mean(nmi_list_sim_default):.2f} ± {np.std(nmi_list_sim_default):.2f}')
        
        print('-' * 200)
        print(f'Average Non-AL F1 Subgraph: {np.mean(f1_sub_list_sim_default):.2f} ± {np.std(f1_sub_list_sim_default):.2f}')
        print(f'Average Non-AL Recall Subgraph: {np.mean(recall_sub_list_sim_default):.2f} ± {np.std(recall_sub_list_sim_default):.2f}')
        print(f'Average Non-AL Precision Subgraph: {np.mean(precision_sub_list_sim_default):.2f} ± {np.std(precision_sub_list_sim_default):.2f}')
        print(f'Average Non-AL Jaccard Subgraph: {np.mean(jaccard_sub_list_sim_default):.2f} ± {np.std(jaccard_sub_list_sim_default):.2f}')
        print(f'Average Non-AL NMI Subgraph: {np.mean(nmi_subgraph_list_sim_default):.2f} ± {np.std(nmi_subgraph_list_sim_default):.2f}')
        
        print('-' * 200)
        print(f'Average Non-AL Size Subgraph: {np.mean(size_sub_sim_default):.2f} ± {np.std(size_sub_sim_default):.2f}')
        print(f'Average Non-AL Size Community: {np.mean(size_comm_sim_default):.2f} ± {np.std(size_comm_sim_default):.2f}')
        
        print('-' * 200)
        print("Interactive")
        print(f'Average AL F1: {np.mean(f1_list_sim):.2f} ± {np.std(f1_list_sim):.2f}')
        print(f'Average AL Recall: {np.mean(recall_list_sim):.2f} ± {np.std(recall_list_sim):.2f}')
        print(f'Average AL Precision: {np.mean(precision_list_sim):.2f} ± {np.std(precision_list_sim):.2f}')
        print(f'Average AL Jaccard: {np.mean(jaccard_list_sim):.2f} ± {np.std(jaccard_list_sim):.2f}')
        print(f'Average AL NMI: {np.mean(nmi_list_sim):.2f} ± {np.std(nmi_list_sim):.2f}')
        
        print('-' * 200)
        print(f'Average AL F1 Subgraph: {np.mean(f1_sub_list_sim):.2f} ± {np.std(f1_sub_list_sim):.2f}')
        print(f'Average AL Recall Subgraph: {np.mean(recall_sub_list_sim):.2f} ± {np.std(recall_sub_list_sim):.2f}')
        print(f'Average AL Precision Subgraph: {np.mean(precision_sub_list_sim):.2f} ± {np.std(precision_sub_list_sim):.2f}')
        print(f'Average AL Jaccard Subgraph: {np.mean(jaccard_sub_list_sim):.2f} ± {np.std(jaccard_sub_list_sim):.2f}')
        print(f'Average AL NMI Subgraph: {np.mean(nmi_subgraph_list_sim):.2f} ± {np.std(nmi_subgraph_list_sim):.2f}')
        
        print('-' * 200)
        print(f'Average AL Size Subgraph: {np.mean(size_sub_sim):.2f} ± {np.std(size_sub_sim):.2f}')
        print(f'Average AL Size Community: {np.mean(size_comm_sim):.2f} ± {np.std(size_comm_sim):.2f}')
        
        print('-' * 200)
        print('-' * 200)
        print()
            
    else:
        comms_index, f1_list, recall_list, precision_list, jaccard_list, nmi_list, \
            t_sub_list, t_gnn_list, t_al_list, t_eval_list, \
            f1_sub_list, recall_sub_list, precision_sub_list, jaccard_sub_list, nmi_subgraph_list, \
            size_sub, size_comm, subgraph_ablation_para_track_list = \
            PPR_ICS.from_args(args).binary_cls_revise_active_learning()

        # print results
        t_sub_sum = [sum(t) for comm_index, t in t_sub_list.items()]
        t_gnn_sum = [sum(t) for comm_index, t in t_gnn_list.items()]
        t_al_sum = [sum(t) for comm_index, t in t_al_list.items()]
        t_eval_sum = [sum(t) for comm_index, t in t_eval_list.items()]
        t_all_sum = [sum(t_sub_list[comm_index]) + sum(t_gnn_list[comm_index]) + sum(t_al_list[comm_index]) + sum(t_eval_list[comm_index]) for comm_index in comms_index]

        print(f'average time for subgraph generation:\t {sum(t_sub_sum) / len(t_sub_sum):.4f}\n standard deviation: {np.std(t_sub_sum):.4f}')
        print(f'average time for GNN training:\t {sum(t_gnn_sum) / len(t_gnn_sum):.4f}\n standard deviation: {np.std(t_gnn_sum):.4f}')
        print(f'average time for active learning:\t {sum(t_al_sum) / len(t_al_sum):.4f}\n standard deviation: {np.std(t_al_sum):.4f}')
        print(f'average time for evaluation:\t {sum(t_eval_sum) / len(t_eval_sum):.4f}\n standard deviation: {np.std(t_eval_sum):.4f}')
        print(f'average total time:\t {sum(t_all_sum) / len(t_all_sum):.4f}\n standard deviation: {np.std(t_all_sum):.4f}')
        print(f'total time:\t {sum(t_all_sum):.4f}')
        print('-' * 200)

        # padding the results for each community
        for comm_index in comms_index:
            if len(f1_list[comm_index]) < args.al_round + 1:
                f1_list[comm_index] += [f1_list[comm_index][-1]] * (args.al_round + 1 - len(f1_list[comm_index]))
                recall_list[comm_index] += [recall_list[comm_index][-1]] * (args.al_round + 1 - len(recall_list[comm_index]))
                precision_list[comm_index] += [precision_list[comm_index][-1]] * (args.al_round + 1 - len(precision_list[comm_index]))
                jaccard_list[comm_index] += [jaccard_list[comm_index][-1]] * (args.al_round + 1 - len(jaccard_list[comm_index]))
                nmi_list[comm_index] += [nmi_list[comm_index][-1]] * (args.al_round + 1 - len(nmi_list[comm_index]))
                
                f1_sub_list[comm_index] += [f1_sub_list[comm_index][-1]] * (args.al_round + 1 - len(f1_sub_list[comm_index]))
                recall_sub_list[comm_index] += [recall_sub_list[comm_index][-1]] * (args.al_round + 1 - len(recall_sub_list[comm_index]))
                precision_sub_list[comm_index] += [precision_sub_list[comm_index][-1]] * (args.al_round + 1 - len(precision_sub_list[comm_index]))
                jaccard_sub_list[comm_index] += [jaccard_sub_list[comm_index][-1]] * (args.al_round + 1 - len(jaccard_sub_list[comm_index]))
                nmi_subgraph_list[comm_index] += [nmi_subgraph_list[comm_index][-1]] * (args.al_round + 1 - len(nmi_subgraph_list[comm_index]))
                
                size_sub[comm_index] += [size_sub[comm_index][-1]] * (args.al_round + 1 - len(size_sub[comm_index]))
                size_comm[comm_index] += [size_comm[comm_index][-1]] * (args.al_round + 1 - len(size_comm[comm_index]))
                if args.is_kcore or args.is_ktruss or args.is_metis or args.is_metis_cohesive:
                    subgraph_ablation_para_track_list[comm_index] += [subgraph_ablation_para_track_list[comm_index][-1]] * (args.al_round + 1 - len(subgraph_ablation_para_track_list[comm_index]))
            # assert len(f1_list[comm_index]) == args.al_round + 1, print(len(f1_list[comm_index]))
            # assert len(recall_list[comm_index]) == args.al_round + 1, print(len(recall_list[comm_index]))
            # assert len(precision_list[comm_index]) == args.al_round + 1, print(len(precision_list[comm_index]))
        
        f1_np = np.array([f1 for comm_index, f1 in f1_list.items()])
        precision_np = np.array([precision for comm_index, precision in precision_list.items()])
        recall_np = np.array([recall for comm_index, recall in recall_list.items()])
        jaccard_np = np.array([jaccard for comm_index, jaccard in jaccard_list.items()])
        nmi_np = np.array([nmi for comm_index, nmi in nmi_list.items()])
        
        f1_sub_np = np.array([f1 for comm_index, f1 in f1_sub_list.items()])
        precision_sub_np = np.array([precision for comm_index, precision in precision_sub_list.items()])
        recall_sub_np = np.array([recall for comm_index, recall in recall_sub_list.items()])
        jaccard_sub_np = np.array([jaccard for comm_index, jaccard in jaccard_sub_list.items()])
        nmi_subgraph_np = np.array([nmi for comm_index, nmi in nmi_subgraph_list.items()])
        
        size_sub_np = np.array([size for comm_index, size in size_sub.items()])
        size_comm_np = np.array([size for comm_index, size in size_comm.items()])
        # subgraph ablation parameter
        if args.is_kcore or args.is_ktruss or args.is_metis or args.is_metis_cohesive:
            subgraph_ablation_para_track_np = np.array([para for comm_index, para in subgraph_ablation_para_track_list.items()])
        
        # print all arguments
        print('-' * 200)
        print('-' * 200)
        print('Arguments:')
        for arg in vars(args):
            print(arg, getattr(args, arg))

        print('-' * 200)
        
        for i in range(args.al_round + 1):
            print(f'Round {i}:')
            print('Community-level:')
            print(f'Avg Recall:\t {recall_np[:, i].mean() * 100:.2f}\t\t', f'Avg Precision:\t {precision_np[:, i].mean() * 100:.2f}')
            print(f'Avg F1:\t {f1_np[:, i].mean() * 100:.2f}\t\t', f'Avg Jaccard:\t {jaccard_np[:, i].mean() * 100:.2f}\t\t', f'Avg NMI:\t {nmi_np[:, i].mean() * 100:.2f}')
            print(f'Avg Size:\t {size_comm_np[:, i].mean():.2f}')
            
            print('Subgraph-level:')
            print(f'Avg Recall:\t {recall_sub_np[:, i].mean() * 100:.2f}\t\t', f'Avg Precision:\t {precision_sub_np[:, i].mean() * 100:.2f}')
            print(f'Avg F1:\t {f1_sub_np[:, i].mean() * 100:.2f}\t\t', f'Avg Jaccard:\t {jaccard_sub_np[:, i].mean() * 100:.2f}\t\t', f'Avg NMI:\t {nmi_subgraph_np[:, i].mean() * 100:.2f}')
            print(f'Avg Size:\t {size_sub_np[:, i].mean():.2f}')
            
            if args.is_kcore or args.is_ktruss:
                print(f'Avg k:\t {subgraph_ablation_para_track_np[:, i].mean():.2f}')
                print(f'Min k:\t {subgraph_ablation_para_track_np[:, i].min():.2f}')
            elif args.is_metis or args.is_metis_cohesive:
                print(f'Avg num_partition:\t {subgraph_ablation_para_track_np[:, i].mean():.2f}')
                print(f'Min num_partition:\t {subgraph_ablation_para_track_np[:, i].min():.2f}')
            
            print('-' * 200)
        print('-' * 200)
        print()