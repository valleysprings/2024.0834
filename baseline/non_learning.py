import warnings
warnings.filterwarnings("ignore")

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from auxiliary.utils import *
from auxiliary.preprocess_nofeat import *
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from cdlib import algorithms


class tradition_CS:
    def __init__(self, dataset='amazon', num_queries=100, seed=False, simulation=False, simulation_graph_type=None, simulation_details=[]):
        self.dataset = dataset
        self.inv_map = None
        self.simulation = simulation
        self.simulation_graph_type = simulation_graph_type
        self.simulation_details = simulation_details
        if seed:
            seed_torch()
            
        if self.dataset in ['cora', 'citeseer', 'pubmed'] and not self.simulation:
            self.graph_nx, self.comms = self.load_graph_pyg_nonlearning()
        elif self.dataset in ['dolphins', 'karate', 'football', 'polblogs', 'eu-core'] and not self.simulation:
            graph_pyg = torch.load('../preprocess/' + self.dataset + '/' + 'single_node2vec_graph_pyg.pt')
            self.graph_nx = to_networkx(graph_pyg, to_undirected=True)
            self.comms = communities_generation_pyg(graph_pyg)
        elif self.dataset in ['eu-core_undirected'] and not self.simulation:
            graph_pyg = torch.load('../preprocess/' + self.dataset + '/' + 'node2vec_graph_pyg.pt')
            self.graph_nx = to_networkx(graph_pyg, to_undirected=True)
            self.comms = communities_generation_pyg(graph_pyg)
        elif self.dataset in ['eu-core_directed'] and not self.simulation:
            graph_pyg = torch.load('../preprocess/' + self.dataset + '/' + 'node2vec_graph_pyg.pt')
            self.graph_nx = to_networkx(graph_pyg)
            self.comms = communities_generation_pyg(graph_pyg)
        elif self.simulation:
            print(self.simulation_graph_type, self.simulation_details, flush=True)
            graph_pyg = load_graph_pyg_feat_simulation(graph_type=self.simulation_graph_type, details=self.simulation_details)
            self.graph_nx = to_networkx(graph_pyg, to_undirected=True)
            self.comms = communities_generation_pyg(graph_pyg)
        else:
            self.graph_nx, self.inv_map, self.comms = self.load_data_gt_nonlearning()
        # remove loop
        self.graph_nx.remove_edges_from(nx.selfloop_edges(self.graph_nx))
        self.num_nodes = self.graph_nx.number_of_nodes()
        self.num_queries = num_queries
        self.comms_index, self.q = self.query_generation(num_queries)

    def load_graph_pyg_nonlearning(self):
        graph_pyg = Planetoid(root='../dataset', name=self.dataset)
        graph_nx = to_networkx(graph_pyg[0], to_undirected=True)
        comms = communities_generation_pyg(graph_pyg[0])

        return graph_nx, comms
    
    def load_data_gt_nonlearning(self):
        graph_gt, map_gt = parse_ungraph_gt_fast(
            osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset', self.dataset,
                     'com-' + self.dataset + '.ungraph.txt'))
        inv_map = find_inv_map(map_gt)
        comms = read_communities_from_file(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset', self.dataset,
                                                    'com-' + self.dataset + '.top5000.cmty.txt'))
        graph_nx = gt_to_nx(graph_gt)

        return graph_nx, inv_map, comms

    def query_generation(self, num_queries=100):
        self.q = []
        if self.inv_map is None:
            self.comms_index = np.random.choice(np.arange(len(self.comms)), num_queries, replace=True)
            for i in self.comms_index:
                self.q.append(np.random.choice(self.comms[i]))
        else:
            self.comms_index = np.random.choice(np.arange(len(self.comms)), num_queries, replace=False)
            for i in self.comms_index:
                self.q.append(self.inv_map[np.random.choice(self.comms[i])])
        
        return self.comms_index, self.q

    def kcore_fixed(self, k=3):
        subgraph = nx.k_core(self.graph_nx, k=k)
        subgraph = nx.connected_components(subgraph)
        subgraph = [list(i) for i in subgraph]

        return subgraph

    def kcore_adaptable(self):
        suit_subgraph = []
        kcore_graph = []
        max_k = 10
        for i in range(1, max_k):
            t = nx.k_core(self.graph_nx, k=i)
            t = nx.connected_components(t)
            t = [list(j) for j in t]
            kcore_graph.append(t)
            
        for i in self.q:
            adaptable_k = 1
            flag = True
            candidate = None
            while flag:
                flag = False
                for j in kcore_graph[adaptable_k - 1]:
                    if i in j:
                        flag = True
                        adaptable_k += 1
                        candidate = j
                        break
                if adaptable_k >= max_k:
                    flag = False
                if not flag:
                    suit_subgraph.append(candidate)

        return suit_subgraph

    def ktruss_fixed(self, k=3):
        subgraph = nx.k_truss(self.graph_nx, k=k)
        subgraph = nx.connected_components(subgraph)
        subgraph = [list(i) for i in subgraph]

        return subgraph

    def ktruss_adaptable(self):
        suit_subgraph = []
        ktruss_graph = []
        max_k = 10
        for i in range(1, max_k):
            t = nx.k_truss(self.graph_nx, k=i)
            t = nx.connected_components(t)
            t = [list(j) for j in t]
            ktruss_graph.append(t)
            
        for i in self.q:
            adaptable_k = 1
            flag = True
            candidate = None
            while flag:
                flag = False
                for j in ktruss_graph[adaptable_k - 1]:
                    if i in j:
                        flag = True
                        adaptable_k += 1
                        candidate = j
                        break
                if adaptable_k >= max_k:
                    flag = False
                if not flag:
                    suit_subgraph.append(candidate)

        return suit_subgraph

    def kecc_fixed(self, k=3):
        subgraph = nx.k_edge_components(self.graph_nx, k=k)
        subgraph = [list(i) for i in subgraph]

        return subgraph

    def kecc_adaptable(self):
        suit_subgraph = []
        kecc_graph = []
        max_k = 3
        for i in range(1, max_k):
            # print(i,flush=True)
            t = nx.k_edge_components(self.graph_nx, k=i)
            t = [list(j) for j in t]
            kecc_graph.append(t)
            
        for i in self.q:
            adaptable_k = 1
            flag = True
            candidate = None
            while flag:
                flag = False
                for j in kecc_graph[adaptable_k - 1]:
                    if i in j:
                        flag = True
                        adaptable_k += 1
                        candidate = j
                        break
                if adaptable_k >= max_k:
                    flag = False
                if not flag:
                    suit_subgraph.append(candidate)

        return suit_subgraph
    

    def modularity_fixed(self):
        subgraph = nx.community.greedy_modularity_communities(self.graph_nx, cutoff=5000)
        subgraph = [list(i) for i in subgraph]
        
        return subgraph
    
    def louvain_fixed(self, resolution=0.1):
        subgraph = algorithms.louvain(self.graph_nx, resolution=resolution)
        subgraph = [list(i) for i in subgraph.communities]
        
        return subgraph
        
    
    def fixed_evaluation(self, subgraph):
        precision_list, recall_list, f1_list, jaccard_list, nmi_list = [], [], [], [], []
        for i in range(self.num_queries):
            pred_subgraph = []
            ture_subgraph = self.comms[self.comms_index[i]]
            for j in subgraph:
                if self.q[i] in j:
                    pred_subgraph = j
            if len(pred_subgraph) == 0:
                pre, rec, f1, jac, nmi = 0, 0, 0, 0, 0
            else:
                pre, rec, f1, jac, nmi = calculate_stats_tradition(self.num_nodes, self.inv_map, ture_subgraph, pred_subgraph)
            precision_list.append(pre)
            recall_list.append(rec)
            f1_list.append(f1)
            jaccard_list.append(jac)
            nmi_list.append(nmi)

        return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), np.mean(jaccard_list), np.mean(nmi_list)

    def adaptable_evaluation(self, subgraph):
        precision_list, recall_list, f1_list, jaccard_list, nmi_list = [], [], [], [], []
        assert len(subgraph) == self.num_queries
        for i in range(self.num_queries):
            ture_subgraph = self.comms[self.comms_index[i]]
            pred_subgraph = subgraph[i]
            if pred_subgraph is None or len(pred_subgraph) == 0:
                pre, rec, f1, jac, nmi = 0, 0, 0, 0, 0
            else:
                pre, rec, f1, jac, nmi = calculate_stats_tradition(self.num_nodes, self.inv_map, ture_subgraph, pred_subgraph)
            precision_list.append(pre)
            recall_list.append(rec)
            f1_list.append(f1)
            jaccard_list.append(jac)
            nmi_list.append(nmi)

        return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), np.mean(jaccard_list), np.mean(nmi_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Community serach using GNN with Active Learning.')

    parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'youtube', 'dblp', 'lj', 'orkut', 'cora', 'citeseer', 'pubmed', 'dolphins', 'karate', 'football', 'polblogs', 'eu-core_undirected', 'eu-core_directed'],
                        help='Dataset to use')
    parser.add_argument('--simulation', action='store_true',
                      help='Whether to use simulation graph')
    parser.add_argument('--simulation_graph_type', type=str, default=None,
                      choices=['lfr', 'pp'],
                      help='Type of simulation graph to use')
    parser.add_argument('--simulation_details', nargs='+', type=str, default=[],
                      help='Details for simulation graph')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of queries')
    parser.add_argument('--method', type=str, default='kcore_adaptable',
                        choices=['kcore', 'kcore_adaptable', 'ktruss', 'ktruss_adaptable', 'kecc', 'kecc_adaptable', 'modularity', 'kecc_fast', 'louvain'], help='Method to use')
    parser.add_argument('--k', type=int, default=3, help='minimal k value for k-core / k-truss / k-ecc methods')
    parser.add_argument('--seed', action='store_true', help='Seed for reproducibility')
    parser.add_argument('--resolution', type=float, default=0.1, help='Resolution for Louvain method')

    args = parser.parse_args()

    test = tradition_CS(args.dataset, args.num_queries, args.seed, args.simulation, args.simulation_graph_type, args.simulation_details)

    if args.method == 'kcore':
        subgraph = test.kcore_fixed(args.k)
    elif args.method == 'ktruss':
        subgraph = test.ktruss_fixed(args.k)
    elif args.method == 'kecc':
        subgraph = test.kecc_fixed(args.k)
    elif args.method == 'kcore_adaptable':
        subgraph = test.kcore_adaptable()
    elif args.method == 'ktruss_adaptable':
        subgraph = test.ktruss_adaptable()
    elif args.method == 'kecc_adaptable':
        subgraph = test.kecc_adaptable()
    elif args.method == 'modularity':
        subgraph = test.modularity_fixed()
    elif args.method == 'louvain':
        subgraph = test.louvain_fixed(args.resolution)
    else:
        raise NotImplementedError

    if args.method == 'kcore_adaptable' or args.method == 'ktruss_adaptable' or args.method == 'kecc_adaptable' or args.method == 'kecc_fast':
        precision, recall, f1, jaccard, nmi = test.adaptable_evaluation(subgraph)
    else:
        precision, recall, f1, jaccard, nmi = test.fixed_evaluation(subgraph)
       
    print('-' * 200) 
    print(args)
    print(f'Precision: {precision * 100:.2f}, Recall: {recall * 100:.2f}, F1: {f1 * 100:.2f}, Jaccard: {jaccard * 100:.2f}, NMI: {nmi * 100:.2f}')
    print('-' * 200)

