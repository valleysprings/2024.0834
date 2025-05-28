import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Community search using GNN with Active Learning.')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='amazon',
                      choices=['amazon', 'youtube', 'dblp', 'lj', 'orkut', 'cora', 'citeseer', 
                              'pubmed', 'dolphins', 'karate', 'eu-core', 'football',
                              'eu-core_undirected', 'eu-core_directed', 'yelp', 'amazon_cn'],
                      help='Dataset to use')
    parser.add_argument('--simulation', action='store_true',
                      help='Whether to use simulation graph')
    parser.add_argument('--simulation_graph_type', type=str, default=None,
                      choices=['lfr', 'pp'],
                      help='Type of simulation graph to use')
    parser.add_argument('--simulation_details', nargs='+', type=str, default=[],
                      help='Details for simulation graph')
    parser.add_argument('--num_graphs', type=int, default=1,
                      help='Number of graphs to test (only for simulation)')
    parser.add_argument('--embedding', type=str, default='node2vec',
                      choices=['node2vec', 'grarep', 'diff2vec', 'nodesketch'],
                      help='Type of embedding method to use')
    parser.add_argument('--original', action='store_true',
                      help='Whether to use the original embeddings, only for cora, citeseer, pubmed')
    parser.add_argument('--directed_using_undirected_embedding', action='store_true',
                      help='Whether to use the undirected embedding for directed graph, only for eu-core_directed')
    parser.add_argument('--directed_using_undirected_PPR', action='store_true',
                      help='Whether to use the undirected PPR for directed graph, only for eu-core_directed')
    parser.add_argument('--directed_using_undirected_GNN', action='store_true',
                      help='Whether to use the undirected GNN for directed graph, only for eu-core_directed')

    # label
    parser.add_argument('--pos_num', type=int, default=4, # 1 for core, 3 for others (default to ICS and COCLEP training stage)
                      help='Number of positive labels per community at the first round')
    parser.add_argument('--neg_num', type=int, default=3,
                      help='Number of negative labels per community at the first round')
    parser.add_argument('--num_queries', type=int, default=100, 
                      help='Number of CS queries')

    # subgraph
    parser.add_argument('--not_subgraph', action='store_true', 
                      help='Whether to use subgraph')
    parser.add_argument('--subgraph_size_fixed', action='store_true',
                      help='Whether to use fixed subgraph size')
    parser.add_argument('--subgraph_max_node', type=int, default=200,
                      help='Maximum number of nodes in a subgraph')
    parser.add_argument('--subgraph_min_node', type=int, default=10,
                      help='Minimum number of nodes in a subgraph')
    parser.add_argument('--is_bfs', action='store_true',
                      help='Whether to use BFS for subgraph generation, only for ablation study')
    parser.add_argument('--is_simrank', action='store_true',
                      help='Whether to use SimRank for subgraph generation, only for ablation study')
    parser.add_argument('--is_panther', action='store_true',
                      help='Whether to use Panther for subgraph generation, only for ablation study')
    parser.add_argument('--is_kcore', action='store_true',
                      help='Whether to use k-core for subgraph generation, only for ablation study')
    parser.add_argument('--is_ktruss', action='store_true',
                      help='Whether to use k-truss for subgraph generation, only for ablation study')
    parser.add_argument('--is_metis', action='store_true',
                      help='Whether to use metis for subgraph generation, only for ablation study')
    parser.add_argument('--is_metis_cohesive', action='store_true',
                      help='Whether to use metis cohesive for subgraph generation, only for ablation study')
    parser.add_argument('--k_selection', type=int, default=-1,
                      help='k for k-core or k-truss, -1 for adaptable')
    parser.add_argument('--num_partition', type=int, default=-1,
                      help='Number of partitions for metis, -1 for adaptable')

    # PPR
    parser.add_argument('--ppr_threshold', type=float, default=0.25,
                      help='Threshold for PPR method (old version)')
    parser.add_argument('--degree_normalized', action='store_true',
                      help='Whether to use degree normalized PPR')
    parser.add_argument('--damping', type=float, default=0.95,
                      help='Damping factor for PPR')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                      help='Epsilon for PPR')
    parser.add_argument('--not_log_transform', action='store_true',
                      help='Whether not to use log transform for PPR')

    # active learning
    parser.add_argument('--al_method', type=str, default='uncertainty',
                      choices=['random', 'uncertainty', 'entropy', 'ppr', 
                              'subgraph_degree', 'global_degree', 'aggregated'],
                      help='Active learning method')
    parser.add_argument('--deterministic', action='store_true',
                      help='Whether to use deterministic active learning or not')
    parser.add_argument('--incremental_num', type=int, default=3,
                      help='Number of nodes to incrementally add during active learning')
    parser.add_argument('--al_round', type=int, default=1,
                      help='Number of rounds for active learning')
    parser.add_argument('--temperature_1', type=float, default=1.0,
                      help='Temperature for aggregation')
    parser.add_argument('--temperature_2', type=float, default=1.0,
                      help='Temperature for aggregation')
    parser.add_argument('--percentile_normalize', action='store_true',
                      help='Whether to use percentile normalization for aggregation, default is min-max normalization')
    parser.add_argument('--not_time_decay', action='store_true',
                      help='Whether not to use time decay for aggregation')
    parser.add_argument('--simple_aggregation', action='store_true',
                      help='Whether to use simple aggregation')
    parser.add_argument('--beta_sampled_aggregation', action='store_true',
                      help='Whether to use beta sampled aggregation')

    # GNN
    parser.add_argument('--model', type=str, default='GCN',
                      choices=['GCN', 'GraphSAGE', 'GAT', 'MLP'],
                      help='Type of GNN model to use')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs for GNN model')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for GNN model')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[128],
                      help='Hidden layers for GNN model')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate for GNN model')
    parser.add_argument('--gnn_threshold', type=float, default=0.5,
                      help='Threshold for GNN model')
    parser.add_argument('--gnn_learned_embeddings', action='store_true',
                      help='Whether to use learned embeddings for GNN model')
    parser.add_argument('--gnn_embed_size', type=int, default=128,
                      help='Size of learned embeddings for GNN model')
    parser.add_argument('--dynamic_subgraph_method', type=str, default=None,
                      choices=['coreness', 'ppr', 'gaussian', 'nodesketch'],
                      help='Dynamic subgraph embeddings for GNN model (compute on the fly)')
    parser.add_argument('--use_ELPH_modified', action='store_true',
                      help='Whether to use modified ELPH for subgraph sketching')
    parser.add_argument('--not_reset_parameters', action='store_true',
                      help='Whether not to reset parameters for GNN model, only for GNN, not for GNN-learned-embeddings, as each round node number is different')

    # community
    parser.add_argument('--community_size_fixed', action='store_true',
                      help='Whether to use fixed community size')
    parser.add_argument('--community_size', type=int, default=5,
                      help='Size of community if not adaptable, only for ablation study')
    parser.add_argument('--sim1_threshold', type=float, default=0.9,
                      help='Threshold for subgraph similarity 1')
    parser.add_argument('--sim2_threshold', type=float, default=1.0,
                      help='Threshold for subgraph similarity 2')
    parser.add_argument('--loss_ma_threshold', type=float, default=1e-2,
                      help='Threshold for moving average loss')

    # other
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                      help='Whether to print out intermediate results')
    parser.add_argument('--draw', action='store_true',
                      help='Whether to draw the graph')

    return parser.parse_args()