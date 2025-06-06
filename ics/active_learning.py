import numpy as np
from torch_geometric.utils import degree
from scipy.stats import rankdata
from auxiliary.utils import model_eval

epsilon = 1e-12

def filter_indices(subgraph_pyg):
    selected_indices = [i for i in range(subgraph_pyg.num_nodes) if subgraph_pyg.train_mask[i] == 0]
    selected_indices = np.array(selected_indices)

    return selected_indices


def random_sampling(selected_indices, incremental_num):
    # randomly select k nodes
    selected_indices = np.random.choice(selected_indices, incremental_num, replace=False)

    return selected_indices


def subgraph_degree_sampling(selected_indices, subgraph_pyg, incremental_num, deterministic):
    # based on the degree of the nodes, select the top-k nodes
    degrees_out = degree(subgraph_pyg.edge_index[0], subgraph_pyg.num_nodes).to("cpu").detach().numpy()
    degrees_in = degree(subgraph_pyg.edge_index[1], subgraph_pyg.num_nodes).to("cpu").detach().numpy()
    degrees = np.sum([degrees_out, degrees_in], axis=0)
    degrees = degrees[selected_indices]
    
    if deterministic or len(selected_indices) == 1:
        selected_indices = selected_indices[np.argsort(degrees)[-incremental_num:]]
    else:
        selected_indices = np.random.choice(selected_indices, incremental_num, replace=False,
                                            p=(degrees + epsilon) / np.sum(degrees + epsilon))

    return selected_indices


def global_degree_sampling(selected_indices, subgraph_pyg, original_pyg, incremental_num, deterministic):
    # based on the degree of the nodes, select the top-k nodes
    degrees_out = degree(original_pyg.edge_index[0], original_pyg.num_nodes).to("cpu").detach().numpy()
    degrees_in = degree(original_pyg.edge_index[1], original_pyg.num_nodes).to("cpu").detach().numpy()
    degrees = np.sum([degrees_out, degrees_in], axis=0)
    original_idx = subgraph_pyg.gt_index[selected_indices]
    degrees = degrees[original_idx]
    
    if deterministic or len(selected_indices) == 1:
        selected_indices = selected_indices[np.argsort(degrees)[-incremental_num:]]
    else:
        selected_indices = np.random.choice(selected_indices, incremental_num, replace=False,
                                            p=(degrees + epsilon) / np.sum(degrees + epsilon))

    return selected_indices


def uncertainty_sampling(selected_indices, uncertainty_scores, incremental_num, deterministic):
    # based on the uncertainty scores, select the top-k nodes, note that uncertainty_scores closer to 0.5 are more uncertain
    uncertainty_scores = uncertainty_scores[selected_indices]
    uncertainty_indices = np.argsort(np.abs(uncertainty_scores - 0.5))
    
    if deterministic:
        selected_indices = selected_indices[uncertainty_indices[:incremental_num].tolist()]
    else:
        selected_indices = np.random.choice(selected_indices, incremental_num, replace=False,
                                            p=((0.5 - np.abs(uncertainty_scores - 0.5)) + epsilon)/ np.sum(
                                                0.5 - np.abs(uncertainty_scores - 0.5) + epsilon))

    return selected_indices


def entropy_sampling(selected_indices, uncertainty_scores, incremental_num, deterministic):
    # based on the entropy scores, select the top-k nodes, note that it is the same as uncertainty scores
    entropy_scores = - uncertainty_scores * np.log(uncertainty_scores) - (1 - uncertainty_scores) * np.log(
        1 - uncertainty_scores)
    entropy_scores[np.isnan(entropy_scores)] = 0
    entropy_scores = entropy_scores[selected_indices]
    entropy_indices = np.argsort(entropy_scores)
    
    if deterministic:
        selected_indices = selected_indices[entropy_indices[-incremental_num:].tolist()]
    else:
        selected_indices = np.random.choice(selected_indices, incremental_num, replace=False,
                                            p=(entropy_scores + epsilon) / np.sum(entropy_scores + epsilon))

    return selected_indices


def ppr_sampling(selected_indices, subgraph_pyg, ppr_vec, incremental_num, deterministic):
    # based on the ppr scores, select the top-k nodes
    original_idx = subgraph_pyg.gt_index[selected_indices]
    ppr_score = ppr_vec[original_idx]
    ppr_indices = np.argsort(ppr_score)
    
    if deterministic:
        selected_indices = selected_indices[ppr_indices[-incremental_num:]]
    else:
        selected_indices = np.random.choice(selected_indices, incremental_num, replace=False,
                                            p=(ppr_score+ epsilon)/ np.sum(ppr_score + epsilon))

    return selected_indices

def aggregated_sampling(selected_indices, uncertainty_scores, ppr_vec, 
                        subgraph_pyg, original_pyg, 
                        temperature_1=1.0, temperature_2=1.0, 
                        current_active_learning_round=0, max_active_learning_round=3, 
                        incremental_num=3, deterministic=True, 
                        percentile_normalize=False, time_decay=True, 
                        simple_aggregation=False, 
                        beta_sampled_aggregation=False, 
                        verbose=False):
    # based on the entropy scores
    entropy_scores = - uncertainty_scores * np.log(uncertainty_scores) - (1 - uncertainty_scores) * np.log(
        1 - uncertainty_scores)
    entropy_scores[np.isnan(entropy_scores)] = 0
    entropy_scores = entropy_scores[selected_indices]

    original_idx = subgraph_pyg.gt_index[selected_indices]
    ppr_score = ppr_vec[original_idx]

    # choose ppr as the topo_score
    topo_score = ppr_score
    # topo_score = degrees
    info_score = entropy_scores
    
    # aggregation

    if len(selected_indices) == 1:
        aggregation_scores = 1e10 # a large number, so it will be selected

    else:
        if percentile_normalize:
            topo_score = rankdata(topo_score) / len(topo_score)
            info_score = rankdata(info_score) / len(info_score)
        else:

            if max(topo_score) != min(topo_score):
                topo_score = (topo_score - min(topo_score)) / (max(topo_score) - min(topo_score) + epsilon)

            if max(info_score) != min(info_score):
                info_score = (info_score - min(info_score)) / (max(info_score) - min(info_score) + epsilon)
            
        if simple_aggregation:
            aggregation_scores = topo_score + info_score
            
        elif beta_sampled_aggregation:
            a, b = 2, 2 * (max_active_learning_round - current_active_learning_round + 1) / (max_active_learning_round + 1)
            sample = np.random.beta(a, b, size=1)
            aggregation_scores = topo_score * (1 - sample) + info_score * sample
        else:

            if time_decay:
                aggregation_scores = np.log(np.exp(topo_score / (temperature_1 * (current_active_learning_round + 1))) + 
                                    np.exp(info_score / (temperature_2 * (max_active_learning_round - current_active_learning_round))))
            else:
                aggregation_scores = np.log(np.exp(topo_score / temperature_1) + np.exp(info_score / temperature_2))


    if deterministic or sum(aggregation_scores < 0) > 0:
        selected_indices = selected_indices[np.argsort(aggregation_scores)[-incremental_num:]]
    else:
        selected_indices = np.random.choice(selected_indices, incremental_num, replace=False,
                                            p=(aggregation_scores + epsilon) / np.sum(aggregation_scores + epsilon))
    
    return selected_indices


def active_learning(model, subgraph_pyg, device, method, deterministic, incremental_num=3, original_pyg=None,
                    ppr_vec=None, temperature_1=1.0, temperature_2=1.0, current_active_learning_round=0, max_active_learning_round=3, 
                    percentile_normalize=False, time_decay=True, simple_aggregation=False, beta_sampled_aggregation=False, verbose=False):
    uncertainty_scores = model_eval(model, subgraph_pyg, origin=True, device=device).to("cpu").detach().numpy()
    selected_indices = filter_indices(subgraph_pyg)

    if verbose:
        print(f'line 173 in active_learning.py: current active learning round: {current_active_learning_round}')
        print(f'line 174 in active_learning.py: selected_indices: {selected_indices}')


    incremental_num = min(incremental_num, len(selected_indices))

    if method == "uncertainty":
        selected_indices = uncertainty_sampling(selected_indices, uncertainty_scores, incremental_num, deterministic)
    elif method == "entropy":
        selected_indices = entropy_sampling(selected_indices, uncertainty_scores, incremental_num, deterministic)
    elif method == "ppr":
        selected_indices = ppr_sampling(selected_indices, subgraph_pyg, ppr_vec, incremental_num, deterministic)
    elif method == "subgraph_degree":
        selected_indices = subgraph_degree_sampling(selected_indices, subgraph_pyg, incremental_num, deterministic)
    elif method == "global_degree":
        selected_indices = global_degree_sampling(selected_indices, subgraph_pyg, original_pyg, incremental_num,
                                                  deterministic)
    elif method == "aggregated":
        selected_indices = aggregated_sampling(selected_indices = selected_indices, uncertainty_scores = uncertainty_scores, ppr_vec = ppr_vec, 
                                               subgraph_pyg = subgraph_pyg, original_pyg = original_pyg, 
                                               temperature_1 = temperature_1, temperature_2 = temperature_2, 
                                               current_active_learning_round = current_active_learning_round, 
                                               max_active_learning_round = max_active_learning_round, 
                                               incremental_num = incremental_num, deterministic = deterministic, 
                                               percentile_normalize = percentile_normalize, time_decay = time_decay, 
                                               simple_aggregation = simple_aggregation, 
                                               beta_sampled_aggregation = beta_sampled_aggregation, 
                                               verbose = verbose)
    else:
        selected_indices = random_sampling(selected_indices, incremental_num)

    if verbose:
        print(f'line 205 in active_learning.py: selected indices: {subgraph_pyg.gt_index[selected_indices]}')

    pos = [i for i in selected_indices if subgraph_pyg.y[i] == 1]
    neg = [i for i in selected_indices if subgraph_pyg.y[i] == 0]
    pos = subgraph_pyg.gt_index[pos]
    neg = subgraph_pyg.gt_index[neg]

    return pos, neg
