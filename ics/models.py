import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, MLP
from auxiliary.hashing import ElphHashes
from torch_geometric.utils import add_self_loops
from torch.nn import Linear
from torch_geometric.utils import scatter

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

  
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, layer_type='GCN', dropout=0.5):
        super(GNN, self).__init__()
        self.layer_type = layer_type
        self.setup_layers(in_channels, hidden_layers, out_channels)
        self.dropout = dropout

    def setup_layers(self, in_channels, hidden_layers, out_channels):
        self.layers_size = [in_channels] + hidden_layers + [out_channels]
        self.layers = []
        for i, _ in enumerate(self.layers_size[:-1]):
            if self.layer_type == 'GCN':
                self.layers.append(GCNConv(self.layers_size[i],self.layers_size[i+1]))
            elif self.layer_type == 'GraphSAGE':
                self.layers.append(SAGEConv(self.layers_size[i],self.layers_size[i+1]))
            elif self.layer_type == 'GAT':
                self.layers.append(GATv2Conv(self.layers_size[i],self.layers_size[i+1]))
            elif self.layer_type == 'MLP':
                self.layers.append(MLP(in_channels=self.layers_size[i], out_channels=self.layers_size[i+1], num_layers=1))
        self.layers = ListModule(*self.layers)

    def forward(self, x, edge_index):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = torch.nn.functional.dropout(x, p = self.dropout, training = self.training)
        return torch.sigmoid(x)


class GNN_learned_embeddings(torch.nn.Module):
    def __init__(self, subgraph_size, in_channels, hidden_layers, out_channels, layer_type='GCN', dropout=0.5, device=None):
        super(GNN_learned_embeddings, self).__init__()
        self.layer_type = layer_type
        self.setup_layers(in_channels, hidden_layers, out_channels)
        self.dropout = dropout
        self.device = device
        self.embed = nn.Embedding(subgraph_size, in_channels)

    def setup_layers(self, in_channels, hidden_layers, out_channels):
        self.layers_size = [in_channels] + hidden_layers + [out_channels]
        self.layers = []
        for i, _ in enumerate(self.layers_size[:-1]):
            if self.layer_type == 'GCN':
                self.layers.append(GCNConv(self.layers_size[i],self.layers_size[i+1]))
            elif self.layer_type == 'GraphSAGE':
                self.layers.append(SAGEConv(self.layers_size[i],self.layers_size[i+1]))
            elif self.layer_type == 'GAT':
                self.layers.append(GATv2Conv(self.layers_size[i],self.layers_size[i+1]))
            elif self.layer_type == 'MLP':
                self.layers.append(MLP(in_channels=self.layers_size[i], out_channels=self.layers_size[i+1], num_layers=1))
        self.layers = ListModule(*self.layers)

    def forward(self, x, edge_index):
        x = torch.arange(x.shape[0]).to(self.device)
        # print(f"pre.x.shape: {x.shape}", flush=True)
        x = self.embed(x)
        # print(f"post.x.shape: {x.shape}", flush=True)
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = torch.nn.functional.dropout(x, p = self.dropout, training = self.training)
        return torch.sigmoid(x)

class ELPH_modified(torch.nn.Module):
    """
    modified ELPH for node-level subgraph sketching, max_hops = 2 (for now, meaning hidden_layers = [128])
    """
    def __init__(self, subgraph_size, in_channels, hidden_layers, out_channels, dropout=0.5, device=None):
        super(ELPH_modified, self).__init__()
        self.device = device
        
        # hashing
        self.elph_hashes = ElphHashes()
        self.init_hashes = None
        self.init_hll = None
        self.num_perm = 128
        self.hll_size = 256
        
        # embedding
        self.embed = nn.Embedding(subgraph_size, in_channels)
        
        # GNN
        self.layers_size = [in_channels] + hidden_layers + [out_channels]
        self.num_layers = len(hidden_layers) + 1
        self.feature_dropout = dropout
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(GCNConv(self.layers_size[i], self.layers_size[i+1]))
        
        # linear
        self.dim = 2 * 4 # max_hops * (max_hops + 2)
        self.label_lin_layer = Linear(self.dim, self.dim)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        
        # output
        self.lin = Linear(self.dim + self.layers_size[-1], out_channels)
        
    def forward(self, x, edge_index):
        x = torch.arange(x.shape[0]).to(self.device)
        x = self.embed(x)
        num_nodes = x.size(0)
        hash_edge_index, _ = add_self_loops(edge_index)

        if self.init_hashes is None:
            self.init_hashes = self.elph_hashes.initialise_minhash(num_nodes).to(self.device)
        if self.init_hll is None:
            self.init_hll = self.elph_hashes.initialise_hll(num_nodes).to(self.device)
            
        node_hashings_table = {}
        cards = torch.zeros((num_nodes, self.num_layers))
        
        for k in range(self.num_layers + 1):
            node_hashings_table[k] = {
                'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8, device=self.device),
                'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64, device=self.device)}
            
            if k == 0:
                node_hashings_table[k] = {
                    'hll': self.init_hll,
                    'minhash': self.init_hashes
                }
            else:
                node_hashings_table[k] = {
                    'hll': self.elph_hashes.hll_prop(node_hashings_table[k-1]['hll'], hash_edge_index),
                    'minhash': self.elph_hashes.minhash_prop(node_hashings_table[k-1]['minhash'], hash_edge_index)
                }
                cards[:, k-1] = self.elph_hashes.hll_count(node_hashings_table[k]['hll'])
                
                x = self.convs[k-1](x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.feature_dropout, training=self.training)
        
        
        # transpose edge_index
        edge_index_transpose = edge_index.transpose(0, 1)
        
        sf = self.elph_hashes.get_subgraph_features(edge_index_transpose, node_hashings_table, cards)
        sf = self.label_lin_layer(sf)
        sf = self.bn_labels(sf)
        sf = F.relu(sf)
        sf = F.dropout(sf, p=self.feature_dropout, training=self.training)
        
        # add the node features to the structural features
        # x shape: (num_nodes, out_channels)
        # sf shape: (num_edges, dim)
        # aggregated shape: (num_nodes, out_channels + dim)
        # print(f"x.shape: {x.shape}, sf.shape: {sf.shape}", flush=True) 
        sf_nodes = scatter(sf, edge_index[1], dim=0, dim_size=num_nodes, reduce='sum')
        # print(f"sf_nodes.shape: {sf_nodes.shape}", flush=True)
        x = torch.cat([x, sf_nodes], dim=1)
        
        x = self.lin(x)
        
        return torch.sigmoid(x)
