import torch
import torch_geometric as torch_geometric
from torch_geometric.data import Data
import pygsp

def get_cloud(name, **kwargs):
    graph_class = getattr(pygsp.graphs, name)
    graph = graph_class(**kwargs)

    y = torch.tensor(graph.coords, dtype=torch.float)
    edge_index = torch_geometric.utils.dense_to_sparse(torch.tensor(graph.W.todense(), dtype=torch.float))[0]

    data = Data(x=y, edge_index=edge_index)
    data.name = name

    return data
