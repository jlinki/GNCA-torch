import torch
import torch.nn as nn
from torch_geometric.nn import GeneralConv


class GNNCASimple(nn.Module):
    """
    GNCA that uses GeneralConv to update the state.
    """

    def __init__(
        self,
        input_dim,
        activation=None,
        message_passing=1,
        batch_norm=False,
        hidden=256,
        connectivity="cat",
        aggregate="add",
        **kwargs
    ):
        super(GNNCASimple, self).__init__(**kwargs)
        self.activation = activation
        self.message_passing = message_passing
        self.batch_norm = batch_norm
        self.hidden = hidden
        self.connectivity = connectivity
        self.aggregate = aggregate

        self.input_dim = input_dim
        self.mp = GeneralConv(
            in_channels=self.hidden,
            out_channels=self.hidden,
            aggr=self.aggregate,
            root_weight=False,
            bias=True,
        )
        self.encoder = MLP(self.input_dim, self.hidden, batch_norm=self.batch_norm, activation=self.activation, final_activation=self.activation)
        if self.connectivity == "cat":
            self.decoder = MLP(2*self.hidden, self.input_dim, batch_norm=self.batch_norm, activation=self.activation, final_activation='tanh')
        elif self.connectivity == "add":
            self.decoder = MLP(self.hidden, self.input_dim, batch_norm=self.batch_norm, activation=self.activation, final_activation='tanh')
        else:
            raise ValueError("Unknown connectivity type")

    def forward(self, x, edge_index, steps):
        for _ in range(steps):
            x = self.encoder(x)
            out = self.mp(x, edge_index)
            if self.connectivity == "cat":
                x = self.decoder(torch.cat((out, x), dim=-1))
            elif self.connectivity == "add":
                x = self.decoder(out + x)
            else:
                raise ValueError("Unknown connectivity type")
        return x


class MLP(nn.Module):
    def __init__(
            self,
            input,
            output,
            hidden=256,
            layers=2,
            batch_norm=True,
            dropout=0.0,
            activation="relu",
            final_activation=None,
    ):
        super(MLP, self).__init__()
        self.config = {
            "input": input,
            "output": output,
            "hidden": hidden,
            "layers": layers,
            "batch_norm": batch_norm,
            "dropout": dropout,
            "activation": activation,
            "final_activation": final_activation,
        }
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        layers_list = []
        for i in range(layers):
            if i == 0:
                in_dim = input
            else:
                in_dim = hidden
            out_dim = hidden if i < layers - 1 else output
            layers_list.append(nn.Linear(in_dim, out_dim))
            if self.batch_norm:
                layers_list.append(nn.BatchNorm1d(out_dim))
            layers_list.append(nn.Dropout(self.dropout_rate))
            activation_i = activation if i < layers - 1 else final_activation
            layers_list.append(self.get_act(activation_i))

        self.mlp = nn.Sequential(*layers_list)

    def forward(self, inputs):
        return self.mlp(inputs)

    def get_config(self):
        return self.config

    # Define the activation functions
    def get_act(self, activation):
        if activation == "prelu":
            return nn.PReLU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        elif activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


    # def steps(self, x, edge_index, steps):
    #     for _ in range(steps):
    #         x = self.forward(x, edge_index)
    #
    #     return x


# import torch
# import torch.nn as nn
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
#
# class GNNCASimple(MessagePassing):
#     """
#     GNCA that uses MessagePassing (from PyTorch Geometric) for the update step.
#     """
#
#     def __init__(
#         self,
#         activation=None,
#         message_passing=1,
#         batch_norm=False,
#         hidden=256,
#         hidden_activation="relu",
#         connectivity="cat",
#         **kwargs
#     ):
#         super(GNNCASimple, self).__init__(aggr='add', **kwargs)
#         self.activation = activation
#         self.message_passing = message_passing
#         self.batch_norm = batch_norm
#         self.hidden = hidden
#         self.hidden_activation = hidden_activation
#         self.connectivity = connectivity
#
#         self.lin = nn.Linear(hidden, hidden)
#
#         if self.batch_norm:
#             self.bn = nn.BatchNorm1d(hidden)
#
#     def forward(self, x, edge_index):
#         if self.message_passing > 1:
#             for _ in range(self.message_passing):
#                 x = self.propagate(edge_index, x=x)
#         else:
#             x = self.propagate(edge_index, x=x)
#
#         return x
#
#     def message(self, x_j, edge_weight):
#         # Perform a linear transformation on the source node features (x_j)
#         return edge_weight.view(-1, 1) * x_j
#
#     def update(self, aggr_out):
#         # The aggregation operation is simply taking the mean
#         return aggr_out
#
#     # def update(self, aggr_out):
#     #     x = self.lin(aggr_out)
#     #     if self.batch_norm:
#     #         x = self.bn(x)
#     #     if self.activation is not None:
#     #         x = self.activation(x)
#     #
#     #     return x
#
#     # def message_and_aggregate(self, adj_t, x):
#     #     edge_index, _, size = adj_t.storage()
#     #     row, col = edge_index
#     #
#     #     if self.connectivity == 'cat':
#     #         x_i = x[row]
#     #         x_j = x[col]
#     #     elif self.connectivity == 'sum':
#     #         x_i = x
#     #
#     #     out = self.message(x_j - x_i)
#     #
#     #     return scatter_add(out, col, dim=0, dim_size=size[0])
#     #
#     # def propagate(self, edge_index, size=None, **kwargs):
#     #     adj_t = edge_index
#     #     if size is None:
#     #         size = (None, None)
#     #
#     #     x = kwargs['x']
#     #
#     #     out = self.message_and_aggregate(adj_t, x)
#     #
#     #     return self.update(out)
