import math
import sys
import numpy as np
import torch
import itertools
from torch import layer_norm, nn
from typing import Union, List

import torch_geometric
from torch_sparse import spmm
from torch.utils.data import Dataset


def get_chord_indices_assym(n_vec, n_link):
    """
    Generates position indicies, based on the Chord protocol (incl. itself).

    :param n_vec: sequence length
    :param n_link: number of links in the Chord protocol
    :return: target indices in two lists, each is of size n_vec * n_link
    """

    rows = list(
        itertools.chain(
            *[
                [i for j in range(n_link)] for i in range(n_vec)
            ]
        )
    )

    cols = list(
        itertools.chain(
            *[
                [i] + [(i + 2 ** k) % n_vec for k in range(n_link - 1)] for i in range(n_vec)
            ]
        )
    )

    return rows, cols


def get_dil_indices_assym(n_vec, n_link, n_layer):
    """
    Generates the position indicies, based on the symmetric Chord protocol (incl. itself).
    So n_link is an odd number
    """
    dil_ws = []
    for n in range(n_layer):
        dilation = 2 ** n
        half_link = int((n_link - 1) / 2)

        rows = list(
            itertools.chain(
                *[
                    [r for _ in range(n_link)] for r in range(n_vec)
                ]
            )
        )
        cols = list(
            itertools.chain(
                *[
                    [i] + [(i + k * dilation) % n_vec for k in range(1, 1 + half_link)] +
                    [(i - k * dilation) % n_vec for k in range(1, 1 + half_link)] for i in range(n_vec)
                ]
            )
        )

        rc_tensor = torch.tensor([rows, cols])
        rc_list = rc_tensor.tolist()
        dil_ws.append(rc_list)

    return dil_ws


def MakeMLP(cfg: List[Union[str, int]], in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Constructs an MLP based on a given structural config.
    """
    layers: List[nn.Module] = []
    for i in cfg:
        if isinstance(i, int):
            layers += [nn.Linear(in_channels, i)]
            in_channels = i
        else:
            layers += [nn.GELU()]
    layers += [nn.Linear(in_channels, out_channels)]
    return nn.Sequential(*layers)


class MLPBlock(nn.Module):
    """
    Constructs a MLP with the specified structure.

    """

    def __init__(self, cfg, in_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.network = MakeMLP(cfg, in_dim, out_dim)

    def forward(self, data):
        return self.network(data)


class AttentionModule(nn.Module):
    def __init__(self,
                 embedding_size,
                 max_seq_len,
                 protocol,
                 ):
        super(AttentionModule, self).__init__()
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.n_W = int(np.log2(self.max_seq_len))
        self.n_links = self.n_W + 1
        self.protocol = protocol
        self.embedding_each_head = int(embedding_size)
        self.Ws = [self.embedding_size, 'GELU']
        self.V = [self.embedding_size, 'GELU']

        if self.protocol == "dil":
            self.n_links = 9
            self.protocol_indicies = torch.tensor(
                get_dil_indices_assym(self.max_seq_len, self.n_links, self.n_W)).cuda()
        elif self.protocol == "chord":
            self.protocol_indicies = torch.tensor(get_chord_indices_assym(self.max_seq_len, self.n_links)).cuda()

        # Init Ws
        self.fs = nn.ModuleList(
            [
                MLPBlock(
                    self.Ws,
                    self.embedding_size,
                    self.n_links
                )
                for _ in range(self.n_W)
            ]
        )

        # Init V
        self.g = MLPBlock(
            self.V,
            self.embedding_size,
            self.embedding_size
        )

        # Init O (head mixing)
        self.o = nn.Linear(self.embedding_size, self.embedding_size, bias=False)


    def forward(self, V, data):
        # Get V
        V = self.g(V)
        w_index = 0
        # Init residual connection
        res_conn = V
        for m in range(self.n_W):
            # Get W_m
            # W = self.fs[h][m](data)
            W = self.fs[m](data)
            # print(W.shape)
            # Multiply W_m and V, get new V

            if self.protocol == "dil":
                V = spmm(
                    self.protocol_indicies[w_index],
                    W.reshape(W.size(0), W.size(1) * W.size(2)),
                    self.max_seq_len,
                    self.max_seq_len,
                    V
                )
            else:
                V = spmm(
                    self.protocol_indicies,
                    W.reshape(W.size(0), W.size(1) * W.size(2)),
                    self.max_seq_len,
                    self.max_seq_len,
                    V
                )
            w_index += 1
            # Vs.append(V)
            V = V + res_conn
        return V


class Paramixer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 max_seq_len,
                 n_layers,
                 n_class,
                 protocol,
                 problem,
                 positional_embedding
                 ):
        super(Paramixer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_class = n_class
        self.protocol = protocol
        self.problem = problem
        self.use_positional_embedding = positional_embedding


        if self.problem == 'adding':
            self.init_linear = nn.Linear(
                2,
                self.embedding_size,
                bias=True
            )

        # Init embedding layer
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size
        )

        # Init APC
        self.apc_embedding = nn.Embedding(
            self.max_seq_len,
            self.embedding_size
        )

        self.attention = nn.ModuleList(
            [
                AttentionModule(self.embedding_size, self.max_seq_len, self.protocol)
                for _ in range(self.n_layers)
            ]

        )

        self.final = nn.Linear(
            self.max_seq_len * self.embedding_size,
            self.n_class
        )

    def forward(self, data):
        if self.problem == 'order':
            data = self.embedding(data).squeeze(-2)
        else :
            data = self.init_linear(data)

        if self.use_positional_embedding:
            # Add APC
            positions = torch.arange(0, self.max_seq_len).expand(data.size(0), self.max_seq_len).cuda()
            pos_embed = self.apc_embedding(positions)
            data = data + pos_embed
        # print(data.shape)
        # Iterate over layers
        V = data
        # print(V.shape)
        for l in range(self.n_layers):
            V = self.attention[l](V, data)
        # print(V.shape)
        V = self.final(V.view(V.size(0), -1))
        # print(V.shape)
        return V
