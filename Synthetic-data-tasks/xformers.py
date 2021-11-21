import math
import torch
import torch.nn as nn
from performer_pytorch import Performer
from linformer import Linformer
from nystrom_attention import Nystromer
from long_short_transformer import LongShortTransformer
from reformer_pytorch import Reformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerHead(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem
     ):
        super(TransformerHead, self).__init__()
        self.n_vec = n_vec
        self.encoder = nn.Embedding(vocab_size,  dim)
        self.posenc = nn.Embedding(n_vec, dim)
        encoder_layers = TransformerEncoderLayer(dim, heads, dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, depth)
        self.final = nn.Linear(n_vec*dim, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.transformer_encoder(x)
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x).squeeze(-2)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).cuda()
            x = self.posenc(positions) + x
            x = self.transformer_encoder(x)
            x = self.final(x.view(x.size(0), -1))
        return x


class PerformerHead(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem
     ):
        super(PerformerHead, self).__init__()
        self.n_vec = n_vec
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.performer = Performer(
            dim = dim,
            depth=depth,
            heads = heads,
            dim_head=dim,
            causal = True
        )
        self.n_vew = n_vec
        self.final = nn.Linear(n_vec*dim, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.performer(x)
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x).squeeze(-2)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).cuda()
            x = self.posenc(positions) + x
            x = self.performer(x)
            x = self.final(x.view(x.size(0), -1))
        return x


class LinformerHead(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem
     ):
        super(LinformerHead, self).__init__()
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.linformer = Linformer(
            dim=dim,
            seq_len=n_vec,
            depth=depth,
            heads=heads,
            k=256,
            one_kv_head=True,
            share_kv=True
        )
        self.n_vec = n_vec
        self.final = nn.Linear(n_vec*dim, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.linformer(x)
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x).squeeze(-2)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).cuda()
            x = self.posenc(positions) + x
            x = self.linformer(x)
            x = self.final(x.view(x.size(0), -1))
        return x


class ReformerHead(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem
     ):
        super(ReformerHead, self).__init__()
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.reformer = Reformer(
            dim=dim,
            depth=depth,
            heads=heads,
            lsh_dropout=0.1,
            causal=True
        )
        self.n_vec = n_vec
        self.final = nn.Linear(n_vec*dim, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.reformer(x)
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x).squeeze(-2)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).cuda()
            x = self.posenc(positions) + x
            x = self.reformer(x)
            x = self.final(x.view(x.size(0), -1))
        return x


class LSTransformerHead(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem
     ):
        super(LSTransformerHead, self).__init__()
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.lstransformer = LongShortTransformer(
            num_tokens = dim,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=int(dim/heads),
            max_seq_len = n_vec,  # maximum sequence length
            window_size = 1,  # local attention window size
            r = 256,
            causal=False
        )
        self.n_vec = n_vec
        self.final = nn.Linear(n_vec * vocab_size, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            print(x.shape)
            x = x.long()
            x = self.lstransformer(x)
            print(x.shape)
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.lstransformer(x)
            x = self.final(x.view(x.size(0), -1))
        return x


class nystromFormerHead(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem
     ):
        super(nystromFormerHead, self).__init__()
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.nystromformer = Nystromer(
            dim=dim,
            dim_head=int(dim/heads),
            heads=heads,
            depth=depth,
            num_landmarks=256,  # number of landmarks
            pinv_iterations=6
        )
        self.n_vec = n_vec
        self.final = nn.Linear(n_vec * dim, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.nystromformer(x)
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x).squeeze(-2)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).cuda()
            x = self.dropout1(x)
            x = self.posenc(positions) + x
            x = self.nystromformer(x)
            x = self.final(x.view(x.size(0), -1))
        return x
