# -*- coding: utf-8 -*-
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dgl
from rgcn import RGCN


class GraphEncoderScore(nn.Module):
    def __init__(self, 
            g,
            feat_dim,
            layer_num,
            output_dim=None):
        super().__init__()

        self.g = g

        self.relation_embedding = nn.Parameter(torch.Tensor(output_dim, 100))
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (100, 100, 100)), 
                                    dtype=torch.float, requires_grad=True))

        nn.init.xavier_uniform_(self.relation_embedding, gain=1)
        nn.init.xavier_uniform_(self.W, gain=1)

        self.bn0 = torch.nn.BatchNorm1d(100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(100)

        self.input_dropout_1 = torch.nn.Dropout(0.5)
        self.input_dropout_2 = torch.nn.Dropout(0.5)
        self.hidden_dropout1 = torch.nn.Dropout(0.5)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.gnn = RGCN(g=g,
                        h_dim=feat_dim,
                        out_dim=100,
                        dropout=0.3,
                        num_hidden_layers=layer_num)

    def forward(self, subgraph, feature, heads_ord, tails_ord):
        feature = self.gnn(feature, subgraph)
        heads = torch.vstack([feature[t][idx] for t, idx in heads_ord])
        tails = torch.vstack([feature[t][idx] for t, idx in tails_ord])
        epscore = self.score_function(heads, tails)
        return epscore

    def score_function(self, h, t):
        ## DistMult
        #score = torch.mm(torch.mul(h, t), self.relation_embedding.transpose(1,0))
        ## TransE
        #N = h.size(0)
        #R = self.relation_embedding.size(0)
        #h_exp = h.repeat_interleave(R, dim=0)
        #t_exp = t.repeat_interleave(R, dim=0)
        #r_exp = self.relation_embedding.repeat(N,1)
        #score = -torch.norm(h_exp+r_exp-t_exp, p=2, dim=1).view(-1, R)
        ## TuckER
        h_exp = self.bn0(h)
        h_exp = self.input_dropout_1(h_exp)
        h_exp = h_exp.view(-1,1,1,h_exp.size(1))
        t_exp = self.bn2(t)
        t_exp = self.input_dropout_2(t_exp)
        t_exp = t_exp.view(-1,1,t_exp.size(1))

        W_mat = torch.mm(self.relation_embedding, self.W.view(self.relation_embedding.size(1), -1))
        W_mat = W_mat.view(-1, h.size(1), h.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        score = torch.matmul(h_exp, W_mat).squeeze()
        score = torch.bmm(score, t_exp.transpose(2,1)).squeeze()
        score = F.sigmoid(score)
        return score
    
    def __repr__(self):
        return self.__class__.__name__


class GraphEncoderScoreTuckER(nn.Module):
    def __init__(self, 
            g,
            node_num,
            feat_dim,
            meta_paths,
            output_dim=None):
        super().__init__()

        self.g = g
        self.h = torch.nn.Parameter(torch.Tensor(node_num, feat_dim))

        self.relation_embedding = nn.Parameter(torch.Tensor(output_dim, 100))
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (100, 100, 100)), 
                                    dtype=torch.float, requires_grad=True))

        nn.init.xavier_uniform_(self.h, gain=1)
        nn.init.xavier_uniform_(self.relation_embedding, gain=1)
        nn.init.xavier_uniform_(self.W, gain=1)

        self.bn0 = torch.nn.BatchNorm1d(100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(100)

        self.input_dropout_1 = torch.nn.Dropout(0.5)
        self.input_dropout_2 = torch.nn.Dropout(0.5)
        self.hidden_dropout1 = torch.nn.Dropout(0.5)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)
        

    def forward(self, head, tail):
        epscore = self.score_function(self.h[head,], self.h[tail,])
        return epscore

    def score_function(self, h, t):
        ## DistMult
        #score = torch.mm(torch.mul(h, t), self.relation_embedding.transpose(1,0))
        ## TransE
        #N = h.size(0)
        #R = self.relation_embedding.size(0)
        #h_exp = h.repeat_interleave(R, dim=0)
        #t_exp = t.repeat_interleave(R, dim=0)
        #r_exp = self.relation_embedding.repeat(N,1)
        #score = -torch.norm(h_exp+r_exp-t_exp, p=2, dim=1).view(-1, R)
        ## TuckER
        h_exp = self.bn0(h)
        h_exp = self.input_dropout_1(h_exp)
        h_exp = h_exp.view(-1,1,1,h_exp.size(1))
        t_exp = self.bn2(t)
        t_exp = self.input_dropout_2(t_exp)
        t_exp = t_exp.view(-1,1,t_exp.size(1))

        W_mat = torch.mm(self.relation_embedding, self.W.view(self.relation_embedding.size(1), -1))
        W_mat = W_mat.view(-1, h.size(1), h.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        score = torch.matmul(h_exp, W_mat).squeeze()
        score = torch.bmm(score, t_exp.transpose(2,1)).squeeze()
        score = F.sigmoid(score)
        return score
    
    def __repr__(self):
        return self.__class__.__name__
