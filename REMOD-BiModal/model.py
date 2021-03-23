# -*- coding: utf-8 -*-
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertConfig, BertModel, BertPreTrainedModel
from rgcn import RGCN


LABEL_NUM = 4


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(nn.Linear(hidden_dim, 64),
                                        nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        energy = energy / math.sqrt(self.hidden_dim)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.sent_attention_1 = SelfAttention(768)
        self.sent_attention_2 = SelfAttention(768)
        self.bert_linear = nn.Linear(768, 100)

    def forward(self, tokens, split_points, entity_1_begin_idxs, entity_2_begin_idxs):
        bert_output = self.bert(**tokens)[0]
        ep_text_info_h = []
        ep_text_info_t = []
        for i in range(len(split_points) - 1):
            ep_text_tmp = bert_output[
                split_points[i]:split_points[i + 1]]
            ep_text_tmp_e1_begin = ep_text_tmp[entity_1_begin_idxs[i][0], entity_1_begin_idxs[i][1], :]
            ep_text_tmp_e2_begin = ep_text_tmp[entity_2_begin_idxs[i][0], entity_2_begin_idxs[i][1], :]
            ep_text_info_tmp_e1, _ = self.sent_attention_1(ep_text_tmp_e1_begin.unsqueeze(0))
            ep_text_info_tmp_e2, _ = self.sent_attention_2(ep_text_tmp_e2_begin.unsqueeze(0))
            ep_text_info_h.append(ep_text_info_tmp_e1)
            ep_text_info_t.append(ep_text_info_tmp_e2)
        ep_text_info_h = self.bert_linear(torch.cat(ep_text_info_h, 0))
        ep_text_info_t = self.bert_linear(torch.cat(ep_text_info_t, 0))
        return ep_text_info_h, ep_text_info_t

    def __repr__(self):
        return self.__class__.__name__


class GraphEncoderScore(nn.Module):
    def __init__(self, 
            g,
            feat_dim,
            layer_num,
            output_dim=None):
        super().__init__()

        self.g = g

        self.gnn = RGCN(g=g,
                        h_dim=feat_dim,
                        out_dim=100,
                        dropout=0.3,
                        num_hidden_layers=layer_num)

    def forward(self, subgraph, feature, head_ord, tail_ord):
        feature = self.gnn(feature, subgraph)
        heads = torch.vstack([feature[t][idx] for t, idx in head_ord])
        tails = torch.vstack([feature[t][idx] for t, idx in tail_ord])
        return heads, tails
    
    def __repr__(self):
        return self.__class__.__name__


class JointModel(nn.Module):
    def __init__(self, config, g, feat_dim, layer_num, output_dim=None):
        super().__init__()

        self.text_encoder = BertEncoder.from_pretrained(config)
        self.graph_encoder = GraphEncoderScore(g, feat_dim, layer_num, output_dim)

        self.relation_embedding = nn.Parameter(torch.Tensor(output_dim, 100))
        self.W_t = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (100, 100, 100)), dtype=torch.float, requires_grad=True))
        self.W_g = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (100, 100, 100)), dtype=torch.float, requires_grad=True))

        nn.init.xavier_uniform_(self.relation_embedding, gain=1)
        nn.init.xavier_uniform_(self.W_t, gain=1)
        nn.init.xavier_uniform_(self.W_g, gain=1)

        self.bn0 = torch.nn.BatchNorm1d(100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(100)

        self.input_dropout_1 = torch.nn.Dropout(0.5)
        self.input_dropout_2 = torch.nn.Dropout(0.5)
        self.hidden_dropout1 = torch.nn.Dropout(0.5)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

    def forward(self, subgraph, feature, heads_ord, tails_ord,
            tokens, split_points, entity_1_begin_idxs, entity_2_begin_idxs):
        h_text, t_text = self.text_encoder(tokens, split_points, entity_1_begin_idxs, entity_2_begin_idxs)
        h_graph, t_graph = self.graph_encoder(subgraph, feature, heads_ord, tails_ord)
        score_text = self.score_function(h_text, t_text, self.W_t)
        score_graph = self.score_function(h_graph, t_graph, self.W_g)
        return score_text, score_graph

    def score_function(self, h, t, W):
        h_exp = self.bn0(h)
        h_exp = self.input_dropout_1(h_exp)
        h_exp = h_exp.view(-1,1,1,h_exp.size(1))
        t_exp = self.bn2(t)
        t_exp = self.input_dropout_2(t_exp)
        t_exp = t_exp.view(-1,1,t_exp.size(1))

        W_mat = torch.mm(self.relation_embedding, W.view(self.relation_embedding.size(1), -1))
        W_mat = W_mat.view(-1, h.size(1), h.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        score = torch.matmul(h_exp, W_mat).squeeze()
        score = torch.bmm(score, t_exp.transpose(2,1)).squeeze()
        score = torch.sigmoid(score)
        return score

    def __repr__(self):
        return self.__class__.__name__
