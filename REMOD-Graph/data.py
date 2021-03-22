import os
import json
import pickle
import re
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from transformers import AutoTokenizer

import dgl
import dgl.nn as dglnn
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler 
from transformers import BertTokenizer

from hetero_model import HeteroGAT
from rgcn import RGCN


class graph_reader:
    node_file = "node.csv"
    relation_file = "relation.csv"
    triplet_file = "triplet.csv"
    def __init__(self, path, feat_dim=100, h=None, layer_num=1):
        self.node_dict = self.load_node(os.path.join(path, self.node_file))
        self.type_cnt = {key: len(self.node_dict[key]) for key in self.node_dict}
        self.relation_dict = self.load_relation(os.path.join(path, self.relation_file))
        self.triplet = self.load_triplet(os.path.join(path, self.triplet_file))

        self.relation_num = len(self.relation_dict)

        self.feat_dim = feat_dim

        self.g = dgl.heterograph(self.triplet, self.type_cnt)

        for key in self.type_cnt:
            self.g.nodes[key].data["h"] = torch.rand(self.type_cnt[key], self.feat_dim)

        self.layer_num = layer_num

    def sample_subgraph(self, subnodes):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.layer_num)
        subgraph = sampler.sample_blocks(self.g, subnodes)
        feature = subgraph[0].srcdata["h"]
        return subgraph, feature

    def load_node(self, path):
        node = defaultdict(list)
        with open(path) as f:
            line = f.readline()
            while line:
                idx, name, t = line.replace("\n", "").split("\t")
                if name not in node[t]:
                    node[t].append(name)
                line = f.readline()
        return node

    def load_relation(self, path):
        relation = {}
        with open(path) as f:
            line = f.readline()
            while line:
                idx, name = line.replace("\n", "").split("\t")
                relation[name] = int(idx)
                line = f.readline()
        return relation

    def load_triplet(self, path):
        triplet = defaultdict(list)
        with open(path) as f:
            line = f.readline()
            while line:
                h, typeh, r, t, typet = line.replace("\n", "").split("\t")
                triplet[(typeh, r, typet)].append((self.node_dict[typeh].index(h), self.node_dict[typet].index(t)))
                line = f.readline()
        return triplet


def grap_dict(heads, tails):
    sub_dict = {}
    head_ord = []
    tail_ord = []
    for t, idx in heads:
        if t not in sub_dict:
            sub_dict[t] = []
        if idx not in sub_dict[t]:
            sub_dict[t].append(idx)
            head_ord.append((t, len(sub_dict[t])-1))
        else:
            head_ord.append((t, sub_dict[t].index(idx)))
    for t, idx in tails:
        if t not in sub_dict:
            sub_dict[t] = []
        if idx not in sub_dict[t]:
            sub_dict[t].append(idx)
            tail_ord.append((t, len(sub_dict[t])-1))
        else:
            tail_ord.append((t, sub_dict[t].index(idx)))
    return sub_dict, head_ord, tail_ord


def merge_nodes(*nodes):
    merge = {}
    for node in nodes:
        for each in node:
            if each in merge:
                merge[each] += node[each]
            else:
                merge[each] = node[each]
    return merge


def bert_collate_func(arrays):
    sentences = {key: torch.cat([array[0][key] for array in arrays], 0) for key in arrays[0][0].keys()}
    split_points = np.cumsum([0] + [array[1] for array in arrays])
    labels = torch.LongTensor([array[2] for array in arrays])
    subnodes, heads_ord, tails_ord = grap_dict([array[3] for array in arrays], [array[4] for array in arrays])
    entity_1_begin_idxs = [array[5] for array in arrays]
    entity_2_begin_idxs = [array[6] for array in arrays]
    return (subnodes, heads_ord, tails_ord,  
            sentences, split_points, entity_1_begin_idxs, entity_2_begin_idxs),labels


class BertEntityPairDataset(Dataset):
    def __init__(self, 
            path = None,
            dataset = "train",
            graph = None,
            max_length=None,
            tokenizer=None):

        with open(os.path.join(path, dataset+".json")) as f:
            self.data = [json.loads(line) for line in f.readlines()]

        self.graph = graph
        types = list(self.graph.type_cnt.keys())
        entity_1_begin_tokens = ["<entity-%s-1>" % t for t in types]
        entity_2_begin_tokens = ["<entity-%s-2>" % t for t in types]
        entity_1_end_tokens = ["<entity-%s-1/>" % t for t in types]
        entity_2_end_tokens = ["<entity-%s-2/>" % t for t in types]
        special_tokens_dict = {
                'additional_special_tokens':
                    entity_1_begin_tokens+
                    entity_1_end_tokens+
                    entity_2_begin_tokens+
                    entity_2_end_tokens
                    }

        assert tokenizer is not None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.entity_1_begin_ids = self.tokenizer.convert_tokens_to_ids(entity_1_begin_tokens)
        self.entity_2_begin_ids = self.tokenizer.convert_tokens_to_ids(entity_2_begin_tokens)

        self.max_length = max_length

    def cal_begin_idxs(self, input_ids):
        entity_1_begin_idxs = None
        entity_2_begin_idxs = None
        for entity_1_begin_id in self.entity_1_begin_ids:
            idxs = torch.where(input_ids==entity_1_begin_id)
            if len(idxs[0]) > 0:
                entity_1_begin_idxs = idxs
        for entity_2_begin_id in self.entity_2_begin_ids:
            idxs = torch.where(input_ids==entity_2_begin_id)
            if len(idxs[0]) > 0:
                entity_2_begin_idxs = idxs
        return entity_1_begin_idxs, entity_2_begin_idxs

    def __getitem__(self, index):
        sample = self.data[index]
        head = (sample["type_h"], self.graph.node_dict[sample["type_h"]].index(sample["h"]))
        tail = (sample["type_t"], self.graph.node_dict[sample["type_t"]].index(sample["t"]))
        text = sample["sentences"]
        text_tokenized = self.tokenizer.batch_encode_plus(
                text,
                add_special_token=True,
                max_length=self.max_length,
                pad_to_max_length=True,return_tensors="pt")
        entity_1_begin_idxs, entity_2_begin_idxs = self.cal_begin_idxs(text_tokenized["input_ids"])
        num = len(text)
        label = self.graph.relation_dict[sample["r"]]
        #if entity_1_begin_idxs is None or entity_2_begin_idxs is None:
        #    print(text)
        #    print(text_tokenized["input_ids"], text_tokenized["input_ids"].size())
        #    raise RuntimeError
        return (text_tokenized, num, label, head, 
                tail, entity_1_begin_idxs, entity_2_begin_idxs)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    path = "/media/sdb1/Yucong/Dataset/i2b2/2010i2b2/processed_clean/"
    tokenizer = "allenai/scibert_scivocab_uncased"
    graph = graph_reader(path)
    dataset = BertEntityPairDataset(path=path,graph=graph,tokenizer=tokenizer, max_length=128, dataset="train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=bert_collate_func)
    g = graph.g

    layer_num = 3
    for (subnodes, heads_ord, tails_ord, *others), label in loader:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(layer_num)
        subgraph = sampler.sample_blocks(g, subnodes)
        model = RGCN(g=g, 
                h_dim=10,
                out_dim=10,
                dropout=0.3,
                num_hidden_layers=layer_num,
                use_self_loop=True)
        input_feature = subgraph[0].srcdata["h"]
        output = model(input_feature, subgraph)
        heads = torch.vstack([output[t][idx] for t, idx in heads_ord])
        tails = torch.vstack([output[t][idx] for t, idx in tails_ord])
        print(output, heads, tails)
        break
