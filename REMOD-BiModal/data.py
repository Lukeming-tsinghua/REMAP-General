import os
import json
import pickle
import re
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from transformers import AutoTokenizer

import dgl
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from transformers import BertTokenizer


class graph_reader:
    node_file = "node.csv"
    relation_file = "relation.csv"
    triplet_file = "triplet.csv"
    def __init__(self, path, h=None):
        self.node_dict, self.type_cnt = self.load_node(os.path.join(path, self.node_file))
        self.relation_dict = self.load_relation(os.path.join(path, self.relation_file))
        self.triplet, self.relation_cnt = self.load_triplet(os.path.join(path, self.triplet_file))

        self.relation_num = len(self.relation_dict)

        self.g = dgl.heterograph(self.triplet, {'NODE': len(self.node_dict)})
        self.h = h

    def load_node(self, path):
        node = {}
        type_cnt = defaultdict(lambda: 0)
        with open(path) as f:
            line = f.readline()
            while line:
                idx, name, t = line.replace("\n", "").split("\t")
                node[(name, t)] = int(idx)
                type_cnt[t] += 1
                line = f.readline()
        return node, type_cnt

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
        rel_cnt = defaultdict(lambda: 0)
        with open(path) as f:
            line = f.readline()
            while line:
                h, typeh, r, t, typet = line.replace("\n", "").split("\t")
                triplet[("NODE", "_".join([typeh, r, typet]), "NODE")].append((self.node_dict[(h, typeh)], self.node_dict[(t, typet)]))
                rel_cnt[r] += 1
                line = f.readline()
        return triplet, rel_cnt


def bert_collate_func(arrays):
    sentences = {key: torch.cat([array[0][key] for array in arrays], 0) for key in arrays[0][0].keys()}
    split_points = np.cumsum([0] + [array[1] for array in arrays])
    labels = torch.LongTensor([array[2] for array in arrays])
    heads = torch.LongTensor([array[3] for array in arrays])
    tails = torch.LongTensor([array[4] for array in arrays])
    entity_1_begin_idxs = [array[5] for array in arrays]
    entity_2_begin_idxs = [array[6] for array in arrays]
    return (heads, tails, sentences, split_points, entity_1_begin_idxs, entity_2_begin_idxs),labels


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
        head = self.graph.node_dict[(sample["h"], sample["type_h"])]
        tail = self.graph.node_dict[(sample["t"], sample["type_t"])]
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
    path = "../../i2b2/2010i2b2/processed/"
    tokenizer = "allenai/scibert_scivocab_uncased"
    graph = graph_reader(path)
    print(graph.relation_num)
    dataset = BertEntityPairDataset(path=path,graph=graph,tokenizer=tokenizer, max_length=128, dataset="test")
    print(dataset[0])
