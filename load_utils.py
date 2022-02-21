import pickle
import networkx as nx
import os
import numpy as np
import re
import torch


def pad_to_max_seq_len(arr, max_seq_len=None, pad_token_id=0, max_len=None):
    """
    a = [ [1, 2, 3], [1, 3] ]
    pad_to_max_seq_len(a, 5)
    a -> [[1, 2, 3, 0, 0], [1, 3, 0, 0, 0]]
    """
    if max_seq_len is None:
        max_seq_len = 0
        for sub_a in arr:
            if len(sub_a) >= max_seq_len:
                max_seq_len = len(sub_a)
    if max_len is not None:
        if max_seq_len > max_len:
            max_seq_len = max_len
    for index, text in enumerate(arr):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [
                pad_token_id for _ in range(max_seq_len - seq_len)
            ]
            new_text = text + padded_tokens
            arr[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            arr[index] = new_text
    return max_seq_len


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


class GraphTokenizer():
    def __init__(self, dialog_graph):
        self.dialog_graph = dialog_graph
        # self.node2idx = {'[PAD]': 0, '[EOS]': 1, '[BOS]': 2}
        # self.idx2node = {0: '[SEP]', 1: '[EOS]', 2: '[BOS]'}
        self.node2idx = {}
        self.idx2node = {}
        for idx, node in enumerate(self.dialog_graph.nodes):
            self.node2idx[node] = idx
            self.idx2node[idx] = node
        self.vocab_size = len(self.dialog_graph.nodes)

    def get_weight(self, graph_embed):
        pretrain_weight = []
        for idx in range(0, self.vocab_size):
            pretrain_weight.append(graph_embed[self.idx2node[idx]])
        return pretrain_weight

    def encode(self, text, sep=' ', return_tensors=None):
        ids = []
        for t in text.split(sep):
            if t in self.node2idx:
                ids.append(self.node2idx[t])
        if return_tensors == 'pt':
            return torch.tensor(ids)
        return ids

    def encode_tokens(self, tokens, return_tensors=None):
        ids = []
        for t in tokens:
            if t in self.node2idx:
                ids.append(self.node2idx[t])
        if return_tensors == 'pt':
            return torch.tensor(ids)
        return ids

    # def decode(self, ids):
    #     return ' '.join([self.idx2node[i] for i in ids])

    def word_in_graph(self, word):
        return word in self.node2idx


def load_concept_net_zh():
    if os.path.exists('data_graph/concept_net_zh.pkl'):
        with open('data_graph/concept_net_zh.pkl', 'rb') as f:
            graph = pickle.load(f)
        print('load pkl')
    else:
        graph = nx.Graph()
        with open('data_raw/concept_net_zh.csv', 'r') as f:
            for line in f:
                row = line.rstrip('\n').split(',')
                if '_' in row[0] or '_' in row[1]:
                    continue
                if row[0] == row[1]:
                    continue
                graph.add_edge(row[0], row[1])
                # graph.add_edge(row[0], row[1], relation=row[1])
        with open('data_graph/concept_net_zh.pkl', 'wb') as f:
            pickle.dump(graph, f)
    return graph


def load_concept_net(graph_embed=None):
    if os.path.exists('data/conceptnet/concept_net.pkl'):
        with open('data/conceptnet/concept_net.pkl', 'rb') as f:
            graph = pickle.load(f)
        print('load_concept_net')
    else:
        if graph_embed is None:
            graph_embed = load_graph_embedding()
        graph = nx.Graph()
        with open('data/conceptnet/concept_net_en.csv', 'r') as f:
            for line in f:
                row = line.rstrip('\n').split(' ')
                if len(row[0]) == 1 or len(row[2]) == 1:
                    continue
                if '_' in row[0] or '_' in row[2]:
                    continue
                if row[0].isalpha() is False or row[2].isalpha() is False:
                    continue
                if row[0] not in graph_embed or row[2] not in graph_embed:
                    continue
                graph.add_edge(row[0], row[2], relation=row[1])
        with open('data/conceptnet/concept_net.pkl', 'wb') as f:
            pickle.dump(graph, f)
    return graph


def load_dialog_graph_zh():
    if os.path.exists('data_graph/dialog_graph_zh.pkl'):
        with open('data_graph/dialog_graph_zh.pkl', 'rb') as f:
            graph = pickle.load(f)
        print('load pkl')
    else:
        graph = nx.Graph()
        with open('data_graph/dialog_graph_zh.csv', 'r') as f:
            for line in f:
                row = line.rstrip('\n').split(',')
                graph.add_edge(row[0], row[1])
        with open('data_graph/dialog_graph_zh.pkl', 'wb') as f:
            pickle.dump(graph, f)
    return graph


def load_dialog_graph(graph_embed=None):
    if os.path.exists('data_graph/dialog_graph.pkl'):
        with open('data_graph/dialog_graph.pkl', 'rb') as f:
            graph = pickle.load(f)
        print('load pkl')
    else:
        if graph_embed is None:
            graph_embed = load_graph_embedding()
        graph = nx.Graph()
        with open('data_graph/dialog_graph.csv', 'r') as f:
            for line in f:
                row = line.rstrip('\n').split(',')
                if row[0] not in graph_embed or row[1] not in graph_embed:
                    continue
                graph.add_edge(row[0], row[1], relation=row[2])
        with open('data_graph/dialog_graph.pkl', 'wb') as f:
            pickle.dump(graph, f)
    return graph


def load_di_concept_net(graph_embed=None):
    if os.path.exists('data_graph/concept_net.pkl'):
        with open('data_graph/concept_net.pkl', 'rb') as f:
            graph = pickle.load(f)
        print('load pkl')
    else:
        graph = nx.DiGraph()
        with open('data_raw/concept_net_en.csv', 'r') as f:
            for line in f:
                row = line.rstrip('\n').split(' ')
                if row[0].isalpha() is False or row[2].isalpha() is False:
                    continue
                if row[0] not in graph_embed or row[2] not in graph_embed:
                    continue
                graph.add_edge(row[0], row[2], relation=row[1])
                if row[1] in ['RelatedTo', 'Synonym', 'Antonym', 'DistinctFrom', 'SimilarTo', 'EtymologicallyRelatedTo']:
                    graph.add_edge(row[2], row[0], relation=row[1])
        with open('data_graph/concept_net.pkl', 'wb') as f:
            pickle.dump(graph, f)
    return graph


def load_graph_embedding():
    if os.path.exists('data/conceptnet/graph_embedding.pkl'):
        with open('data/conceptnet/graph_embedding.pkl', 'rb') as f:
            graph_embedding = pickle.load(f)
        print('load_graph_embedding')
    else:
        graph_embedding = {}
        with open('data/conceptnet/numberbatch-en-17.06.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n').split(' ')
                if len(line) != 301:
                    continue
                word = line[0]
                # if word.isalpha() is False:
                #     continue
                graph_embedding[word] = np.array(line[1:], dtype=np.float64)
        with open('data/conceptnet/graph_embedding.pkl', 'wb') as f:
            pickle.dump(graph_embedding, f)
    return graph_embedding
