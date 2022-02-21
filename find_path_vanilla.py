import pickle
from networkx.algorithms.shortest_paths import weighted
import torch
import networkx as nx
import os
import numpy as np
import re
from load_utils import GraphTokenizer, load_concept_net, get_cos_similar_multi, load_dialog_graph, load_graph_embedding


class FindPathModel():
    def __init__(self, MAX_STEP=3, test=False):
        self.graph_embed = load_graph_embedding()  # len(graph_embedding.keys()) 286002
        if test == True:
            self.graph = load_dialog_graph(self.graph_embed)  # len(graph.nodes) 256170
            self.graph_tok = GraphTokenizer(self.graph)
        else:
            self.graph = load_concept_net(self.graph_embed)  # len(graph.nodes) 256170
        # list(nx.connected_components(self.graph))
        self.MAX_STEP = MAX_STEP
        self.target_embed = None
        self.start_embed = None

    def getOneHopWords(self, word, target):
        words_list = []
        neighbors_word = []
        neighbors_embed = []
        target_embed = self.graph_embed[target]
        if word not in self.graph:
            return []
        for w in list(self.graph.neighbors(word)):
            if w in self.graph_embed:
                if self.graph_embed[w].sum() == 0:
                    continue
                neighbors_word.append(w)
                neighbors_embed.append(self.graph_embed[w])
        if len(neighbors_embed) == 0:
            return []
        combile_s = get_cos_similar_multi(target_embed, neighbors_embed).reshape(-1)
        words_list = [(w, round(cos, 4)) for w, cos in zip(neighbors_word, combile_s) if cos > 0]
        # 按相似度由大到小排序
        sorted_word_list = sorted(words_list, key=lambda pair: pair[1], reverse=True)
        return sorted_word_list[:3]

    def getWords(self, word, cur_path):
        words_list = []
        neighbors_word = []
        neighbors_embed = []
        if word not in self.graph:
            return []
        for w in list(self.graph.neighbors(word)):
            if w in self.graph_embed and w not in cur_path:
                if self.graph_embed[w].sum() == 0:
                    continue
                neighbors_word.append(w)
                neighbors_embed.append(self.graph_embed[w])
        if len(neighbors_embed) == 0:
            return []
        smooth_score = get_cos_similar_multi(self.start_embed, neighbors_embed).reshape(-1)
        forward_score = get_cos_similar_multi(self.target_embed, neighbors_embed).reshape(-1)
        combile_s = smooth_score * 0.2 + forward_score * 0.8
        words_list = [(w, round(cos, 4)) for w, cos in zip(neighbors_word, combile_s) if cos > 0]
        # 按相似度由大到小排序
        sorted_word_list = sorted(words_list, key=lambda pair: pair[1], reverse=True)
        for pair in sorted_word_list[:20]:
            # relation=self.graph[word][w[0]]['relation']
            self.sub_g.add_edge(word, pair[0], weight=(1 - pair[1]))
        return sorted_word_list[:20]

    def tree_search(self, word, target_word, cur_path, edges):
        if word and (len(cur_path) >= self.MAX_STEP or word == target_word):
            path_vector = ""
            for v in cur_path:
                path_vector = path_vector + v + " "
            edges.append(path_vector.rstrip(' '))
            return edges

        new_choices = self.getWords(word, cur_path)

        if len(new_choices) == 0:
            path_vector = ""
            for v in cur_path:
                path_vector = path_vector + v + " "
            edges.append(path_vector.rstrip(' '))

        for new_word in new_choices:
            # new_word:[word,cos]
            if new_word[0] in cur_path:
                continue
            cur_path.append(new_word[0])
            self.tree_search(new_word[0], target_word, cur_path, edges)
            cur_path.pop()
        return edges

    def find_path(self, start_word, end_word):
        if start_word not in self.graph_embed or end_word not in self.graph_embed:
            return []
        self.sub_g = nx.Graph()
        cur_path = [start_word]
        self.target_embed = self.graph_embed[end_word]
        self.start_embed = self.graph_embed[start_word]
        res_paths = []
        edges = self.tree_search(start_word, end_word, cur_path, [])
        res_paths.extend(edges)

        cur_path = [end_word]
        edges = self.tree_search(end_word, start_word, cur_path, [])
        res_paths.extend(edges)
        try:
            res = list(nx.all_simple_edge_paths(self.sub_g, source=start_word, target=end_word, cutoff=2))
        except:
            print("can not find ", start_word, end_word)
            res = []
        return res[:10]

    def get_node_by_keyword(self, start_words, end_word, keyword=None):
        words = [w for row in start_words for w in row]
        self.sub_g = nx.Graph()
        res_paths = []
        nodes = []
        for start_word in words:
            if start_word not in self.graph_embed or end_word not in self.graph_embed:
                continue
            cur_path = [start_word]
            self.target_embed = self.graph_embed[end_word]
            self.start_embed = self.graph_embed[start_word]
            edges = self.tree_search(start_word, end_word, cur_path, [])
            res_paths.extend(edges)
            cur_path = [end_word]
            edges = self.tree_search(end_word, start_word, cur_path, [])
            res_paths.extend(edges)
            try:
                # nx.shortest_path(G, source, target)
                _, path_nodes = nx.single_source_dijkstra(self.sub_g, start_word, end_word)
                nodes.extend(path_nodes[1:])
            except:
                continue
        if len(nodes) == 0:
            nodes.extend(list(self.sub_g.nodes))
            nodes.append(end_word)
        assert len(nodes) > 0
        return [n for n in nodes if n in self.graph_embed]
        # words = []
        # embeds = []
        # for n in self.sub_g:
        #     if n == start_word:
        #         continue
        #     words.append(n)
        #     embeds.append(self.graph_embed[n])
        # faith_score = get_cos_similar_multi(self.graph_embed[keyword], embeds).reshape(-1)
        # words_list = [(w, round(cos, 4)) for w, cos in zip(words, faith_score)]
        # words_list = sorted(words_list, key=lambda pair: pair[1], reverse=True)
        # return words_list

    def find_path_node_set(self, start_word, end_word):
        if start_word not in self.graph_embed or end_word not in self.graph_embed:
            return []
        if start_word in end_word or end_word in start_word:
            return []
        self.sub_g = nx.Graph()
        cur_path = [start_word]
        self.target_embed = self.graph_embed[end_word]
        self.start_embed = self.graph_embed[start_word]
        res_paths = []
        edges = self.tree_search(start_word, end_word, cur_path, [])
        res_paths.extend(edges)
        cur_path = [end_word]
        edges = self.tree_search(end_word, start_word, cur_path, [])
        res_paths.extend(edges)
        try:
            res = list(nx.all_simple_paths(self.sub_g, source=start_word, target=end_word, cutoff=2))
            candidate_set = set([w for path in res for w in path])  # 候选词
            all_nodes = set(self.sub_g.nodes)
        except:
            print("can not find ", start_word, end_word)
            return None, None
        return candidate_set, all_nodes


if __name__ == '__main__':
    test = FindPathModel(test=False)
    # res = test.find_path('bick', 'bien')
    # print(res)
    res = test.get_node_by_keyword([['favorite', 'type', 'south'], ['moonlight', 'kind', 'dancing', 'music']], 'adventure')
    print(res)
    res = test.find_path('job', 'basketball')
    print(res)
    res = test.find_path('dance', 'computer')
    print(res)
