# -*- coding: utf-8 -*-
import json
import networkx as nx
from tqdm import tqdm
from functools import partial
from load_utils import GraphTokenizer, load_concept_net
from collections import Counter
import multiprocessing
import random


def clip_and_attach_target(ns, start_idx):
    print("clipping and attach target for every conversation...")
    batch_data = ns.all_data[start_idx: start_idx + ns.batch_size]
    dialog_graph = ns.dialog_graph
    graph_tok = ns.graph_tok

    def add_edges(source, target, sub_graph):
        ans = []
        for s in source:
            for t in target:
                if s == t:
                    continue
                if s not in dialog_graph.nodes or t not in dialog_graph.nodes:
                    continue
                sub_graph.add_node(s)
                sub_graph.add_node(t)
                res = list(nx.all_simple_edge_paths(dialog_graph, source=s, target=t, cutoff=2))
                ans.extend(res)
                [sub_graph.add_edge(pair[0], pair[1]) for path in res for pair in path]
        return ans

    good_case = []
    bad_case = []
    for dialog, kws, nn_kws in tqdm(batch_data, mininterval=30):
        for i in range(0, len(kws) - 3):
            if len([k for k in kws[i:i+3] if len(k) > 0]) != 3:
                continue
            sub_graph = nx.Graph()

            add_edges(kws[i], kws[i], sub_graph)
            add_edges(kws[i+1], kws[i+1], sub_graph)
            add_edges(kws[i], kws[i+1], sub_graph)

            res1 = add_edges(kws[i], kws[i+1], sub_graph)
            res2 = add_edges(kws[i+1], kws[i+2], sub_graph)
            if len(sub_graph) == 0 or len(res1) == 0 or len(res2) == 0:
                continue
            start_word = kws[i] + kws[i+1]
            t_paths = {t: [] for t in kws[i+2]}
            for s in start_word:
                for t in kws[i+2]:
                    if s in sub_graph and t in sub_graph and nx.has_path(sub_graph, s, t):
                        path = nx.shortest_path(sub_graph, s, t)
                        if len(path) > 4:
                            continue
                        t_paths[t].append(path)

            max_in_degree = max([len(path) for path in t_paths.values()])

            target, path = random.choice([(t, p) for t, p in t_paths.items() if len(p) >= max_in_degree])

            if max_in_degree > 2:
                # ans = []
                # for p in path:
                #     ans.extend([p[0] + '|' + n for n in p[1:-1]])
                good_case.append((dialog[i:i+3], kws[i:i+3], target, path))
            else:
                bad_case.append((dialog[i:i+3], kws[i:i+3]))
        # if nx.is_connected(sub_graph):
        #     res1.extend(res2)
        #     counter_list = [path[-1][-1] for path in res1]
        #     if len(counter_list) == 0:
        #         continue
        #     target = Counter(counter_list).most_common(1)[0][0]
        #     ans = set()
        #     for path in res1:
        #         if path[-1][-1] == target:
        #             ans.add(path[0][0] + '|' + path[0][1])
        #     if len(ans) <= 5:
        #         continue
        #     good_case.append((dialog[i:i+3], kws[i:i+3], target, list(ans)))
        # else:
        #     bad_case.append((dialog[i:i+3], kws[i:i+3]))
    return [good_case, bad_case]


def main(file1, file2, file3, file4):
    with open(file1) as f:
        dialogs = [json.loads(row)['session'] for row in f]
    with open(file2) as f:
        keywords = [json.loads(row)['all_kw'] for row in f]
    with open(file2) as f:
        nn_keywords = [json.loads(row)['nn_kw'] for row in f]  # nn_kw

    all_data = [(d, k, n) for d, k, n in zip(dialogs, keywords, nn_keywords)]
    concept_net = load_concept_net()
    graph_tok = GraphTokenizer(concept_net)
    ns = multiprocessing.Manager().Namespace()
    ns.all_data = all_data
    ns.batch_size = int(len(ns.all_data) / 8) + 10
    mylist = [i for i in range(0, len(ns.all_data), ns.batch_size)]  # valid
    ns.dialog_graph = concept_net
    ns.graph_tok = graph_tok
    p = multiprocessing.Pool()
    func = partial(clip_and_attach_target, ns)
    result = p.map(func, mylist)
    good_cases = []
    bad_cases = []
    for p_result in result:
        good_cases.extend(p_result[0])
        bad_cases.extend(p_result[1])

    print(file3, len(good_cases))
    fout = open(file3, 'w')
    for context, kws, target, ans in good_cases:
        fout.write(json.dumps({
            'context': context,
            'kws': kws,
            'target': target,
            'ans': ans,
        }) + '\n')
    fout.close()

    print(file4, len(bad_cases))
    fout = open(file4, 'w')
    for context, kws in bad_cases:
        fout.write(json.dumps({
            'context': context,
            'kws': kws,
        }) + '\n')
    fout.close()


if __name__ == "__main__":
    main(
        'data/process/convai/test_raw.json',
        'data/process/convai/test_keywords.json',
        'data/process/convai/test_good_case.json',
        'data/process/convai/test_bad_case.json',
    )
    main(
        'data/process/convai/train_raw.json',
        'data/process/convai/train_keywords.json',
        'data/process/convai/train_good_case.json',
        'data/process/convai/train_bad_case.json',
    )

    # main(
    #     'data/process/daily_dialogue/test_raw.json',
    #     'data/process/daily_dialogue/test_keywords.json',
    #     'data/process/daily_dialogue/test_good_case.json',
    #     'data/process/daily_dialogue/test_bad_case.json',
    # )

    # main(
    #     'data/process/daily_dialogue/train_raw.json',
    #     'data/process/daily_dialogue/train_keywords.json',
    #     'data/process/daily_dialogue/train_good_case.json',
    #     'data/process/daily_dialogue/train_bad_case.json',
    # )


# nohup python -u 2_sampleing_by_reasoning.py  > common.log 2>&1 &
