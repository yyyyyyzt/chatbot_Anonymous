import multiprocessing
import os
import time
from nltk.corpus import stopwords
from functools import partial
from tqdm import tqdm, trange
import numpy as np
import spacy  # python -m spacy download en_core_web_sm
from collections import Counter
import yake  # pip install git+https://github.com/LIAAD/yake
import json
from textblob import Word, TextBlob
import string


def handleFile(filename, target_name):
    fw = open(target_name, 'w')
    with open(filename) as f:
        dialog_id = 0
        session = []
        for line in f:
            turn_num = int(line[:2])
            if turn_num == 1 and len(session) > 1:
                dialog_id += 1
                fw.write(json.dumps({
                    'dialog_id': dialog_id,
                    'session': session,
                }) + '\n')
                session = []
            line = line[2:] if turn_num < 10 else line[3:]
            pair = line.rstrip('\n').split('\t')
            if '__SILENCE__' not in pair:
                session.append(pair[0])
            session.append(pair[1])
    dialog_id += 1
    fw.write(json.dumps({
        'dialog_id': dialog_id,
        'session': session,
    }) + '\n')
    print('total dialog_session', dialog_id)


def stem_uttr(utterance):
    lem = []
    for w in utterance.split():
        word1 = Word(w).lemmatize("n")
        word2 = Word(word1).lemmatize("v")
        word3 = Word(word2).lemmatize("a")
        lem.append(Word(word3).lemmatize())
    return ' '.join(lem)


def handleData(ns, start_idx):
    print("Worker process id for ", os.getpid())
    print('start_idx ', start_idx)
    # time.sleep(1)
    batch_data = ns.docs[start_idx: start_idx + ns.batch_size]
    if len(batch_data) == 0:
        return []
    idf_dict = ns.idf_dict
    TOPK = 3
    data = []
    kw_extractor = yake.KeywordExtractor(top=4, n=2, stopwords=ns.stopword_set)

    def get_keywords_from_uttr(utterance, exclude_words=[]):
        utterance = utterance.lstrip('. ')
        all_set = set()
        nn_set = set()
        yake_keywords = kw_extractor.extract_keywords(utterance)
        try:
            blob = TextBlob("".join([i for i in utterance if i not in string.punctuation]))
            nn_words = [pair[0] for pair in blob.tags if 'NN' in pair[1]]
        except:
            print('utterance ', utterance)  # error case: wow ! ! ! that's a lot lol
            raise RuntimeError('TextBlob Error')

        for pair_k in yake_keywords:
            for k in pair_k[0].split(' '):
                if k not in exclude_words:
                    all_set.add(k)
                    if k in nn_words:
                        nn_set.add(k)
        return list(nn_set), list(all_set)

    for dialog in tqdm(batch_data, desc='pid:' + str(os.getpid())):
        keywords_list1 = []
        keywords_list2 = []
        exclude_words = [
            'everyday', 'today', 'yesterday', 'tomorrow', 'year',
            'morning', 'afternoon', 'evening', 'tonight', 'week', 'great', 'good', 'nice',
            'hey', 'ready', 'lot', 'awesome',
        ]
        for line in dialog['session']:
            if '__SILENCE__' in line:
                continue
            # tokens = line.split()
            # tf = Counter(tokens)  # [ 'a', 'b' ,'c', 'a', 'd' ] { a: 2, b: 1 }
            # tokens_len = float(len(tokens))
            # prob_list = np.array([idf_dict[t] * tf[t] / tokens_len for t in tokens])
            nn_set, all_set = get_keywords_from_uttr(line, exclude_words)
            # for i, t in enumerate(tokens):
            #     if t in yake_keywords:
            #         prob_list[i] *= 2
            # # [ (a, b) for a, b in zip(tokens, prob_list) ] # 查看所有词频率
            # topk_idx = np.argsort(prob_list)[-TOPK:]
            # topk_keywords = np.array(tokens)[topk_idx].tolist()
            keywords_list1.append(nn_set)
            keywords_list2.append(all_set)
        data.append({'dialog_id': dialog['dialog_id'], 'nn_kw': keywords_list1, 'all_kw': keywords_list2})
        # data.append({ 'dialog_id': dialog['dialog_id'], 'tf_kw': tf_keywords_list, 'yake_kw': yake_keywords_list })
    return data


def cal_idf_dict(docs):
    idf_dict = {}
    for line in docs:
        tokens = line.split()
        for t in set(tokens):
            idf_dict[t] = idf_dict.get(t, 0) + 1  # 统计得到词频

    # get idf
    docs_len = len(docs)
    for t in idf_dict.keys():
        idf_dict[t] = np.log(docs_len / idf_dict[t])
    for k in idf_dict.keys():
        idf_dict[k] = 1.0/(idf_dict[k] + 1e-5)

    # reduce stop word score
    stop_words = set(stopwords.words('english')) | set(['[SEP]', '[PAD]', '[CLS]', 'the', 'of', 'and', 'in', 'a', 'to', 'was', 'is', '"', 'for', 'on',
                                                        'as', 'with', 'by', 'he', "'s", 'at', 'that',   'from', 'it', 'his', 'an', 'which', 's', '.', '?', '!', ',', '(', ')', "'", '%'])
    for t in stop_words:
        if t in idf_dict:
            idf_dict[t] = 0.01/(idf_dict[t])

    # remove number
    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)
    inp = " ".join([k for k in idf_dict.keys() if not hasNumbers(k)])

    # calc token tag
    spacy_nlp = spacy.load('en_core_web_sm')
    inp_results = [(token.text, token.tag_) for token in spacy_nlp(inp)]

    # increase noun and verb score
    allowed_tags = ['VB', 'NN', 'JJ', 'RB']   # UH for "yes", "no", etc.
    ignored_words = ['was', 'were', 'be', 'is', 'are', 'am', "'s", "'re"] + ['do', 'did', 'done', 'does'] + ['[', ']', 'CLS', 'SEP']  # verb of no info
    for word, tag in inp_results:
        if word in idf_dict.keys():
            if len(tag) >= 2 and tag[:2] in allowed_tags and (word not in ignored_words):
                if tag[:2] in ['VB', 'NN']:
                    idf_dict[word] *= 4
                else:
                    idf_dict[word] *= 2
    return idf_dict


def main(filename, targetfile):
    with open(filename) as f:
        dialogs = [json.loads(row) for row in f]

    # flatten_dialogs = [' '.join(dialog['session']) for dialog in dialogs]
    # idf_dict = cal_idf_dict(flatten_dialogs)
    idf_dict = {}

    with open('data/stopwords.txt', encoding='utf-8') as stop_fil:
        stopword_set = set(stop_fil.read().lower().split("\n"))

    # mylist = [ 0, 1000, 1000 ]
    mylist = [i for i in range(0, len(dialogs), 2000)]
    ns = multiprocessing.Manager().Namespace()
    ns.docs = dialogs
    ns.idf_dict = idf_dict
    ns.batch_size = 2000
    ns.stopword_set = stopword_set

    p = multiprocessing.Pool()
    func = partial(handleData, ns)
    result = p.map(func, mylist)
    with open(targetfile, 'w') as fw:
        for data in result:
            for line in data:
                fw.write(json.dumps(line) + '\n')


if __name__ == "__main__":
    handleFile('data/raw/convai/train.txt', 'data/process/convai/train_raw.json')
    handleFile('data/raw/convai/test.txt', 'data/process/convai/test_raw.json')
    main('data/process/convai/train_raw.json', 'data/process/convai/train_keywords.json')
    main('data/process/convai/test_raw.json', 'data/process/convai/test_keywords.json')
    # print(result)
