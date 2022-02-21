import multiprocessing
import os
import re
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


def handleData(ns, start_idx):
    print("Worker process id for ", os.getpid())
    print('start_idx ', start_idx)
    # time.sleep(1)
    batch_data = ns.docs[start_idx: start_idx + ns.batch_size]
    if len(batch_data) == 0:
        return []
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
            print('utterance ', utterance)  # wow ! ! ! that's a lot lol
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
            nn_set, all_set = get_keywords_from_uttr(line, exclude_words)
            keywords_list1.append(nn_set)
            keywords_list2.append(all_set)
        data.append({'dialog_id': dialog['dialog_id'], 'nn_kw': keywords_list1, 'all_kw': keywords_list2})
    return data


def handleFile(filename, target_name):
    if 'valid' in filename:
        fw = open(target_name, 'a')
    else:
        fw = open(target_name, 'w')
    with open(filename, encoding="utf-8") as f:
        dialog_id = 0
        session = []
        for line in f:
            line = line.rstrip(' __eou__\n')
            regex = re.compile('\s\u2019\s')
            line = regex.sub('\'', line)
            session = [l.lower() for l in line.split(" __eou__ ")]
            if len(session) > 2:
                fw.write(json.dumps({
                    'dialog_id': dialog_id,
                    'session': session, }) + '\n')
                dialog_id += 1
    print('total dialog_session', dialog_id)


def main(filename, targetfile):
    with open(filename) as f:
        dialogs = [json.loads(row) for row in f]

    with open('data/stopwords.txt', encoding='utf-8') as stop_fil:
        stopword_set = set(stop_fil.read().lower().split("\n"))

    mylist = [i for i in range(0, len(dialogs), 2000)]
    ns = multiprocessing.Manager().Namespace()
    ns.docs = dialogs
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
    handleFile('data/raw/daily_dialogue/train.txt', 'data/process/daily_dialogue/train_raw.json')
    handleFile('data/raw/daily_dialogue/valid.txt', 'data/process/daily_dialogue/train_raw.json')
    handleFile('data/raw/daily_dialogue/test.txt', 'data/process/daily_dialogue/test_raw.json')

    main('data/process/daily_dialogue/train_raw.json', 'data/process/daily_dialogue/train_keywords.json')
    main('data/process/daily_dialogue/test_raw.json', 'data/process/daily_dialogue/test_keywords.json')
