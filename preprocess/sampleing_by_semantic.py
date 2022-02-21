# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import random

# Import our models. The package will take care of downloading the models automatically
# princeton-nlp/sup-simcse-bert-base-uncased
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to('cuda')
model.eval()


def convai_main(data_type, flatten_ziped):
    CANDIDATE_NUM = 1000
    all_data = []
    with open('data/process/convai/' + data_type + '_good_case.json') as f:
        for row in f:
            all_data.append(json.loads(row))
    with open('data/process/convai/' + data_type + '_bad_case.json') as f:
        for row in f:
            all_data.append(json.loads(row))

    bad_cases = []
    with torch.no_grad():
        for example in tqdm(all_data, mininterval=60):
            response = example['context'][-1]
            candidates = random.sample(flatten_ziped, CANDIDATE_NUM)
            candidates_embed = torch.stack([c[1] for c in candidates]).to('cuda')
            # Get the embeddings
            inputs = tokenizer([response], padding=True, truncation=True, return_tensors="pt").to('cuda')
            response_embed = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output  # 1 384
            cosine_sim = torch.cosine_similarity(response_embed, candidates_embed)
            topk_choice = random.choice(cosine_sim.topk(5)[1].tolist())
            r_ = candidates[topk_choice][0]
            bad_cases.append(example['context'][:2] + [r_])

    fout = open('data/process/convai/' + data_type + '_bad_case_cos.json', 'w')
    for context in bad_cases:
        fout.write(json.dumps({'context': context, }) + '\n')
    fout.close()
    print('data/process/convai/' + data_type + '_bad_case_cos.json saved')


def dd_main(data_type, flatten_ziped):
    CANDIDATE_NUM = 1000
    all_data = []
    with open('data/process/daily_dialogue/' + data_type + '_good_case.json') as f:
        for row in f:
            all_data.append(json.loads(row))
    with open('data/process/daily_dialogue/' + data_type + '_bad_case.json') as f:
        for row in f:
            all_data.append(json.loads(row))

    bad_cases = []
    with torch.no_grad():
        for example in tqdm(all_data, mininterval=60):
            response = example['context'][-1]
            candidates = random.sample(flatten_ziped, CANDIDATE_NUM)
            candidates_embed = torch.stack([c[1] for c in candidates]).to('cuda')
            # Get the embeddings
            inputs = tokenizer([response], padding=True, truncation=True, return_tensors="pt").to('cuda')
            response_embed = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output  # 1 384
            cosine_sim = torch.cosine_similarity(response_embed, candidates_embed)
            topk_choice = random.choice(cosine_sim.topk(5)[1].tolist())
            r_ = candidates[topk_choice][0]
            bad_cases.append(example['context'][:2] + [r_])

    fout = open('data/process/daily_dialogue/' + data_type + '_bad_case_cos.json', 'w')
    for context in bad_cases:
        fout.write(json.dumps({'context': context, }) + '\n')
    fout.close()
    print('data/process/daily_dialogue/' + data_type + '_bad_case_cos.json saved')


if __name__ == "__main__":
    # flatten_uttr = []
    # with open('data/process/daily_dialogue/train_raw.json') as f:
    #     for row in f:
    #         flatten_uttr.extend([line for line in json.loads(row)['session']])
    # with open('data/process/daily_dialogue/test_raw.json') as f:
    #     for row in f:
    #         flatten_uttr.extend([line for line in json.loads(row)['session']])
    # with torch.no_grad():
    #     batch = [i for i in range(0, len(flatten_uttr), 500)]
    #     flatten_ziped = []
    #     for start_idx in tqdm(batch, mininterval=30):
    #         batch_data = flatten_uttr[start_idx: start_idx+500]
    #         inputs = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt").to('cuda')
    #         embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    #         flatten_ziped.extend(zip(batch_data, embeddings.to('cpu')))

    # dd_main('test', flatten_ziped)
    # dd_main('train', flatten_ziped)

    flatten_uttr = []
    with open('data/process/convai/train_raw.json') as f:
        for row in f:
            flatten_uttr.extend([line for line in json.loads(row)['session']])
    with open('data/process/convai/test_raw.json') as f:
        for row in f:
            flatten_uttr.extend([line for line in json.loads(row)['session']])

    # flatten_uttr = flatten_uttr
    with torch.no_grad():
        batch = [i for i in range(0, len(flatten_uttr), 1000)]
        flatten_ziped = []
        for start_idx in tqdm(batch, mininterval=30):
            batch_data = flatten_uttr[start_idx: start_idx+1000]
            inputs = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt").to('cuda')
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            flatten_ziped.extend(zip(batch_data, embeddings.to('cpu')))

    convai_main('test', flatten_ziped)
    convai_main('train', flatten_ziped)

# nohup python -u 3_sampleing_by_semantic.py  > common.log 2>&1 &
