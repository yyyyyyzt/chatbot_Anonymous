import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModel
import torch
import random

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to('cuda')
model.eval()


def get_dialog_similarity(all_data):
    all_cos = []
    with torch.no_grad():
        for dialog in tqdm(all_data):
            # Tokenize input texts
            inputs = tokenizer(dialog, padding=True, truncation=True, return_tensors="pt").to('cuda')
            sentence_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            dialog_cos = []
            window = 1
            for i in range(window, len(sentence_embeddings)):
                # cos = torch.cosine_similarity(sentence_embeddings[i].unsqueeze(0), sentence_embeddings[i-window:i, :]).mean().item()
                cos = torch.cosine_similarity(sentence_embeddings[i], sentence_embeddings[i-window], dim=0).item()
                dialog_cos.append(cos)
            all_cos.extend(dialog_cos)

    return torch.tensor(all_cos).mean()


def main(dataset, datatype):
    print(dataset, datatype)
    with open('data/process/' + dataset + '/' + datatype + '_raw.json') as f:
        dialogs = [json.loads(row)['session'] for row in f]
    res = get_dialog_similarity(dialogs)
    print("dialog avg similarity", res)

    transition_path_len = []
    tokens_len = []
    dialogs = []
    with open('data/process/' + dataset + '/' + datatype + '_good_case.json') as f:
        for row in f:
            data = json.loads(row)
            dialogs.append(data['context'])
            transition_path_len.append(len(data['ans']))
            tokens_len.append(len(' '.join(data['context']).split(' ')))
    print('good case len ', len(dialogs))
    res = get_dialog_similarity(dialogs)
    print("good case avg similarity", res)
    print("good case transition_path_len", round(torch.tensor(transition_path_len).float().mean().item(), 2))
    print("good case tokens_len", round(torch.tensor(tokens_len).float().mean().item(), 2))

    with open('data/process/' + dataset + '/' + datatype + '_bad_case.json') as f:
        dialogs = [json.loads(row)['context'] for row in f]
    print('bad case len ', len(dialogs))
    res = get_dialog_similarity(dialogs)
    print("bad case by reasoning avg similarity", res)  # 0.3208

    # with open('data/process/' + dataset + '/' + datatype + '_bad_case_cos.json') as f:
    #     dialogs = [json.loads(row)['context'] for row in f]
    # res = get_dialog_similarity(dialogs)
    # print("bad case by cos avg similarity", res)  # 0.3208


if __name__ == "__main__":
    res = main('convai', 'test')
    # dialog avg similarity tensor(0.3083)  good case len  5964
    # good case avg similarity tensor(0.3229)  good case transition_path_len 3.71
    # good case tokens_len 38.84   bad case len 4183  bad case by reasoning avg similarity tensor(0.2971)
    res = main('convai', 'train')
    # dialog avg similarity tensor(0.3082 good case len  88900  good case avg similarity tensor(0.3204)
    # good case transition_path_len 3.67 good case tokens_len 38.68  bad case len 67246 bad case by reasoning avg similarity tensor(0.2962)
    res = main('daily_dialogue', 'test')  # 0.2985  1397 0.3440 3.9 56.69   1393 0.3149
    res = main('daily_dialogue', 'train')  # 0.3004 19424 0.3395 43.83 46.56 13597 0.3267

# python 4_dataset_stastic.py
