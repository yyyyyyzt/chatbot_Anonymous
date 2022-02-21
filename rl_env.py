import json
import torch
import random
import yake
from basemodel_gpt import End2EndModel
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from discriminator import Discriminator
from textblob import Word
import pickle


def load_model_from_experiment(experiment_folder: str):
    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[0]
    print('checkpoint_path', checkpoint_path)
    model = Discriminator.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    return model


class Env(object):
    def __init__(self, base_model, disc_path=None, data_path=None):
        super().__init__()

        self.base_model: End2EndModel = base_model
        # self.graph_embed = base_model.graph_embed
        self.kw_extractor = yake.KeywordExtractor(top=4)

        self.disc: Discriminator = load_model_from_experiment(disc_path)
        self.device = 'cuda'
        self.disc.to(self.device)

        with open(data_path) as f:
            train_data = [json.loads(row) for row in f]
        self.target_set = pickle.load(open('data/target_set.pkl', 'rb'))
        # self.target_set = ['dance', 'music', 'party', 'band', 'movie', 'travel', 'basketball', 'sport', 'football']

        self.train_dataset = train_data
        self.train_idx = -1
        self.counter = []
        self.user_model = None

    def stem(self, w):
        word1 = Word(w).lemmatize("n")
        word2 = Word(word1).lemmatize("v")
        word3 = Word(word2).lemmatize("a")
        return Word(word3)

    def reset(self):
        self.train_idx = self.train_idx + 1
        if self.train_idx >= len(self.train_dataset) - 1:
            self.train_idx = 0
        row = self.train_dataset[self.train_idx]  # context ['','',''] kws
        self.context: list = row['context'][:2].copy()
        self.kws: list = row['kws'][:2].copy()
        # self.target = random.choice(row['kws'][-1])
        self.target = random.choice(self.target_set)

        self.sub_g = self.base_model.find_path.get_node_by_keyword(self.kws, self.target)
        self.sub_g_words = {}
        for n in self.sub_g:
            self.sub_g_words[n] = self.base_model.find_path.graph_embed[n]
        target_ids, prompt_ids = self.base_model._flatten(
            context=self.context,
            target=self.target,
        )
        assert len(target_ids + prompt_ids) > 0
        return target_ids + prompt_ids

    def step_with_user(self, response: str):
        if self.user_model is None:
            self.user_tok = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small", cache_dir='./cache_bert')
            self.user_model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small", output_hidden_states=True, cache_dir='./cache_bert')
            self.user_model.eval()
            self.user_model.to(self.device)

        self.context.append(response)
        eos = self.user_tok.eos_token
        user_inputs = self.user_tok.encode(eos.join(self.context[-3:]) + eos, return_tensors='pt')
        user_out = self.user_model.generate(
            input_ids=user_inputs.to(self.device),
            max_new_tokens=30,
            pad_token_id=50256,
            eos_token_id=50256,
            num_beams=3,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True
        )
        user_out = user_out[0, user_inputs.shape[-1] - 1:]
        user_response = self.user_tok.decode(user_out, skip_special_tokens=True)
        self.context.append(user_response)
        self.kws.append('<usr>')
        target_ids, prompt_ids = self.base_model._flatten(
            context=self.context[-2:],
            target=self.target,
        )
        assert len(prompt_ids) != 0
        seq = '[SEP]'.join(self.context[-3:])
        input_encoding = self.disc.tokenizer.encode(seq, return_token_type_ids=False, return_tensors="pt")
        output = self.disc.model(
            input_encoding.to(self.device),
        )['logits'].squeeze(1)
        reward = round(output.sigmoid().item() - 0.5, 4)
        done = 0
        res_tokens = [self.stem(t) for t in response.split(' ')]
        target = self.stem(self.target)
        if target in res_tokens or self.target in response:
            return target_ids + prompt_ids, reward * 4.0, 1, 0
        return target_ids + prompt_ids, reward * 2.0, done, 0

    def step(self, response: str):
        # 已经得到了action对应的单词
        self.context.append(response)
        target_ids, prompt_ids = self.base_model._flatten(
            context=self.context[-2:],
            target=self.target,
        )
        assert len(prompt_ids) != 0

        seq = '[SEP]'.join(self.context[-3:])
        input_encoding = self.disc.tokenizer.encode(seq, return_token_type_ids=False, return_tensors="pt")
        output = self.disc.model(
            input_encoding.to(self.device),
        )['logits'].squeeze(1)
        reward = round(output.sigmoid().item() - 0.5, 4)
        done = 0
        res_tokens = [self.stem(t) for t in response.split(' ')]
        target = self.stem(self.target)
        if target in res_tokens or self.target in response:
            return target_ids + prompt_ids, 4.0, 1, 0
        if len(self.context) >= 10:
            return target_ids + prompt_ids, -2.0 + reward * 2.0, 1, 0
        return target_ids + prompt_ids, reward * 2.0, done, 0
