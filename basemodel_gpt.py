# -*- coding: utf-8 -*-
from email.policy import default
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, progress
from transformers.optimization import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchmetrics import MetricCollection, Precision, Recall, BLEUScore, ROUGEScore
import json
import random
from load_utils import pad_to_max_seq_len
from textblob import Word
from find_path_vanilla import FindPathModel
import re
import argparse


class End2EndModel(pl.LightningModule):
    def __init__(self, learning_rate=5e-4, datatype='convai', **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.datatype = datatype
        self.max_len = 128
        self.save_hyperparameters()  # 把上面参数都保存下来
        self.find_path = FindPathModel()
        self.dec_tok = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small", cache_dir='./cache_bert')
        self.dec_tok.add_special_tokens({
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': ["<s_k>", "<e_k>"]
        })
        self.dec_tok.k_start = self.dec_tok.all_special_ids[-2]
        self.dec_tok.k_end = self.dec_tok.all_special_ids[-1]
        self.decoder = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small", output_hidden_states=True, cache_dir='./cache_bert')
        self.decoder.resize_token_embeddings(len(self.dec_tok))
        self.val_recall = Recall(ignore_index=0)  # top_k=10 正样本中有多少是被找了出来
        self.val_precision = Precision(ignore_index=0)  # 你认为的正样本中，有多少是真的正确的概率
        self.bleu1 = BLEUScore(n_gram=1)
        self.bleu2 = BLEUScore(n_gram=2)
        self.rouge = ROUGEScore(rouge_keys=("rouge1", "rouge2"))
        self.epochs = 1

    def prepare_data(self):
        self.test_result = {'keyword': [], 'response': [], 'ref': [], 'context': []}
        self.counter = []

    def setup(self, stage: str = None):
        prefix = 'data/process/' + self.datatype
        print('use prefix ', prefix)
        if stage == 'fit':
            self.batch_size = 24
            with open(prefix + '/train_good_case.json') as f:
                train_data = [json.loads(row) for row in f]
            self.train_dataset = train_data[:-2500]
            self.val_dataset = train_data[-2500:]
            print(f"train_len: {len(self.train_dataset)}, valid_len: {len(self.val_dataset)}")
        elif stage == 'test':
            print('test')
            self.batch_size = 16
            self.test_dataset = []
            with open(prefix + '/test_good_case.json') as f:
                self.test_dataset = [json.loads(row) for row in f]
            print(f"test_len: {len(self.test_dataset)}")
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(self.device)
            self.model.eval()

    def stem(self, w):
        word1 = Word(w).lemmatize("n")
        word2 = Word(word1).lemmatize("v")
        word3 = Word(word2).lemmatize("a")
        return Word(word3)

    def _flatten(self, context=None, target=None, ans=None, response=None):
        SAMPLE_NUM = 6
        # target = "<t>" + target
        target_id = self.dec_tok.encode(target, add_special_tokens=False)
        eos = self.dec_tok.eos_token
        prompt = eos + eos.join(context) + eos
        context_id = self.dec_tok.encode(prompt, add_special_tokens=False)
        if ans is None:
            return target_id, context_id + [self.dec_tok.k_start]

        raw_keyword_text = list(set([w for p in ans for w in p[1:-1]]))
        keyword_text_list = random.sample(raw_keyword_text, SAMPLE_NUM) if len(raw_keyword_text) > SAMPLE_NUM else raw_keyword_text
        keyword_id_list = []
        for keyword in keyword_text_list:
            keyword_id_list.append(self.dec_tok.encode("<s_k>" + keyword + "<e_k>", add_special_tokens=False))

        response = eos + response + eos
        response_id = self.dec_tok.encode(response, add_special_tokens=False)
        return target_id, context_id, keyword_id_list, response_id, raw_keyword_text

    def collate_fn(self, batch):
        response_text = []
        keyword_text = []

        response_ids = []
        response_labels = []
        response_mask = []

        keyword_ids = []
        keyword_labels = []
        keyword_mask = []

        for row in batch:  # context kws target ans
            #target_id, context_id, keyword_id_list, response_id, raw_keyword_text
            t_id, c_id, k_id_list, r_id, k_text_list = self._flatten(
                context=row['context'][:-1],
                target=row['target'],
                ans=row['ans'],
                response=row['context'][-1],
            )
            if len(k_text_list) == 0:
                continue
            response_text.append(row['context'][-1])
            keyword_text.append(k_text_list)
            keyword_ids.extend([t_id + c_id + k_id for k_id in k_id_list])
            keyword_labels.extend([[-100] * len(t_id + c_id) + k_id for k_id in k_id_list])
            keyword_mask.extend([[1] * len(k_id) for k_id in keyword_ids[-len(k_id_list):]])

            random_k_id = random.choice(k_id_list)
            response_ids.append(t_id + c_id + random_k_id + r_id)
            response_labels.append([-100] * len(t_id) + c_id + random_k_id + r_id)
            response_mask.append([1] * len(response_ids[-1]))

        max_seq_len = pad_to_max_seq_len(keyword_ids, pad_token_id=self.dec_tok.k_end, max_len=self.max_len)
        pad_to_max_seq_len(keyword_mask, max_seq_len, 0, self.max_len)
        pad_to_max_seq_len(keyword_labels, max_seq_len, -100, self.max_len)

        max_seq_len = pad_to_max_seq_len(response_ids, pad_token_id=50256, max_len=self.max_len)
        pad_to_max_seq_len(response_mask, max_seq_len, 0, self.max_len)
        pad_to_max_seq_len(response_labels, max_seq_len, -100, self.max_len)

        return {
            'response_text': response_text,
            'keyword_text': keyword_text,
            'response_ids': torch.tensor(response_ids),
            'response_mask': torch.tensor(response_mask),
            'response_labels': torch.tensor(response_labels),
            'keyword_ids': torch.tensor(keyword_ids),
            'keyword_labels': torch.tensor(keyword_labels),
            'keyword_mask': torch.tensor(keyword_mask),
        }

    def test_collate_fn(self, batch):
        response_text = []
        context_text = []
        keyword_text = []
        test_input_ids = []
        test_input_mask = []
        for row in batch:  # context kws target ans
            t_id, c_id, k_id_list, r_id, k_text_list = self._flatten(
                context=row['context'][:-1],
                target=row['target'],
                ans=row['ans'],
                response=row['context'][-1],
            )
            if len(k_text_list) == 0:
                continue
            response_text.append(row['context'][-1])
            context_text.append(row['context'][-2])
            keyword_text.append(k_text_list)
            test_input_ids.append(t_id + c_id + [self.dec_tok.k_start])
            test_input_mask.append([1] * len(test_input_ids[-1]))

        raw_ids = test_input_ids.copy()
        max_seq_len = pad_to_max_seq_len(test_input_ids, pad_token_id=self.dec_tok.k_start, max_len=self.max_len)
        pad_to_max_seq_len(test_input_mask, max_seq_len, 0, self.max_len)

        return {
            'context_text': context_text,
            'response_text': response_text,
            'keyword_text': keyword_text,
            'raw_input_ids': raw_ids,
            'test_input_ids': torch.tensor(test_input_ids),
            'test_input_mask': torch.tensor(test_input_mask),
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=8, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,  shuffle=False, num_workers=8, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  shuffle=False, num_workers=0, batch_size=self.batch_size, collate_fn=self.test_collate_fn)

    def forward(self,
                response_text=None,
                context_text=None,
                keyword_text=None,
                response_ids=None,
                response_mask=None,
                response_labels=None,
                keyword_ids=None,
                keyword_labels=None,
                keyword_mask=None,
                position_ids=None,
                ):

        loss = self.decoder(
            response_ids,
            attention_mask=response_mask,
            labels=response_labels,
        )['loss']
        # loss += self.decoder(
        #     keyword_ids,
        #     attention_mask=keyword_mask,
        #     labels=keyword_labels,
        # )['loss']
        return {'loss': loss}

    def training_step(self, batch: tuple, batch_idx: int):
        outputs = self(**batch)
        return outputs['loss']

    def on_train_epoch_end(self, *args, **kwargs):
        print("on_train_epoch_end", self.epochs - 1)
        self.epochs += 1

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self(**batch)
        return outputs['loss']

    def validation_epoch_end(self, val_step_outputs):
        val_loss = torch.tensor(val_step_outputs).mean()
        self.log('val_loss', val_loss.item())

    def test_step(self, batch: dict, batch_idx: int):
        key_preds, res_preds = self.predict(
            batch['test_input_ids'],
            batch['test_input_mask'],
            batch['raw_input_ids'],
        )
        for k, ans in zip(key_preds, batch['keyword_text']):
            self.test_result['keyword'].append(1 if k in ans else 0)

        self.model.eval()
        inputs = self.tokenizer(res_preds, padding=True, truncation=True, return_tensors="pt").to('cuda')
        response_embed = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output  # 1 384
        inputs = self.tokenizer(batch['context_text'], padding=True, truncation=True, return_tensors="pt").to('cuda')
        ref_embed = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output  # 1 384
        cos = []
        for res_e, ref_e in zip(response_embed, ref_embed):
            cos.append(torch.cosine_similarity(res_e, ref_e, dim=0))

        for pred, ref, context in zip(res_preds, batch['response_text'], batch['context_text']):
            self.test_result['response'].append(pred)
            self.test_result['ref'].append(ref)
            self.test_result['context'].append(context)
            # if len(pred) > 0:
            #     pred = re.sub(r'[^\w\s]', '', pred.lstrip(' '))
            #     ref = re.sub(r'[^\w\s]', '', ref)
            #     self.rouge.update(pred, ref)  # pred 在上面正则了
            #     self.bleu1.update(pred, [ref])
            #     self.bleu2.update(pred, [ref])
        if batch_idx % 100 == 0:
            print('\n context | pred | ref \n', context + ' | ' + pred + ' | ' + ref)
        return torch.tensor(cos).mean()

    def test_epoch_end(self, test_step_outputs):
        # print("total len ", torch.tensor(self.counter).sum(0).tolist())
        acc = torch.tensor(self.test_result['keyword']).sum().item()
        total = len(self.test_result['keyword'])
        print("keyword precision: ", acc, total, acc / total)
        self.log('precision', round(acc / total, 5))

        print('metric_cos', torch.tensor(test_step_outputs).mean())
        # score1 = self.rouge.compute()
        # score21 = self.bleu1.compute()
        # score22 = self.bleu2.compute()
        # self.rouge.reset()
        # self.bleu1.reset()
        # self.bleu2.reset()

        # print('metric_rouge', score1)
        # print('metric_bleu1', score21)
        # print('metric_bleu2', score22)
        self.test_result = {'keyword': [], 'response': [], 'ref': [], 'context': []}
        pass

    def predict(self, input_ids, attention_mask, raw_input_ids, return_ids=False):
        keyword_out = self.decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            pad_token_id=self.dec_tok.k_end,
            eos_token_id=self.dec_tok.k_end,
            num_beams=3,
            top_k=40,
            top_p=0.95,
            length_penalty=1.0,
            early_stopping=True
        )
        seq_len = input_ids.shape[1] - 1
        key_out = keyword_out[:, seq_len:]
        key_preds = [self.dec_tok.decode(k, skip_special_tokens=True) for k in key_out]

        keyword_input_ids = []
        keyword_input_mask = []
        for k, raw_ids in zip(key_preds, raw_input_ids):
            ids = raw_ids + self.dec_tok.encode(k + '<e_k>' + self.dec_tok.eos_token, add_special_tokens=False)
            keyword_input_ids.append(ids)
            keyword_input_mask.append([1] * len(ids))
        max_seq_len = pad_to_max_seq_len(keyword_input_ids, pad_token_id=50256, max_len=self.max_len)
        pad_to_max_seq_len(keyword_input_mask, max_seq_len, 0, self.max_len)
        # 得到了原始句子加上预测的关键词的id
        response_out = self.decoder.generate(
            input_ids=torch.tensor(keyword_input_ids).to(self.device),
            attention_mask=torch.tensor(keyword_input_mask).to(self.device),
            max_new_tokens=30,
            pad_token_id=50256,
            eos_token_id=50256,
            num_beams=3,
            # do_sample=True,
            top_k=40,
            top_p=0.95,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True
        )
        res_out = response_out[:, max_seq_len-1:]
        res_preds = [self.dec_tok.decode(k, skip_special_tokens=True) for k in res_out]
        if return_ids is True:
            r_ids = []
            # return_masks = []
            for k, r, raw_ids in zip(key_preds, res_preds, raw_input_ids):
                ids = raw_ids + self.dec_tok.encode(k + '<e_k>' + self.dec_tok.eos_token + r + self.dec_tok.eos_token, add_special_tokens=False)
                r_ids.append(ids)
                # return_masks.append([1] * len(ids))
            # max_seq_len = pad_to_max_seq_len(r_ids, pad_token_id=50256, max_len=self.max_len)
            # pad_to_max_seq_len(return_masks, max_seq_len, 0, self.max_len)
            return key_preds, res_preds, r_ids
        return key_preds, res_preds

    def configure_optimizers(self):
        # params = [param for param in self.parameters() if param.requires_grad]  # 不微调嵌入
        optimizer = AdamW(self.parameters(), weight_decay=0.001, lr=self.learning_rate, eps=1e-8)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--datatype", default="convai", type=str)
    args = parser.parse_args()
    seed_everything(20, workers=True)
    tb_logger = pl_loggers.TensorBoardLogger('logs_base/', name='')
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_last=True,
        verbose=True,
        filename='best',
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=100)
    model = End2EndModel(datatype=args.datatype)
    model = model.load_from_checkpoint(
        'logs_base/version_1/checkpoints/last.ckpt',
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=3,
        logger=tb_logger,
        detect_anomaly=True,
        accumulate_grad_batches=3,
        gradient_clip_val=0.5,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback, bar_callback],
    )
    # trainer.fit(model)
    trainer.test(model)

# nohup python -u basemodel_gpt.py --datatype daily_dialog   > basemodel_gpt.log 2>&1 &
# keyword precision:  3525 8816 0.39984119782214156
# metric_cos tensor(0.28)
