# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, progress, EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, BLEUScore, ROUGEScore
import numpy as np
import json
import random
import argparse


class Discriminator(pl.LightningModule):
    def __init__(self, learning_rate=3e-5, datatype='convai', reasoning=False, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.datatype = datatype
        self.reasoning = reasoning
        self.save_hyperparameters()  # 把上面参数都保存下来
        self.max_len = 128
        self.tokenizer = BertTokenizer.from_pretrained('./bert_tok')
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1, cache_dir='./cache_bert')
        self.metrics = MetricCollection([
            Accuracy(),
            Precision(ignore_index=0),
            Recall(ignore_index=0),
        ])

    def prepare_data(self):
        # train 的时候会默认第一顺序执行
        self.val_result = {'pred': [], 'ref': []}
        self.test_result = {'pred': [], 'ref': []}

    def setup(self, stage: str = None):
        self.prefix = 'data/process/' + self.datatype
        print('use prefix ', self.prefix)
        positive = []
        negative = []
        if stage == 'fit':
            self.batch_size = 64
            if self.reasoning:
                with open(self.prefix + '/train_good_case.json') as f:
                    positive.extend([json.loads(row) for row in f])
                with open(self.prefix + '/train_bad_case.json') as f:
                    negative.extend([json.loads(row) for row in f])
            else:
                with open(self.prefix + '/train_good_case.json') as f:
                    positive.extend([json.loads(row) for row in f])
                with open(self.prefix + '/train_bad_case.json') as f:
                    positive.extend([json.loads(row) for row in f])
                with open(self.prefix + '/train_bad_case_cos.json') as f:
                    negative.extend([json.loads(row) for row in f])

            self.pos_weight = round(len(negative) / len(positive), 1)
            for row in positive:
                row['label'] = 1
            for row in negative:
                row['label'] = 0
            positive.extend(negative)
            random.shuffle(positive)
            train_data = positive[1000:]
            valid_data = positive[:1000]
            print(f"train_len: {len(train_data)}, valid_len: {len(valid_data)}")
            self.train_dataset = train_data
            self.val_dataset = valid_data
        elif stage == 'test':
            self.batch_size = 128
            self.pos_weight = 1.0
            if self.reasoning:
                with open(self.prefix + '/test_good_case.json') as f:
                    positive.extend([json.loads(row) for row in f])
                with open(self.prefix + '/test_bad_case.json') as f:
                    negative.extend([json.loads(row) for row in f])
            else:
                with open(self.prefix + '/test_good_case.json') as f:
                    positive.extend([json.loads(row) for row in f])
                with open(self.prefix + '/test_bad_case.json') as f:
                    positive.extend([json.loads(row) for row in f])
                with open(self.prefix + '/test_bad_case_cos.json') as f:
                    negative.extend([json.loads(row) for row in f])
            for row in positive:
                row['label'] = 1
            for row in negative:
                row['label'] = 0
            self.test_dataset = positive
            self.test_dataset.extend(negative)

    def collate_fn(self, batch):
        # context kws target ans
        eos = '[SEP]'
        labels = []
        seqs = []
        weights = []
        for idx, row in enumerate(batch):
            source = eos.join(row['context'])
            seqs.append(source)
            if 'score' in row:
                labels.append(1 if row['score'] > 0.5 else 0)
                continue
            if row['label'] == 1:
                labels.append(1)
                weights.append(self.pos_weight)
            else:
                labels.append(0)
                weights.append(1)

        input_encoding = self.tokenizer.batch_encode_plus(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            max_length=self.max_len,
        )

        return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': torch.tensor(labels).float(),
            'weights': torch.tensor(weights).float(),
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=8, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,  shuffle=False, num_workers=8, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  shuffle=False, num_workers=8, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def forward(self, input_ids, input_mask=None, labels=None, weights=None):
        output = self.model(
            input_ids,
            attention_mask=input_mask,
        )['logits'].squeeze(1)
        loss = None
        prob = output.sigmoid()
        pred = torch.where(prob > 0.5, 1, 0)
        if labels is not None:
            loss_fct = nn.BCELoss(weight=weights)
            loss = loss_fct(prob, labels)
        return {'logits': prob,  'loss': loss, 'pred': pred}

    def training_step(self, batch: tuple, batch_idx: int):
        # 必须返回loss用于反向传播
        outputs = self(batch['input_ids'], input_mask=batch['attention_mask'], labels=batch['labels'], weights=batch['weights'])
        self.log('train_loss', outputs['loss'])
        return outputs['loss']

    def validation_step(self, batch: tuple, batch_idx: int):
        outputs = self(batch['input_ids'], input_mask=batch['attention_mask'], labels=batch['labels'], weights=batch['weights'])
        self.metrics.update(outputs['pred'], batch['labels'].int())
        return outputs['loss']

    def validation_epoch_end(self, val_step_outputs=None):
        val_loss = torch.tensor(val_step_outputs).mean()
        self.log('val_loss', val_loss.item())
        score = self.metrics.compute()
        print('metrics', score)
        self.log('metrics', score)
        self.metrics.reset()

    def test_step(self, batch: tuple, batch_idx: int):
        outputs = self(batch['input_ids'], input_mask=batch['attention_mask'], labels=batch['labels'], weights=batch['weights'])
        self.metrics.update(outputs['pred'], batch['labels'].int())

    def test_epoch_end(self, test_step_outputs=None):
        score = self.metrics.compute()
        print('testing metrics', score)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2, eps=1e-8)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--datatype", default="convai", type=str)
    args = parser.parse_args()

    seed_everything(20, workers=True)
    tb_logger = pl_loggers.TensorBoardLogger('logs_discri/', name='')
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_last=True,
        verbose=True,
        filename='best',
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=100)
    # early_stopping = EarlyStopping(
    #     monitor='val_loss', mode='min', verbose=True,
    # )
    model = Discriminator(datatype=args.datatype, reasoning=False)
    # model = Discriminator.load_from_checkpoint(
    #     'logs_discri/version_4/checkpoints/last.ckpt',
    # )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=2,
        logger=tb_logger,
        detect_anomaly=True,
        gradient_clip_val=0.5,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback, bar_callback],
        # amp_backend='apex',
        # precision=16,
        # overfit_batches=0.01,
    )
    trainer.fit(model)
    trainer.test(model)

# nohup python -u discriminator.py > discriminator.log 2>&1 &
# dd_reasoning_best {'Accuracy': tensor(0.7803, device='cuda:0'), 'Precision': tensor(0.8861, device='cuda:0'), 'Recall': tensor(0.7029, device='cuda:0')}
# dd_cos_best {'Accuracy': tensor(0.8189, device='cuda:0'), 'Precision': tensor(0.8200, device='cuda:0'), 'Recall': tensor(0.8170, device='cuda:0')}
# cv_reasoning_best {'Accuracy': tensor(0.7753, device='cuda:0'), 'Precision': tensor(0.8005, device='cuda:0'), 'Recall': tensor(0.8287, device='cuda:0')}
# cv_cos_best {'Accuracy': tensor(0.8019, device='cuda:0'), 'Precision': tensor(0.8597, device='cuda:0'), 'Recall': tensor(0.7216, device='cuda:0')}
