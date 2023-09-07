import json
import os
import random
import time
import sklearn
from sklearn.model_selection import train_test_split


from model import ClassifierCodeT5
from loss import SMARTLoss, kl_loss, sym_kl_loss
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from transformers import get_cosine_schedule_with_warmup
from transformers import logging as hf_logging
from transformers import set_seed
from collections import Counter
import wandb
import pdb
import itertools
import torch
import numpy as np
import math
import scipy.stats as stats
import datasets
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel
from torch.cuda.amp import autocast, GradScaler



hf_logging.set_verbosity_error()  # or hf_logging.set_verbosity(hf_logging.ERROR)
set_seed(7) # Set all random seed from pytorch, numpy, random
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set path to SentEval
PATH_TO_SENTEVAL = '/SimCSE/SentEval'
PATH_TO_DATA = '/SimCSE/SentEval/data'

# Import SentEval
import sys
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock
from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count



class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_lst, tokenizer):
        self.tokenizer= tokenizer
        self.data = []
        for i, inputs in enumerate(data_lst):
            self.data.append([inputs['sent0'],inputs['sent1'],inputs['hard_neg']])


    def __len__(self):
        # has to override __len__
        return len(self.data)

    def __getitem__(self, index):
        # has to override __getitem__
        return self.data[index]


    def custom_collate(self,data_lst):
        # this function takes a list of data sample, merge them into one batch of data
        data_batch = {'input_ids_1': [], 'input_ids_2': [], 'input_ids_3': [], 'attention_mask_1': [], 'attention_mask_2': [],'attention_mask_3': []}

        encode0 = self.tokenizer.batch_encode_plus([i[0]for i in data_lst],padding = 'longest', truncation = True, return_tensors='pt')
        encode1 = self.tokenizer.batch_encode_plus([i[1]for i in data_lst],padding = 'longest', truncation = True, return_tensors='pt')
        encode2 = self.tokenizer.batch_encode_plus([i[2]for i in data_lst],padding = 'longest', truncation = True, return_tensors='pt')

        data_batch['input_ids_1'] = encode0['input_ids']
        data_batch['input_ids_2'] = encode1['input_ids']
        data_batch['input_ids_3'] = encode2['input_ids']
        data_batch['attention_mask_1'] = encode0['attention_mask']
        data_batch['attention_mask_2'] = encode1['attention_mask']
        data_batch['attention_mask_3'] = encode2['attention_mask']

        del data_lst
        return data_batch





class Trainer():
    def __init__(self, train_dataset, tokenizer, model, lr, alpha, temp, device, epochs, batch_size):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.lr = lr
        self.alpha = alpha
        self.temp = temp
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.best_metric = None

    def sim_loss(self,x1,x2,x3): # NTX Loss

        loss_fn = torch.nn.CrossEntropyLoss()
        t = self.temp
        sim_pos = torch.nn.functional.cosine_similarity(x1.unsqueeze(1),x2.unsqueeze(0),dim=-1).to(self.device) / t
        sim_neg = torch.nn.functional.cosine_similarity(x1.unsqueeze(1), x3.unsqueeze(0), dim=-1).to(self.device) / t

        cos_sim = torch.cat([sim_pos,  sim_neg], dim =1)

        weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - sim_neg.size(-1)) + [0.0] * i + [1] + [0.0] * (sim_neg.size(-1) - i - 1) for i in range(sim_neg.size(-1))]).to(self.device) # adding weights for true negative pairs which are diagnal
        cos_sim = cos_sim + weights

        labels = torch.arange(cos_sim.size(0)).long().to(self.device)

        contr_loss = loss_fn(cos_sim, labels)

        return contr_loss



    def prepare(self, params, samples):
                return

    def batcher(self, params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding='longest',
        )
        for k in batch:
            batch[k] = batch[k].to(self.device)
        with torch.no_grad():
            outputs= self.model.raw_emb(**batch)

        return outputs.cpu()

    def _save_checkpoint(self, model, metrics=None):
        output_dir = '/content/data'
        self.metric_for_best_model = 'stsb_spearman'

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.metric_for_best_model is not None:
            metric_to_check = self.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater
            if (
                self.best_metric is None
                or self.best_model_checkpoint is None
                or operator(metric_value, self.best_metric)
            ):
                self.best_metric = metric_value
                self.best_model_checkpoint = output_dir

                # Only save model when it is the best one
                if wandb_on:
                    self.model.save(os.path.join(output_dir,'best_model.pth'))
                    wandb.save(os.path.join(output_dir,'best_model.pth'))
                    torch.save(model.state_dict(), os.path.join(output_dir,'best_layer.pth'))
                    wandb.save(os.path.join(output_dir,'best_layer.pth'))

        return self.best_metric


    def evaluation(self, eval_senteval_transfer=False):

        self.model.eval() # prevent taking gradient
        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, self.batcher, self.prepare)
        tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if eval_senteval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        results = se.eval(tasks)

        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        sts12 = results['STS12']['all']['spearman']['all']
        sts13 =results['STS13']['all']['spearman']['all']
        sts14 = results['STS14']['all']['spearman']['all']
        sts15 = results['STS15']['all']['spearman']['all']
        sts16 = results['STS16']['all']['spearman']['all']
        avg = (stsb_spearman + sickr_spearman + sts12 + sts13 + sts14 + sts15 + sts16) / 7
        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "sts12": sts12, "sts13": sts13, "sts14": sts14, "sts15":sts15, "sts16":sts16, "avg": avg}
        if eval_senteval_transfer:
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer



        return metrics


    def finetune(self):
        train_dataset = Train_Dataset(self.train_dataset, self.tokenizer)
        print('finishied processing train dataset')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size= self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=train_dataset.custom_collate
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(self.epochs * len(train_loader))
        )
        smart_loss_fn = SMARTLoss(eval_fn = self.model.forward, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        max_test_accuracy = 0
        scaler = GradScaler()

        for epoch in range(self.epochs):
            self.model.train()

            train_loss = []
            start_time = time.time()

            for i, data in enumerate(train_loader):

                data = {
                    'input_ids_1': data['input_ids_1'].to(device),
                    'input_ids_2': data['input_ids_2'].to(device),
                    'input_ids_3': data['input_ids_3'].to(device),
                    'attention_mask_1': data['attention_mask_1'].to(device),
                    'attention_mask_2': data['attention_mask_2'].to(device),
                    'attention_mask_3': data['attention_mask_3'].to(device)
                    }
                with autocast(dtype = torch.float16):
                    logits1, emb1 = self.model(input_ids=data['input_ids_1'], attention_mask=data['attention_mask_1'])
                    logits2, emb2 = self.model(input_ids=data['input_ids_2'], attention_mask=data['attention_mask_2'])
                    logits3, emb3 = self.model(input_ids=data['input_ids_3'], attention_mask=data['attention_mask_3'])

                    logits = torch.stack([logits1, logits2, logits3], dim = 0)
                    loss_adv = smart_loss_fn(emb1,logits[0])
                    loss_cont = self.sim_loss(x1 =logits1,x2 = logits2, x3= logits3)

                    loss = self.alpha*loss_adv + loss_cont


                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                self.model.zero_grad()
                optimizer.zero_grad()
                train_loss.append(loss.item())

                if (i+1) % 125 ==0:
                    test_accuracy  = self.evaluation()
                    best_metric = self._save_checkpoint(self.model, test_accuracy)

                    print('step: {}, train_loss: {}, Test STSB: {}, TEST SICK: {}, STS12: {}, STS13: {}, STS14: {}, STS15: {}, STS16: {}, AVG: {},  Best_STSB: {}, time: {}s'.format(
                    i + 1,
                    round(sum(train_loss) / len(train_loss), 4),
                    round(test_accuracy['eval_stsb_spearman'],4),
                    round(test_accuracy['eval_sickr_spearman'], 4),
                    round(test_accuracy['sts12'], 4),
                    round(test_accuracy['sts13'], 4),
                    round(test_accuracy['sts14'], 4),
                    round(test_accuracy['sts15'], 4),
                    round(test_accuracy['sts16'], 4),
                    round(test_accuracy['avg'], 4),
                    round(best_metric,4),
                    int(time.time() - start_time)
                    ))
                    if wandb_on:
                        wandb.log({'epoch': epoch + 1, 'train_loss': round(sum(train_loss) / len(train_loss), 4), 'Test STSB':round(test_accuracy['eval_stsb_spearman'],4), 'TEST SICK': round(test_accuracy['eval_sickr_spearman'], 4),'Best_STSB': round(best_metric,4)})

            test_accuracy  = self.evaluation()

            best_metric = self._save_checkpoint(self.model, test_accuracy)
            print('step: {}, train_loss: {}, Test STSB: {}, TEST SICK: {}, Best_STSB: {}, time: {}s'.format(
                i + 1,
                round(sum(train_loss) / len(train_loss), 4),
                round(test_accuracy['eval_stsb_spearman'],4),
                round(test_accuracy['eval_sickr_spearman'], 4),
                round(best_metric,4),
                int(time.time() - start_time)
            ))
            if wandb_on:
                wandb.log({'epoch': epoch + 1, 'train_loss': round(sum(train_loss) / len(train_loss), 4), 'Test STSB':round(test_accuracy['eval_stsb_spearman'],4), 'TEST SICK': round(test_accuracy['eval_sickr_spearman'], 4),'Best_STSB': round(best_metric,4)})


        return


def run():

    if wandb_on:
        wandb.login()
        wandb.init(config = {'method' : 'no mlp','batch': 60, 'model': 'roberta'}, group = '10 authors contrastive_sim_cse')
        cfig = wandb.config
        batch = cfig.batch
        lr = cfig.lr
        alpha = cfig.alpha
        temp = cfig.temp
    else:
        lr = 5e-5
        alpha = 0.5
        temp = 0.05
        batch= 60
    print('start train')

    ds = load_dataset("csv", data_files = '/SimCSE/data/nli_for_simcse.csv', split="train")
    test_accuracy_lst = []

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    max_length = model.config.max_position_embeddings

    model = ClassifierCodeT5(encoder=model)
    model = model.to(device)
    simcse_trainer = Trainer(train_dataset=ds, tokenizer=tokenizer, model=model, lr= lr, alpha= alpha, temp=temp, device=device, epochs=3, batch_size=batch)

    simcse_trainer.finetune()

    # free out gpu
    torch.cuda.empty_cache()
    torch.cuda.amp._amp_state = None
    del model
    del simcse_trainer
    return


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_on = True # change this if you don't want to run using wandb
    if wandb_on:
        sweep_config = dict()
        sweep_config['method'] = 'grid'

        sweep_config['parameters'] = {'lr' : {'values':[5e-5]}, 'alpha' : {'values' : [0]}, 'temp':{'values': [0.05]}}
        sweep_id = wandb.sweep(sweep_config, project = 'Authorship_Method_1')
        wandb.agent(sweep_id, run)
    else:
        run()

