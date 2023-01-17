from cmath import phase
import os
from statistics import mean
from mymodel import model
import numpy as np
import torch
torch.cuda.current_device()
import torch.nn as nn
import copy
import pickle
import time
from transformers import AdamW

from datasets import data_loader
from datasets_glove import data_loader_glove

# import all baseline models
import baselines.all_baselines as md
from mymodel.model import MyModel
from mymodel.ablation import WoAdv, WOBK, WOGeoenc, WGeoemb

class Engine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        print('Preparing data.... \n')

        if args.tokenizer_type == 'bert':

            print('****************************************************************')
            print('Training data based on AutoTokenizer....')
            print('**************************************************************** \n')
            train_loader = data_loader(args.data, 'train', args.topic, args.batch_size, model=args.model, backbone=args.backbone, wiki_model=args.wiki_model, max_len = args.max_len, wsmode=args.wsmode)
            
            print('****************************************************************')
            print('Val data....')
            print('**************************************************************** \n')
            val_loader = data_loader(args.data, 'val', args.topic, 2*args.batch_size, model=args.model, backbone=args.backbone, wiki_model=args.wiki_model, max_len = args.max_len, wsmode=args.wsmode)
            
            print('****************************************************************')
            print('Test data....')
            print('**************************************************************** \n')
            test_loader = data_loader(args.data, 'test', args.topic, 2*args.batch_size, model=args.model, backbone=args.backbone, wiki_model=args.wiki_model, max_len = args.max_len, wsmode=args.wsmode)
            
            print('\n Done\n')
        
        elif args.tokenizer_type == 'glove':

            print('****************************************************************')
            print('Training data based on GloVe....')
            print('**************************************************************** \n')
            train_loader = data_loader_glove(args.data, 'train', args.topic, args.batch_size)
            
            print('****************************************************************')
            print('Val data....')
            print('**************************************************************** \n')
            val_loader = data_loader_glove(args.data, 'val', args.topic, 2*args.batch_size)
            
            print('****************************************************************')
            print('Test data....')
            print('**************************************************************** \n')
            test_loader = data_loader_glove(args.data, 'test', args.topic, 2*args.batch_size)
            
            print('Done\n')

        print('Initializing model....')
        num_labels = 2 if args.data == 'pstance' else 3
        first_topic, second_topic = args.topic.split(',')
        num_topics = 3 if first_topic == 'zeroshot' else 2

        # selecting models
        if args.model in ['ws_bert', 'bilstm', 'textcnn', 'bicond', 'crossnet', 'tan', 'siamnet', 'bert_base', 'ct_bert', 'dan_bert', 'bertweet', 'toad', 'mymodel']:
            if args.model == 'ws_bert':
                model = md.WSBert(num_labels=num_labels, model=args.backbone, wsmode=args.wsmode)
            elif args.model == 'bilstm':
                model = md.BiLSTM(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'textcnn':
                model = md.TextCNN(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'bicond':
                model = md.BiCond(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'crossnet':
                model = md.CrossNet(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'tan':
                model = md.TAN(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'siamnet':
                model = md.SiamNet(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'bert_base':
                model = md.BERTBase(num_labels=num_labels, model='bert_base')
            elif args.model == 'bertweet':
                model = md.BERTBase(num_labels=num_labels, model='bertweet')
            elif args.model == 'ct_bert':
                model = md.BERTBase(num_labels=num_labels, model='covid-twitter-bert')
            elif args.model == 'dan_bert':
                model = md.DAN_Bert(num_labels=num_labels)
            elif args.model == 'toad':
                model = md.TOAD(num_labels=num_labels, hidden=args.hidden)
            elif args.model == 'mymodel':
                model = MyModel(num_labels=num_labels, num_topics=num_topics, hidden=args.hidden, backbone=args.backbone)
        else:
            if args.model == 'woadv':
                model = WoAdv(num_labels=num_labels, hidden=args.hidden, backbone=args.backbone)
            if args.model == 'wogeoenc': 
                model = WOGeoenc(num_labels=num_labels, num_topics=num_topics, hidden=args.hidden, backbone=args.backbone)
            if args.model == 'wgeoemb':
                model = WGeoemb(num_labels=num_labels, num_topics=num_topics, hidden=args.hidden, backbone=args.backbone)
            if args.model == 'wobk':
                model = WOBK(num_labels=num_labels, num_topics=num_topics, hidden=args.hidden, backbone=args.backbone)

        #model = nn.DataParallel(model)
        print("\n ********** model's architecture ********** \n", model)
        model.to(device)
        paramnum = sum([param.nelement() for param in model.parameters()])
        print("\n ********** model's parameter size: %.2fM ********** \n" % (paramnum/1e6))

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        criterion = nn.CrossEntropyLoss(ignore_index=3) # CELoss @input: predict value [batch, class_num] and ground-truth [batch]; ignore_index is used to ignore some classes in ground-truth that do not need to participate in the calculation
        criterion_des = nn.CrossEntropyLoss()

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion_des = criterion_des
        self.args = args

    def train(self):
        best_epoch = 0
        best_epoch_f1 = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        epoch_time_list = []
        for epoch in range(self.args.epochs):
            print(f"{'*' * 30}Epoch: {epoch + 1}{'*' * 30}")
            epoch_start_time = time.time()
            loss = self.train_epoch()
            epoch_time = time.time() - epoch_start_time
            epoch_time_list.append(epoch_time)
            f1, f1_favor, f1_against, f1_neutral, f1m = self.eval('val')
            if f1 > best_epoch_f1:
                best_epoch = epoch
                best_epoch_f1 = f1
                best_state_dict = copy.deepcopy(self.model.state_dict())
            print(f'Epoch: {epoch+1}\tTrain Loss: {loss:.3f}\tVal F1: {f1:.3f}\n'
                  f'Val F1_favor: {f1_favor:.3f}\tVal F1_against: {f1_against:.3f}\tVal F1_Neutral: {f1_neutral:.3f}\tVal F1_micro_macro: {f1m:.3f}\n'
                  f'Best Epoch: {best_epoch+1}\tBest Epoch Val F1: {best_epoch_f1:.3f}\n'
                  f'epoch training time: {epoch_time:.3f}s\n')
            if epoch - best_epoch >= self.args.patience:
                break

        print('Saving the best checkpoint....')
        # torch.save(self.model.state_dict(), 'ckp/model.pt')   # uncomment this line to save the ckp
        self.model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, f"ckp/model_{self.args.data}.pt")

        inference_start_time = time.time()
        f1_avg, f1_favor, f1_against, f1_neutral, f1m = self.eval('test')
        inference_time = time.time() - inference_start_time
        print('-------------------------------------\n')
        print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
              f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}\tTest F1_micro_macro: {f1m:.3f}\n'
              f'inference time: {inference_time:.3f}s\n'
              f'mean epoch time: {mean(epoch_time_list):.3f}s')
        # record results
        with open(f'results/{self.args.model}.csv', 'a', encoding='utf-8') as file:
            file.write('\n')
            if self.args.model == 'ws_bert':
                modelname = f"{self.args.model}-{self.args.wsmode}"
            else:
                modelname = self.args.model
            if self.args.model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
                file.write(f"{self.args.model}, {'-'.join(self.args.topic.split(','))}, {f1_avg:.3f}, {f1m:.3f}, {self.args.hidden}, {self.args.p_lambda}, {self.args.alpha}, {self.args.seed}, {self.args.batch_size}, {f1_favor:.3f}, {f1_against:.3f}, {f1_neutral:.3f}, {self.args.lr}")
            else:
                file.write(f"{modelname}, {'-'.join(self.args.topic.split(','))}, {f1_avg:.3f}, {f1m:.3f}, {self.args.hidden}, {self.args.seed}, {self.args.batch_size}, {f1_favor:.3f}, {f1_against:.3f}, {f1_neutral:.3f}, {self.args.backbone}, {self.args.lr}")
        print('\n-------------------------------------\n')

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            stances = copy.deepcopy(batch["stances"]).to(self.device) # ground truth
            batch.pop("stances")
            input_key = batch.keys()
            for key in input_key:
                batch[key] = batch[key].to(self.device) # input dict
            if self.args.model in ['mymodel', 'wogeoenc', 'wgeoemb', 'wobk']:
                seen = copy.deepcopy(batch['seen'])
            logits, sd_labels = self.model(
                                    x_input=batch, 
                                    input_key=input_key,
                                    opt={
                                        'use_external':self.args.wiki_model and self.args.wiki_model != self.args.model,
                                        'p_lambda': self.args.p_lambda, # GRL
                                        'phase': phase
                                    }
                                )
            loss = self.criterion(logits, stances)
            if self.args.model in ['mymodel', 'wogeoenc', 'wgeoemb', 'wobk']:
                loss_discriminator = self.criterion_des(sd_labels, seen)
                loss += self.args.alpha * loss_discriminator
            loss.backward()
            self.optimizer.step()

            interval = max(len(self.train_loader)//10, 1)
            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f'Batch: {i+1}/{len(self.train_loader)}\tLoss:{loss.item():.3f}')

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def eval(self, phase='val'):
        self.model.eval()
        y_pred = []
        y_true = []
        val_loader = self.val_loader if phase == 'val' else self.test_loader
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                labels = copy.deepcopy(batch["stances"]).to(self.device) # ground truth
                batch.pop("stances")
                input_key = batch.keys()
                for key in input_key:
                    batch[key] = batch[key].to(self.device) # input dict
                logits, opt = self.model(
                                    x_input=batch, 
                                    input_key=input_key,
                                    opt={
                                        'use_external':self.args.wiki_model and self.args.wiki_model != self.args.model,
                                        'p_lambda': 0.0, # GRL
                                        'phase': phase
                                    }
                                )
                preds = logits.argmax(dim=1)
                y_pred.append(preds.detach().to('cpu').numpy())
                y_true.append(labels.detach().to('cpu').numpy())

        if phase == 'test'and self.args.model == 'mymodel':
            print('save geo embedding...')
            opt = opt.detach().to('cpu').numpy()
            np.save('geoencoding', opt)

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        from sklearn.metrics import f1_score
        f1_score_list = f1_score(y_true, y_pred, average=None)
        if len(f1_score_list) == 2:
            f1_against = f1_score_list[0]
            f1_favor = f1_score_list[1]
            f1_neutral = 0
        else:
            f1_against, f1_favor, f1_neutral = f1_score_list

        f1_avg = (f1_favor + f1_against + f1_neutral) / 3
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1m = (f1_macro + f1_micro) * 0.5

        return f1_avg, f1_favor, f1_against, f1_neutral, f1m
