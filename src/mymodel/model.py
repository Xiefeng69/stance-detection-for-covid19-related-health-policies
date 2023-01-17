import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel

from mymodel.layers import *

class MyModel(nn.Module):
    def __init__(self, num_labels, num_topics, hidden=128, backbone='bert_base', geo_encoder='gcn_emb', n_layers_freeze=0, fusion='cls'):
        super(MyModel, self).__init__()

        # bert selection
        if backbone == 'bert_base':
            print('bert: BERT_Base')
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        elif backbone == 'bertweet':
            print('bert: BERTweet')
            self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        elif backbone == 'ct_bert':
            print('bert: Covid-Bert')
            self.bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        else:
            raise Exception('Please choose right BERT model !')
        config = self.bert.config

        # parameters
        self.model = 'MyModel'
        self.hidden = hidden
        self.bert_hidden = config.hidden_size
        self.fusion = fusion
        self.final_hidden = self.bert_hidden*2 + self.hidden
        self.geo_encoder = geo_encoder
        self.l = 2
        self.n = 53

        # building blocks
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if geo_encoder == 'gcn_emb':
            self.adj = torch.Tensor(np.loadtxt(open("data/us-adj.txt"), delimiter=",")).cuda()
            self.geohidden = nn.Parameter(torch.FloatTensor(self.n, self.hidden), requires_grad=True)
            nn.init.xavier_uniform_(self.geohidden)
            self.gcnblocks = nn.ModuleList([GraphConvLayer(in_features=self.hidden, out_features=self.hidden) for i in range(self.l)])
        else:
            self.location_enc = nn.Embedding(self.n, self.hidden)

        self.bridge = nn.Linear(self.bert_hidden, self.bert_hidden)

        self.classifier = nn.Linear(self.final_hidden, num_labels)
        
        self.discriminator = nn.Linear(self.bert_hidden, num_topics)

    def forward(self, x_input, input_key, opt):
        # load input from x_input dict
        input_ids = x_input['input_ids']
        attention_mask = x_input['attention_mask']
        token_type_ids = x_input['token_type_ids']
        location_ids = x_input['location_ids']
        # seen = x_input['seen']  print(seen) => [batchsize]

        # step 1: obtain representation based on BERT encoder
        outputs = self.bert(input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            return_dict = True)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # step 2: Bridge distills policy-specific and policy-transfer features
        policy_specific = self.bridge(pooled_output)
        policy_transfer = pooled_output - policy_specific

        # step 3: generate geolocation encoding
        if self.geo_encoder == 'gcn_emb':
            location_encoding = self.geohidden # [n_region, hidden]
            for layer in self.gcnblocks:
                location_encoding = layer(location_encoding, self.adj)
                location_encoding = self.dropout(location_encoding)
            location_encoding = location_encoding[location_ids]
        else:
            location_encoding = self.location_enc(location_ids)

        # step 4: features fusion module for prediction
        policy_fusion = torch.cat([policy_specific, policy_transfer, location_encoding], dim=-1)

        # prediction layer
        logits = self.classifier(policy_fusion)

        # discriminator layer
        reverse_feature = ReverseLayerF.apply(policy_transfer, opt["p_lambda"])
        sd_labels = self.discriminator(reverse_feature)

        # when phase="test" return geoembedding
        if opt["phase"] == 'test' and self.geo_encoder == 'gcn_emb':
            sd_labels = self.geohidden

        return logits, sd_labels
