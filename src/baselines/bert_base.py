import os
#os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn as nn
from transformers import AutoModel

class BERTBase(nn.Module):
    def __init__(self, num_labels, model='bert_base', n_layers_freeze=0, fusion='cls'):
        super(BERTBase, self).__init__()
        # choose a bert
        if model == 'bert_base':
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        elif model == 'bertweet':
            self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        elif model == 'covid-twitter-bert':
            self.bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        else:
            raise Exception('Please choose right BERT model !')
        config = self.bert.config
        
        # freeze layer
        n_layers = 12 if model != 'covid-twitter-bert' else 24
        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # other building block
        self.hidden = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.model = model
        self.fusion = fusion

    def forward(self, x_input, input_key, opt):
        # load input from x_input dict
        input_ids = x_input['input_ids']
        attention_mask = x_input['attention_mask']
        token_type_ids = x_input['token_type_ids']

        outputs = self.bert(input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            return_dict = True)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits, None
