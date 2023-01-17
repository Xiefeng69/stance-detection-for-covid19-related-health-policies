from numpy import single
import torch.nn as nn
import torch
import os


class WSBert(nn.Module):
    def __init__(self, num_labels, model='bert_base', n_layers_freeze=0, wiki_model='bert_base', n_layers_freeze_wiki=10, wsmode=single):
        super(WSBert, self).__init__()
        self.model = model
        self.wiki_model = wiki_model
        self.wsmode = wsmode

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel
        if model == 'bert_base':
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        elif model == 'bertweet':
            self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        elif model == 'ct_bert':  # covid-twitter-bert
            self.bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

        n_layers = 12 if model != 'ct_bert' else 24
        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # Single or Dual
        if self.wiki_model:
            if self.wsmode == 'single':
                self.bert_wiki = self.bert # WS-BERT-Single
                print('mode: WS-BERT-Single')
            else:  # bert-base
                self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased') # WS-BERT-Dual
                print('mode: WS-BERT-Dual')

            n_layers = 12
            if n_layers_freeze_wiki > 0:
                n_layers_ft = n_layers - n_layers_freeze_wiki
                for param in self.bert_wiki.parameters():
                    param.requires_grad = False
                for param in self.bert_wiki.pooler.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                    for param in self.bert_wiki.encoder.layer[i].parameters():
                        param.requires_grad = True

        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.wiki_model and self.wsmode == 'dual':
            hidden = config.hidden_size + self.bert_wiki.config.hidden_size
        else:
            hidden = config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, x_input, input_key, opt, input_ids_wiki=None):
        # print(input_ids.shape) print(attention_mask.shape) print(token_type_ids.shape) [32,128]
        input_ids = x_input['input_ids']
        attention_mask = x_input['attention_mask']
        token_type_ids = x_input['token_type_ids']
        # load wiki information
        if self.wsmode == 'dual': # Dual
            input_ids_wiki = x_input['input_ids_wiki']
            attention_mask_wiki = x_input['attention_mask_wiki']

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        if input_ids_wiki is not None:
            outputs_wiki = self.bert_wiki(input_ids_wiki, attention_mask=attention_mask_wiki)
            pooled_output_wiki = outputs_wiki.pooler_output
            pooled_output_wiki = self.dropout(pooled_output_wiki)
            pooled_output = torch.cat((pooled_output, pooled_output_wiki), dim=1)
        #print(pooled_output.shape)
        logits = self.classifier(pooled_output)
        
        return logits, None