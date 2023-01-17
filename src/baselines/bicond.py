import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import pickle

class BiCond(nn.Module):
    def __init__(self, num_labels, hidden):
        super(BiCond, self).__init__()
        # loading glove embedding 
        self.embed_glove_path = 'embed_glove_embedding.pkl'
        self.embs_np = pickle.load(open(self.embed_glove_path, 'rb'))
        self.embs_pt = torch.from_numpy(self.embs_np)
        self.glove_hidden = self.embs_pt.shape[1]
        self.hidden = hidden

        # building block
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embs_np).float())
        self.target_bilstm = nn.LSTM(self.glove_hidden, self.hidden, bidirectional=True, batch_first=True)
        self.text_bilstm = nn.LSTM(self.glove_hidden, self.hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden*2, num_labels)

    def forward(self, x_input, input_key, opt):
        input_ids = x_input['input_ids'] # batch maxlen
        target_ids = x_input['target_ids']
        
        document_enc = self.embedding(input_ids) # batch maxlen glove-hidden
        target_enc = self.embedding(target_ids)

        batch_size = document_enc.shape[0]
        max_length = document_enc.shape[1]
        hidden_dim = document_enc.shape[2]
        
        # target encoding
        _, target_last_hn_cn = self.target_bilstm(target_enc)
        # document encoding
        _, (txt_last_hn, c) = self.text_bilstm(document_enc, target_last_hn_cn)
        txt_last_hn = self.dropout(txt_last_hn)
        
        output = txt_last_hn.transpose(0,1).reshape((-1, 2*self.hidden)) # the last output of text_bilstm
        output = self.classifier(output)
        # output = F.tanh(output)
        return output, None
