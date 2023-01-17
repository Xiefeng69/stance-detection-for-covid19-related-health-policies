import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import pickle

class CrossNet(nn.Module):
    def __init__(self, num_labels, hidden):
        super(CrossNet, self).__init__()
        # loading glove embedding 
        self.embed_glove_path = 'embed_glove_embedding.pkl'
        self.embs_np = pickle.load(open(self.embed_glove_path, 'rb'))
        self.embs_pt = torch.from_numpy(self.embs_np)
        self.glove_hidden = self.embs_pt.shape[1]
        self.hidden = hidden

        # BiCond building block
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embs_np).float())
        self.target_bilstm = nn.LSTM(self.glove_hidden, self.hidden, bidirectional=True, batch_first=True)
        self.text_bilstm = nn.LSTM(self.glove_hidden, self.hidden, bidirectional=True, batch_first=True)
        
        # building block
        self.W = nn.Linear(2*self.hidden, self.hidden)
        self.w = nn.Linear(self.hidden, 1)
        self.act = nn.Tanh()
        self.classifier = nn.Linear(self.hidden*2, num_labels)
        self.softmax = nn.Softmax()

    def forward(self, x_input, input_key, opt):
        input_ids = x_input['input_ids'] # batch maxlen
        target_ids = x_input['target_ids']

        # embedding layer
        document_enc = self.embedding(input_ids) # batch maxlen glove-hidden
        target_enc = self.embedding(target_ids)

        # Context encoding layer
        _, target_last_hn_cn = self.target_bilstm(target_enc) # target encoding
        last_hidden_output, _ = self.text_bilstm(document_enc, target_last_hn_cn) # document encoding
        last_hidden_output = self.dropout(last_hidden_output)
        # print(last_hidden_output.shape) [batchsize, seqlen(maxlen), 2*hidden]

        # Aspect Attention Layer
        coeff = self.W(last_hidden_output)
        coeff = self.act(coeff)
        coeff = self.w(coeff) # [batch, seqlen, 1]

        attn = F.softmax(coeff, dim=-1) # [batch, seqlen, 1]
        output = torch.mul(last_hidden_output, attn) # the second one could be broadcast
        '''
        # same as above
        attn_exp = attn.expand(attn.shape[0], attn.shape[1], 2*self.hidden)
        outputs = torch.mul(last_hidden_output, attn_exp)
        '''

        # Prediction Layer
        output = torch.sum(output, dim=1) # [batchsize, 2*hidden]
        output = self.classifier(output)
        #output = self.softmax(output)

        return output, None