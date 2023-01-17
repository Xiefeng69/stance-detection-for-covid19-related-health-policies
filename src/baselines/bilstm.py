import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import pickle

class BiLSTM(nn.Module):
    def __init__(self, num_labels, hidden):
        super(BiLSTM, self).__init__()
        # loading glove embedding 
        self.embed_glove_path = 'embed_glove_embedding.pkl'
        self.embs_np = pickle.load(open(self.embed_glove_path, 'rb'))
        self.embs_pt = torch.from_numpy(self.embs_np)
        self.glove_hidden = self.embs_pt.shape[1]
        self.hidden = hidden

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embs_np).float())
        self.bilstm = nn.LSTM(self.glove_hidden, self.hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden*2, num_labels)

    def forward(self, x_input, input_key, opt):
        input_ids = x_input['input_ids'] # [batch, maxlen]
        target_ids = x_input['target_ids']
        
        x = self.embedding(input_ids) # [batch, maxlen, glove-hidden]

        batch_size = x.shape[0]
        max_length = x.shape[1]
        hidden_dim = x.shape[2]

        _, (last_h, last_c) = self.bilstm(x)
        #print(h.shape) #[2, batchsize, hidden]

        output = last_h.transpose(0,1).reshape((-1, 2*self.hidden)) # the last output of Bilstm
        output = self.classifier(output)
        
        return output, None