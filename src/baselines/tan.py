import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import pickle

class TAN(nn.Module):
    def __init__(self, num_labels, hidden):
        super(TAN, self).__init__()
        # loading glove embedding 
        self.embed_glove_path = 'embed_glove_embedding.pkl'
        self.embs_np = pickle.load(open(self.embed_glove_path, 'rb'))
        self.embs_pt = torch.from_numpy(self.embs_np)
        self.glove_hidden = self.embs_pt.shape[1]
        self.hidden = hidden

        # building block
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embs_np).float())
        self.birnn = nn.RNN(self.glove_hidden, self.hidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*self.glove_hidden, 1)
        self.classifier = nn.Linear(self.hidden*2, num_labels)

    def forward(self, x_input, input_key, opt):
        input_ids = x_input['input_ids'] # batch maxlen
        target_ids = x_input['target_ids']
        
        # embedding layer
        document_enc = self.embedding(input_ids) # batch maxlen glove-hidden
        target_enc = self.embedding(target_ids)

        # Bi-RNN
        last_hidden_output, _ = self.birnn(document_enc) # document-encoding
        last_hidden_output = self.dropout(last_hidden_output)

        # Target-augmented embedding
        target_enc = torch.mean(target_enc, dim=1) # [batchsize, n_words, hidden] => [batchsize, hidden]
        target_enc = target_enc.unsqueeze(1)
        target_enc = target_enc.expand(target_enc.shape[0], document_enc.shape[1] ,target_enc.shape[2])
        aug_emb = torch.cat([document_enc, target_enc], dim=-1) # [batchsize, n_words, 2*hidden]

        # Linear
        context_emb = self.linear(aug_emb)
        context_score = F.softmax(context_emb, dim=-1)
        # print('context', context_score.shape) [batchsize words 1]

        # Inner Product
        output = torch.mul(last_hidden_output, context_score)
        # print(output.shape) [batchsize, words, hidden]

        # classification
        output = torch.sum(output, dim=1) # [batchsize, 2*hidden]
        output = self.classifier(output)

        return output, None