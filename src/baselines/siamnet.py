from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import pickle

class SiamNet(nn.Module):
    def __init__(self, num_labels, hidden):
        super(SiamNet, self).__init__()
        # loading glove embedding 
        self.embed_glove_path = 'embed_glove_embedding.pkl'
        self.embs_np = pickle.load(open(self.embed_glove_path, 'rb'))
        self.embs_pt = torch.from_numpy(self.embs_np)
        self.glove_hidden = self.embs_pt.shape[1]
        self.hidden = hidden

        # building block
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embs_np).float())
        self.bilstm = nn.LSTM(self.glove_hidden, self.hidden, batch_first=True, bidirectional=True)
        self.W = nn.Linear(2*self.hidden, 2*self.hidden)
        self.act = nn.Tanh()
        self.context_vector = nn.Parameter(torch.FloatTensor(2*self.hidden, 1), requires_grad=True) # context vector: randomly initialized and jointly learned
        self.classifier = nn.Linear(1, num_labels)

    def forward(self, x_input, input_key, opt):
        input_ids = x_input['input_ids'] # batch maxlen
        target_ids = x_input['target_ids']
        batch_size = input_ids.shape[0]

        # embedding layer
        target_enc = self.embedding(target_ids)
        document_enc = self.embedding(input_ids) # batch maxlen glove-hidden

        # siamese bilstm
        target_output, _ = self.bilstm(target_enc) # [batchsize, seqlen(maxlen), 2*hidden]
        document_output, _ = self.bilstm(document_enc) # [batchsize, seqlen(maxlen), 2*hidden]

        # context-aware attention mechanism
        context_vector = self.context_vector.expand(batch_size, self.context_vector.shape[0], self.context_vector.shape[1])
        # print(context_vector.shape) # [batchsize, 2*hidden, 1]
        
        target_coeff = self.W(target_output)
        target_coeff = self.act(target_coeff)
        target_coeff = torch.bmm(target_coeff, context_vector) # [batch, seqlen, 1]
        target_attn = F.softmax(target_coeff, dim=-1) # [batch, seqlen, 1]
        target_output = torch.mul(target_output, target_attn) # [batch, seqlen, hidden]
        target_output = torch.sum(target_output, dim=1) # [batchsize, 2*hidden]

        document_coeff = self.W(document_output)
        document_coeff = self.act(document_coeff)
        document_coeff = torch.bmm(document_coeff, context_vector) # [batch, seqlen, 1]
        document_attn = F.softmax(document_coeff, dim=-1) # [batch, seqlen, 1]
        document_output = torch.mul(document_output, document_attn) # [batch, seqlen, hidden]
        document_output = torch.sum(document_output, dim=1) # [batchsize, 2*hidden]

        # similarity-based prediction
        #similarity_cos = F.cosine_similarity(target_output, document_output) # [batchsize]
        similarity_man = torch.abs(target_output-document_output) # [batchsize, 2*hidden]
        similarity_man = -torch.sum(similarity_man, dim=1, keepdim=True) # [batchsize, 1]
        similarity_man = F.softmax(similarity_man, dim=0) # [batchsize, 1]
        #similarity = similarity_cos.unsqueeze(-1)
        similarity = similarity_man
        output = self.classifier(similarity)

        return output, None