import pickle
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# according to the implementation: https://github.com/finisky/TextCNN

class TextCNN(nn.Module):
    def __init__(self, num_labels, hidden):
        super().__init__()
        # loadding glove embedding
        self.embed_glove_path = 'embed_glove_embedding.pkl'
        self.embs_np = pickle.load(open(self.embed_glove_path, 'rb'))
        self.embs_pt = torch.from_numpy(self.embs_np)
        self.glove_hidden = self.embs_pt.shape[1]
        self.hidden = hidden
        self.maxlen = 100

        # parameter is following the paper: we use rectified linear units, filter windows (h) of 3, 4, 5 with 100 feature maps each.
        input_channel = 1
        output_channel = 100
        kernel_sizes = [3,4,5]

        self.embedding = nn.Embedding.from_pretrained(self.embs_pt.float())
        self.convs = nn.ModuleList([nn.Conv2d(input_channel, output_channel, (s, self.maxlen)) for s in kernel_sizes])
        self.q = nn.Linear(len(kernel_sizes)*output_channel, len(kernel_sizes)*output_channel)
        self.k = nn.Linear(len(kernel_sizes)*output_channel, len(kernel_sizes)*output_channel)
        self.v = nn.Linear(len(kernel_sizes)*output_channel, len(kernel_sizes)*output_channel)
        self.classifier = nn.Linear(len(kernel_sizes)*output_channel, num_labels)
    
    def forward(self, x_input, input_key, opt):
        input_ids = x_input['input_ids'] # [batch, maxlen]
        target_ids = x_input['target_ids']
        
        x = self.embedding(input_ids) # [batch, maxlen, glove-hidden]
        x = x.unsqueeze(1) # [batch, 1, maxlen, glove-hidden]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [batch, output_channel, len-after-cnn] * len(kernel_sizes)

        # F.max_pool1d(tensor, 1d-dimension)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [batch, output_channel] * len(kernel_sizes)
        x = torch.cat(x, dim=1)

        # self-attention
        query = self.q(x)
        key = self.k(x)
        attn = torch.mm(query, key.T)
        x = torch.mm(attn, self.v(x))

        output = self.classifier(x)
        
        return output, None