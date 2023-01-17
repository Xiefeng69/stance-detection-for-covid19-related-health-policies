import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import TweetTokenizer
import torchtext
import torchtext.vocab as vocab
from torchtext.vocab import Vectors
from tqdm import tqdm


class GloveDataLoader(Dataset):
    def __init__(self, phase, topic):
        max_length = 100
        path = 'data/covid19-policies-stance'
        cross_topic = False
        tokenizer = TweetTokenizer() # tokenizer: breaks sentences into a list of words
        tokenizer_target = lambda x: x.split('_')
        self.vocab, self.embed = self.GloveEmbedding()
        origin_topic = topic # for save

        if ',' in topic:
            from_topic, to_topic = topic.split(',')
            if phase in ['train', 'val']:
                topic = from_topic
            else:
                topic = to_topic
            cross_topic = True

        if cross_topic and phase == 'test':
            file_paths = [f'{path}/{topic}_{p}.csv' for p in ['train', 'val']]
            dfs = [pd.read_csv(file_path) for file_path in file_paths]
            df = pd.concat(dfs)
        else:
            file_path = f'{path}/{topic}_{phase}.csv'
            df = pd.read_csv(file_path)

        print(f'# of {phase} examples: {df.shape[0]}')
        print(df)

        # handle target
        targets = df['Target'].tolist()
        targets = [tokenizer_target(text) for text in targets]
        targets = self.mapping_tokens(targets, maxlength = 5, topics=origin_topic, phase=phase, types='target')

        # handle tweet text
        tweets = self.normalize_text(df['Tweet'])
        tweets = tweets.tolist()
        tweets = [tokenizer.tokenize(text) for text in tweets]
        tweets = self.mapping_tokens(tweets, maxlength = max_length, topics=origin_topic, phase=phase, types='tweet')

        # handle stance
        stances = df['Stance'].map({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}).tolist()
        
        # instantiate
        self.stances = torch.tensor(stances, dtype=torch.long)
        self.input_ids = torch.tensor(tweets, dtype=torch.long)
        self.target_ids = torch.tensor(targets, dtype=torch.long)    

    def normalize_text(self, text):
        text = text.str.lower() # lowercase
        text = text.str.replace(r"\#", "", regex=True) # replaces hashtags
        text = text.str.replace(r"http\S+", "URL", regex=True)  # remove URL addresses
        text = text.str.replace(r"@", "", regex=True)
        text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ", regex=True)
        text = text.str.replace("\s{2,}", " ", regex=True)
        return text

    def mapping_tokens(self, source, maxlength, topics, phase, types='tweet'):
        source_file_path = f'raw/{types}_{topics}_{phase}_tokens2id.pkl'
        if os.path.exists(source_file_path):
            print(f'load exist mapping file ...')
            source = pickle.load(open(source_file_path, 'rb'))
        else:
            print(f'mapping {types} words ...')
            for i in tqdm(range(len(source))):
                for j in range(len(source[i])):
                    source[i][j] = np.where(self.vocab == source[i][j])[0]
                    if len(source[i][j]) == 0:
                        source[i][j] = 1
                    else:
                        source[i][j] = source[i][j][0]
                if len(source[i]) < maxlength:
                    source[i] = source[i] + [0]*(maxlength - len(source[i])) # postprocess + <pad> 0
                elif len(source[i]) > maxlength:
                    source[i] = source[i][0:maxlength]
            pickle.dump(source, open(source_file_path, 'wb'))
        return source

    def GloveEmbedding(self, fname='data/glove.6B.100d.txt'):
        embedding_file_name = 'glove_embedding.pkl'
        if os.path.exists(f'vocab_{embedding_file_name}'):
            print('loading exist word vectors ...')
            vocab_np = pickle.load(open(f'vocab_{embedding_file_name}', 'rb'))
            embed_np = pickle.load(open(f'embed_{embedding_file_name}', 'rb'))
        else:
            print('loading word vectors ...')
            vocab, embeddings = [], []
            with open(fname, 'rt', encoding='utf-8') as f:
                full_content = f.read().strip().split('\n')
            for line in range(len(full_content)):
                i_word = full_content[line].split(' ')[0]
                i_embedding = [float(val) for val in full_content[line].split(' ')[1:]]
                vocab.append(i_word)
                embeddings.append(i_embedding)
            vocab_np = np.array(vocab)
            embed_np = np.array(embeddings)
            vocab_np = np.insert(vocab_np, 0, '<pad>')
            vocab_np = np.insert(vocab_np, 1, '<unk>')
            pad_emb_npa = np.zeros((1, embed_np.shape[1]))
            unk_emb_npa = np.mean(embed_np, axis=0, keepdims=True)
            embed_np = np.vstack((pad_emb_npa, unk_emb_npa, embed_np))
            pickle.dump(vocab_np, open(f'vocab_{embedding_file_name}', 'wb'))
            pickle.dump(embed_np, open(f'embed_{embedding_file_name}', 'wb'))
        # print(embed_np.shape) (400002, 200)
        return vocab_np, embed_np

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'target_ids': self.target_ids[index],
            'stances': self.stances[index],
        }
        return item

    def __len__(self):
        return self.stances.shape[0]

class GloveDataLoader_zero_shot(Dataset):
    def __init__(self, phase, topic):
        max_length = 100
        path = 'data/covid19-policies-stance'
        cross_topic = False
        tokenizer = TweetTokenizer() # tokenizer: breaks sentences into a list of words
        tokenizer_target = lambda x: x.split('_')
        self.vocab, self.embed = self.GloveEmbedding()
        origin_topic = topic # for save

        all_topics = ['face_masks', 'stay_at_home_order', 'vaccination']
        src_topic, dest_topic = topic.split(',') # format at "zeroshot, dest_topic", source-target and destination-target
        if phase in ['train', 'val']:
            # dest_topic_id = all_topics.index(dest_topic)
            all_topics.remove(dest_topic)
            topic = all_topics
            topic_paths = [f'{path}/{t}_{phase}.csv' for t in topic]
            topic_contents = list()
            for f in range(len(topic_paths)):
                df = pd.read_csv(topic_paths[f])
                topic_contents.append(df)
            df = pd.concat(topic_contents, axis=0)

        else:
            topic = dest_topic
            file_paths = [f'{path}/{topic}_{p}.csv' for p in ['train', 'val']]
            dfs = [pd.read_csv(file_path) for file_path in file_paths]
            df = pd.concat(dfs)

        print(f'# of {phase} examples: {df.shape[0]}')
        print(df)

        # handle tweet text
        tweets = self.normalize_text(df['Tweet'])
        tweets = tweets.tolist()
        tweets = [tokenizer.tokenize(text) for text in tweets]
        tweets = self.mapping_tokens(tweets, maxlength = max_length, topics=origin_topic, phase=phase, types='tweet')

        # handle target
        targets = df['Target'].tolist()
        targets = [tokenizer_target(text) for text in targets]
        targets = self.mapping_tokens(targets, maxlength = 5, topics=origin_topic, phase=phase, types='target')

        # handle stance
        stances = df['Stance'].map({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}).tolist()
        
        # instantiate
        self.stances = torch.tensor(stances, dtype=torch.long)
        self.input_ids = torch.tensor(tweets, dtype=torch.long)
        self.target_ids = torch.tensor(targets, dtype=torch.long)    

    def normalize_text(self, text):
        text = text.str.lower() # lowercase
        text = text.str.replace(r"\#", "", regex=True) # replaces hashtags
        text = text.str.replace(r"http\S+", "URL", regex=True)  # remove URL addresses
        text = text.str.replace(r"@", "", regex=True)
        text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ", regex=True)
        text = text.str.replace("\s{2,}", " ", regex=True)
        return text

    def mapping_tokens(self, source, maxlength, topics, phase, types='tweet'):
        source_file_path = f'raw/{types}_{topics}_{phase}_tokens2id.pkl'
        if os.path.exists(source_file_path):
            print(f'load exist mapping file ...')
            source = pickle.load(open(source_file_path, 'rb'))
        else:
            print(f'mapping {types} words ...')
            for i in tqdm(range(len(source))):
                for j in range(len(source[i])):
                    source[i][j] = np.where(self.vocab == source[i][j])[0]
                    if len(source[i][j]) == 0: #返回形式为元组
                        source[i][j] = 1
                    else:
                        source[i][j] = source[i][j][0]
                if len(source[i]) < maxlength:
                    source[i] = source[i] + [0]*(maxlength - len(source[i])) # postprocess + <pad> 0
                elif len(source[i]) > maxlength:
                    source[i] = source[i][0:maxlength]
            pickle.dump(source, open(source_file_path, 'wb'))
        return source

    def GloveEmbedding(self, fname='data/glove.6B.100d.txt'):
        embedding_file_name = 'glove_embedding.pkl'
        if os.path.exists(f'vocab_{embedding_file_name}'):
            print('loading exist word vectors ...')
            vocab_np = pickle.load(open(f'vocab_{embedding_file_name}', 'rb'))
            embed_np = pickle.load(open(f'embed_{embedding_file_name}', 'rb'))
        else:
            print('loading word vectors ...')
            vocab, embeddings = [], []
            with open(fname, 'rt') as f:
                full_content = f.read().strip().split('\n')
            for line in range(len(full_content)):
                i_word = full_content[line].split(' ')[0]
                i_embedding = [float(val) for val in full_content[line].split(' ')[1:]]
                vocab.append(i_word)
                embeddings.append(i_embedding)
            vocab_np = np.array(vocab)
            embed_np = np.array(embeddings)
            vocab_np = np.insert(vocab_np, 0, '<pad>')
            vocab_np = np.insert(vocab_np, 1, '<unk>')
            pad_emb_npa = np.zeros((1, embed_np.shape[1]))
            unk_emb_npa = np.mean(embed_np, axis=0, keepdims=True)
            embed_np = np.vstack((pad_emb_npa, unk_emb_npa, embed_np))
            pickle.dump(vocab_np, open(f'vocab_{embedding_file_name}', 'wb'))
            pickle.dump(embed_np, open(f'embed_{embedding_file_name}', 'wb'))
        # print(embed_np.shape) (400002, 200)
        return vocab_np, embed_np

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'target_ids': self.target_ids[index],
            'stances': self.stances[index],
        }
        return item

    def __len__(self):
        return self.stances.shape[0]

def data_loader_glove(data, phase, topic, batch_size):
    '''
        data: Covid-19
        phase: train / val / test
        topic: single-target / cross-target
    '''

    first_topic, second_topic = topic.split(',')
    if first_topic == 'zeroshot':
        dataset = GloveDataLoader_zero_shot(phase, topic)
    elif first_topic == 'fewshot':
        dataset = GloveDataLoader_zero_shot(phase, topic)
    else:
        dataset = GloveDataLoader(phase, topic)

    shuffle = True if phase == 'train' else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader