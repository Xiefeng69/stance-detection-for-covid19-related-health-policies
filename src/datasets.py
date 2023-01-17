import os
#os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from transformers import AutoTokenizer

geo_map = {
    'AL': 0,
    'AK': 1, 
    'AZ': 2, 
    'AR': 3, 
    'CA': 4, 
    'CO': 5,
    'CT': 6, 
    'DE': 7, 
    'FL': 8, 
    'GA': 9, 
    'HI': 10, 
    'ID': 11,
    'IL': 12, 
    'IN': 13, 
    'IA': 14, 
    'KS': 15, 
    'KY': 16, 
    'LA': 17, 
    'ME': 18, 
    'MD': 19,
    'MA': 20, 
    'MI': 21, 
    'MN': 22, 
    'MS': 23, 
    'MO': 24, 
    'MT': 25, 
    'NE': 26, 
    'NV': 27, 
    'NH': 28, 
    'NJ': 29, 
    'NM': 30, 
    'NY': 31,
    'NC': 32, 
    'ND': 33, 
    'OH': 34, 
    'OK': 35, 
    'OR': 36, 
    'PA': 37,
    'RI': 38, 
    'SC': 39, 
    'SD': 40, 
    'TN': 41, 
    'TX': 42, 
    'UT': 43,
    'VT': 44, 
    'VA': 45, 
    'WA': 46, 
    'WV': 47, 
    'WI': 48, 
    'WY': 49, 
    'DC': 50,
    'USA': 51,
    'None': 52
}

description = {
    'stay_at_home_orders': str('Under a stay at home order, all non-essential workers must stay home. People can leave their homes only for essential needs like grocery stores and medicine, or for solo outdoor exercise.'),
    'face_masks': str('Masks are a key measure to reduce transmission and save lives. Wearing well-fitted masks should be used as part of a comprehensive Do it all! approach including maintaining physical distancing, avoiding crowded, closed and close-contact settings, ensuring good ventilation of indoor spaces, cleaning hands regularly, and covering sneezes and coughs with a tissue of bent elbow.'),
    'Vaccination': str('Getting vaccinated could save your life. COVID-19 vaccines provide strong protection against serious illness, hospitalization and death. There is also some evidence that being vaccinated will make it less likely that you will pass the virus on to others, which means your decision to get the vaccine also protects those around you.')
}

class COVIDTweetStance_cross_target(Dataset):
    def __init__(self, phase, topic, model, max_len, backbone='bert_base', wiki_model='bert_base', wsmode='single'):
        path = 'data/covid19-policies-stance'
        cross_topic = False
        if ',' in topic:
            from_topic, to_topic = topic.split(',')
            if phase in ['train', 'val']:
                topic = from_topic
            else:
                topic = to_topic
            cross_topic = True

        if cross_topic and phase == 'test':
            # destination target
            file_paths = [f'{path}/{topic}_{p}.csv' for p in ['train', 'val']]
            dfs = [pd.read_csv(file_path) for file_path in file_paths]
            df = pd.concat(dfs)
            df.reset_index(inplace=True) # pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
            if 'seen?' not in df.columns:
                np_seen = np.zeros((df.shape[0], 1))
                df_seen = pd.DataFrame(np_seen, columns=['seen?'])
                df = pd.concat([df, df_seen], axis=1)
        else:
            # source target
            file_path = f'{path}/{topic}_{phase}.csv'
            df = pd.read_csv(file_path)
            if 'seen?' not in df.columns:
                np_seen = np.ones((df.shape[0], 1))
                df_seen = pd.DataFrame(np_seen, columns=['seen?'])
                df = pd.concat([df, df_seen], axis=1)

        # add unseen content. in here, we add the 'seen?' and 'location' field
        if phase == 'train' and model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
            df_unseen = pd.read_csv(f'{path}/{to_topic}_train_unseen.csv')
            df_unseen['Stance'] = 'UNSEEN' # make stance labels become to 'UNSEEN'
            if 'seen?' not in df_unseen.columns: # add 'seen?' field
                np_seen = np.zeros((df_unseen.shape[0], 1))
                df_seen = pd.DataFrame(np_seen, columns=['seen?'])
                np_location = np.zeros((df_unseen.shape[0], 1))
                df_location = pd.DataFrame(np_location, columns=['Location'])
                df_location['Location'] = 'None'
                df_unseen = pd.concat([df_unseen, df_seen, df_location], axis=1)
            df = pd.concat([df, df_unseen])
        

        print(f'# of {phase} examples: {df.shape[0]}')
        if phase == 'train' and model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
            print(f'# of unseen examples: {df_unseen.shape[0]}')
        print(df)
        
        tweets = df['Tweet'].tolist()
        targets = df['Target'].tolist()
        stances = df['Stance'].map({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2, 'UNSEEN': 3}).tolist()
        seen = df['seen?'].tolist()
        df.replace({"Location": geo_map}, inplace=True)
        location = df['Location'].tolist()

        if model == 'bert_base' or backbone == 'bert_base':
            print('Tokenizer: bert')
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif model == 'bertweet' or backbone == 'bertweet':
            print('Tokenizer: bertweet')
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
        elif model == 'ct_bert' or backbone == 'ct_bert':
            print('Tokenizer: CT-Bert')
            tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

        if model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
            if model != 'wobk':
                max_len_for_description = 50
                df.replace({"Target": description}, inplace=True)
                targets = df['Target'].tolist()
                
                tweets = [tweets[i].split(' ') for i in range(len(tweets))]
                tweets = [tweets[i][:max_len] for i in range(len(tweets))]
                tweets = [' '.join(s) for s in tweets]

                encodings = tokenizer(tweets, targets, max_length=max_len+max_len_for_description, truncation=True, padding='max_length')
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
            elif model == 'wobk':
                encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}

        elif model == 'ws_bert':
            wiki_dict = pickle.load(open(f'{path}/wiki_dict.pkl', 'rb'))
            t = topic
            wiki_summary = wiki_dict[t]
            if wsmode == 'single':
                tokenizer_wiki = tokenizer
                print('wiki tokenizer: same as text')
            else:   # bert tokenizer
                tokenizer_wiki = AutoTokenizer.from_pretrained('bert-base-uncased')
                print('wiki tokenizer: bert')

            if wsmode == 'single': # WS-BERT-Single
                tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, targets)]
                encodings = tokenizer(tweets_targets, [wiki_summary] * df.shape[0], max_length=max_len*2, truncation=True, padding='max_length')
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
            else: # WS-BERT-Dual
                encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
                encodings_wiki = tokenizer_wiki([wiki_summary] * df.shape[0], max_length=max_len, truncation=True, padding='max_length')

        else:
            encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
            encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}

        # encodings for the texts and tweets
        input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long)

        # encodings for wiki summaries
        input_ids_wiki = torch.tensor(encodings_wiki['input_ids'], dtype=torch.long)
        attention_mask_wiki = torch.tensor(encodings_wiki['attention_mask'], dtype=torch.long)

        # stance
        stances = torch.tensor(stances, dtype=torch.long)
        print(f'max len: {input_ids.shape[1]}, max len wiki: {input_ids_wiki.shape[1]}')

        # unseen?
        seen = torch.tensor(seen, dtype=torch.long) # filed "seen=> 1/0" also refer to the "src/dest"

        # location ids
        location_ids = torch.tensor(location, dtype=torch.long) # filed location means the post location of tweet

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.stances = stances
        self.input_ids_wiki = input_ids_wiki
        self.attention_mask_wiki = attention_mask_wiki
        self.seen = seen
        self.location_ids = location_ids

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'token_type_ids': self.token_type_ids[index],
            'stances': self.stances[index],
            'input_ids_wiki': self.input_ids_wiki[index],
            'attention_mask_wiki': self.attention_mask_wiki[index],
            'seen': self.seen[index],
            'location_ids': self.location_ids[index]
        }
        return item

    def __len__(self):
        return self.stances.shape[0]


class COVIDTweetStance_zero_shot(Dataset):
    def __init__(self, phase, topic, model, max_len, backbone='bert_base', wiki_model='bert_base', wsmode='single'):
        path = 'data/covid19-policies-stance'

        all_topics = ['face_masks', 'stay_at_home_order','vaccination']
        src_topic, dest_topic = topic.split(',') # format at "zeroshot,dest_topic", source-target and destination-target
        if phase in ['train', 'val']:
            # dest_topic_id = all_topics.index(dest_topic)
            all_topics.remove(dest_topic)
            topic = all_topics
            topic_paths = [f'{path}/{t}_{phase}.csv' for t in topic]
            topic_contents = list()
            for f in range(len(topic_paths)):
                df = pd.read_csv(topic_paths[f])
                if 'seen?' not in df.columns:
                    np_seen = np.full((df.shape[0], 1),f+1)
                    df_seen = pd.DataFrame(np_seen, columns=['seen?'])
                    df = pd.concat([df, df_seen], axis=1)
                topic_contents.append(df)
            
            df = pd.concat(topic_contents, axis=0)

        else:
            topic = dest_topic
            file_paths = [f'{path}/{topic}_{p}.csv' for p in ['train', 'val']]
            dfs = [pd.read_csv(file_path) for file_path in file_paths]
            df = pd.concat(dfs, ignore_index=True)
            if 'seen?' not in df.columns:
                np_seen = np.zeros((df.shape[0], 1))
                df_seen = pd.DataFrame(np_seen, columns=['seen?'])
                df = pd.concat([df, df_seen], axis=1)

        # add unseen content. in here, we add the 'seen?' and 'location' field
        if phase == 'train' and model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
            df_unseen = pd.read_csv(f'{path}/{dest_topic}_train_unseen.csv')
            df_unseen['Stance'] = 'UNSEEN' # make stance labels become to 'UNSEEN'
            if 'seen?' not in df_unseen.columns: # add 'seen?' field
                np_seen = np.zeros((df_unseen.shape[0], 1))
                df_seen = pd.DataFrame(np_seen, columns=['seen?'])
                np_location = np.zeros((df_unseen.shape[0], 1))
                df_location = pd.DataFrame(np_location, columns=['Location'])
                df_location['Location'] = 'None'
                df_unseen = pd.concat([df_unseen, df_seen, df_location], axis=1)
            df = pd.concat([df, df_unseen])

        print(f'# of {phase} examples: {df.shape[0]}')
        if phase == 'train' and model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
            print(f'# of unseen examples: {df_unseen.shape[0]}')
        print(df)

        tweets = df['Tweet'].tolist()
        targets = df['Target'].tolist()
        stances = df['Stance'].map({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2, 'UNSEEN': 3}).tolist()
        seen = df['seen?'].tolist()
        df.replace({"Location": geo_map}, inplace=True)
        location = df['Location'].tolist()

        if model == 'bert_base' or backbone == 'bert_base':
            print('Tokenizer: bert')
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif model == 'bertweet' or backbone == 'bertweet':
            print('Tokenizer: bertweet')
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
        elif model == 'ct_bert' or backbone == 'ct_bert':
            print('Tokenizer: CT-Bert')
            tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        
        
        if model in ['mymodel', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
            if model != 'wobk':
                max_len_for_description = 50
                df.replace({"Target": description}, inplace=True)
                targets = df['Target'].tolist()
                
                tweets = [tweets[i].split(' ') for i in range(len(tweets))]
                tweets = [tweets[i][:max_len] for i in range(len(tweets))]
                tweets = [' '.join(s) for s in tweets]

                encodings = tokenizer(tweets, targets, max_length=max_len+max_len_for_description, truncation=True, padding='max_length')
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
            elif model == 'wobk':
                encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
        
        elif model == 'ws_bert':
            if wiki_model:
                wiki_dict = pickle.load(open(f'{path}/wiki_dict.pkl', 'rb'))
                t = topic
                wiki_summary = df['Target'].map(wiki_dict).tolist() # do not need the [wiki_summary] * df.shape[0], because the size is equal to df.shape[0]
                #print(wiki_summary)

                if wsmode == 'single': # WS-BERT-Single
                    tokenizer_wiki = tokenizer
                    print('wiki tokenizer: same as text')
                else:   # bert tokenizer
                    tokenizer_wiki = AutoTokenizer.from_pretrained('bert-base-uncased')
                    print('wiki tokenizer: bert')

                if wsmode == 'single': # WS-BERT-Single
                    tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, targets)]
                    encodings = tokenizer(tweets_targets, wiki_summary, max_length=max_len*2, truncation=True, padding='max_length')
                    encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
                else: # WS-BERT-Dual
                    encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
                    encodings_wiki = tokenizer_wiki(wiki_summary, max_length=max_len, truncation=True, padding='max_length')

            else:
                encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
        else:
            encodings = tokenizer(tweets, targets, max_length=max_len, truncation=True, padding='max_length')
            encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}

        # encodings for the texts and tweets
        input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long)

        # encodings for wiki summaries
        input_ids_wiki = torch.tensor(encodings_wiki['input_ids'], dtype=torch.long)
        attention_mask_wiki = torch.tensor(encodings_wiki['attention_mask'], dtype=torch.long)

        # stance
        stances = torch.tensor(stances, dtype=torch.long)
        print(f'max len: {input_ids.shape[1]}, max len wiki: {input_ids_wiki.shape[1]}')

        # unseen?
        seen = torch.tensor(seen, dtype=torch.long) # filed "seen=> 1/0" also refer to the "src/dest"

        # location ids
        location_ids = torch.tensor(location, dtype=torch.long) # filed location means the post location of tweet
        
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.stances = stances
        self.input_ids_wiki = input_ids_wiki
        self.attention_mask_wiki = attention_mask_wiki
        self.seen = seen
        self.location_ids = location_ids

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'token_type_ids': self.token_type_ids[index],
            'stances': self.stances[index],
            'input_ids_wiki': self.input_ids_wiki[index],
            'attention_mask_wiki': self.attention_mask_wiki[index],
            'seen': self.seen[index],
            'location_ids': self.location_ids[index]
        }
        return item

    def __len__(self):
        return self.stances.shape[0]


def data_loader(data, phase, topic, batch_size, model='bert_base', backbone='bert_base', wiki_model='', max_len=128, wsmode='single'):
    '''
        data: Covid-19
        phase: train / val / test
        topic_type: cross-target / zero-shot / few-shot
    '''

    first_topic, second_topic = topic.split(',')
    if first_topic == 'zeroshot':
        dataset = COVIDTweetStance_zero_shot(phase, topic, model, max_len, backbone, wiki_model=wiki_model, wsmode=wsmode)
    elif first_topic == 'fewshot':
        dataset = COVIDTweetStance_zero_shot(phase, topic, model, max_len, backbone, wiki_model=wiki_model, wsmode=wsmode)
    else:
        dataset = COVIDTweetStance_cross_target(phase, topic, model, max_len, backbone, wiki_model=wiki_model, wsmode=wsmode)

    shuffle = True if phase == 'train' else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
