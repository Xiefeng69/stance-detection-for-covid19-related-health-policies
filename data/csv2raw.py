import os
import pandas as pd
from tqdm import tqdm

stanceMap = {'AGAINST': '0', 'FAVOR': '1', 'NONE': '2'}

def process(file='./SEM16/Atheism.csv'):
    df = pd.read_csv(file)
    tweets = df["Tweet"].values.tolist()
    stance_ = df['Stance'].values.tolist()
    target = df['Target'].values.tolist()
    stance = [stanceMap[value.strip()] for value in stance_]
    datas = []
    assert len(tweets) == len(stance) == len(target)
    for i in tqdm(range(len(tweets))):
        datas.append(tweets[i].strip() + '\n')
        datas.append(' '.join(target[i].strip().split('_')) + '\n')
        datas.append(stance[i].strip() + '\n')
    if 'stay_at_home_orders'.__eq__(target[0].strip()):
        fn = 'SH'
    elif 'Vaccination'.__eq__(target[0].strip()):
        fn = 'VA'
    elif 'face_masks'.__eq__(target[0].strip()):
        fn = 'WM'
    new_F = open('./raw/' + fn + '.raw', 'w', encoding='utf-8', errors='ignore')
    new_F.writelines(datas)
    new_F.close()

if __name__ == '__main__':
    dir = './covid19-policies-stance/'
    files = ['stay_at_home_order_all.csv', 'vaccination_all.csv', 'face_masks_all.csv']
    for file in files:
        process(dir + file)
        print("Done!")