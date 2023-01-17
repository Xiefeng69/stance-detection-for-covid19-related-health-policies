import os
import socket
import argparse

if __name__ == '__main__':
    # fixed parameterss
    data = 'covid'
    lr = 2e-5
    l2_reg = 5e-5
    wiki_model = 'bert_base'
    n_layers_freeze = 0
    n_layers_freeze_wiki = 10
    gpu = '0'

    # changed parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, choices=(
        'stay_at_home_order,face_masks', 'face_masks,stay_at_home_order', 'face_masks,vaccination', 'vaccination,face_masks', 'vaccination,stay_at_home_order', 'stay_at_home_order,vaccination', 
        'zeroshot,vaccination', 'zeroshot,stay_at_home_order', 'zeroshot,face_masks'
    ), default='stay_at_home_orders,face_masks')
    parser.add_argument('--hidden', type=int, default=128, help="128 or 256 in our experiments")
    parser.add_argument('--p_lambda', type=float, default=0.1, help="the parameter of GRL")
    parser.add_argument('--alpha', type=float, default=0.01, help="the trade-off parameter")
    parser.add_argument('--batch', type=int, default=16, help="batch size")
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=100, help="the maximum length of tokens")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dual', action='store_true', help="the version of WS-BERT")
    parser.add_argument('--backbone', type=str, choices=('bert_base', 'bertweet', 'ct_bert'), default='bert_base')
    parser.add_argument('--model', type=str, 
                        choices=('mymodel', 'tan', 'textcnn', 'ws_bert', 'dan_bert', 'bert_base', 'bertweet', 'ct_bert', 'bilstm', 'bicond', 'crossnet', 'siamnet', 'toad', 
                                 'woadv', 'wogeoenc', 'wgeoemb', 'wobk'), default='mymodel')
    
    args = parser.parse_args()
    topic = args.topic
    batch_size = args.batch
    epochs = args.epoch
    patience = args.patience
    model = args.model
    hidden = args.hidden
    p_lambda = args.p_lambda
    alpha = args.alpha
    backbone = args.backbone
    max_len = args.max_len
    if args.dual is False:
        wsmode = 'single'
    else:
        wsmode = 'dual'
    seed = args.seed
    #seed = 42

    # implemented models
    bert_based = ['mymodel', 'ws_bert', 'dan_bert', 'bert_base', 'bertweet', 'ct_bert', 'woadv', 'wogeoenc', 'wgeoemb', 'wobk']
    glove_based = ['bilstm', 'textcnn', 'tan', 'bicond', 'crossnet', 'siamnet', 'toad']
    if model in bert_based:
        tokenizer_type = 'bert'
    elif model in glove_based:
        tokenizer_type = 'glove'

    # external_knowledge
    if wiki_model == model:
        n_layers_freeze_wiki = n_layers_freeze
    if not wiki_model or wiki_model == model:
        n_layers_freeze_wiki = 0

    os.makedirs('results', exist_ok=True)
    if data != 'vast':
        file_name = f'results/{data}-topic={topic}-lr={lr}-bs={batch_size}.txt'
    else:
        file_name = f'results/{data}-lr={lr}-bs={batch_size}.txt'

    if model != 'bert-base':
        file_name = file_name[:-4] + f'-{model}.txt'
    if n_layers_freeze > 0:
        file_name = file_name[:-4] + f'-n_layers_fz={n_layers_freeze}.txt'
    if wiki_model:
        file_name = file_name[:-4] + f'-wiki={wiki_model}.txt'
    if n_layers_freeze_wiki > 0:
        file_name = file_name[:-4] + f'-n_layers_fz_wiki={n_layers_freeze_wiki}.txt'

    n_gpus = len(gpu.split(','))
    file_name = file_name[:-4] + f'-n_gpus={n_gpus}.txt'

    command = f"python -u src/train.py " \
              f"--data={data} " \
              f"--topic={topic} " \
              f"--model={model} " \
              f"--wiki_model={wiki_model} " \
              f"--n_layers_freeze={n_layers_freeze} " \
              f"--n_layers_freeze_wiki={n_layers_freeze_wiki} " \
              f"--batch_size={batch_size} " \
              f"--epochs={epochs} " \
              f"--patience={patience} " \
              f"--lr={lr} " \
              f"--l2_reg={l2_reg} " \
              f"--gpu={gpu} " \
              f"--tokenizer_type={tokenizer_type} " \
              f"--hidden={hidden} " \
              f"--p_lambda={p_lambda} " \
              f"--alpha={alpha} " \
              f"--backbone={backbone} " \
              f"--max_len={max_len} " \
              f"--wsmode={wsmode} " \
              f"--seed={seed} " \
              # f" > {file_name}"

    print(command)
    os.system(command)
