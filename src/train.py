from yaml import parse
from engine import Engine

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='covid', help='which dataset to use')
    parser.add_argument('--tokenizer_type', type=str, default='bert', choices=('bert', 'glove'))
    parser.add_argument('--topic', type=str, help='the topic to use')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2_reg', type=float, default=5e-5)
    parser.add_argument('--max_grad', type=float, default=0)
    parser.add_argument('--n_layers_freeze', type=int, default=0)
    parser.add_argument('--model', type=str)
    parser.add_argument('--wiki_model', type=str, choices=('', 'bert_base'), default='')
    parser.add_argument('--n_layers_freeze_wiki', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--p_lambda', type=float, default=0.1, help="the parameter of GRL")
    parser.add_argument('--alpha', type=float, default=0.01, help="trade-off parameter")
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--wsmode', type=str, default='single')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    engine = Engine(args)
    engine.train()