[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=Xiefeng69.stance-detection-for-covid19-related-health-policies
[repo-url]: https://github.com/Xiefeng69/stance-detection-for-covid19-related-health-policies

# Stance Detection for COVID-19-related Health Policies

[DASFAA2023] The source codes and datasets for paper: `Adversarial Learning-based Stance Classifier for COVID-19-related Health Policies`.

[![visitors][visitors-img]][repo-url]

## Installation
For embedding-based methods, we apply [GloVe](https://github.com/stanfordnlp/GloVe) to initialize the words embedding layer, while for BERT-based methods, we directly use the [AutoTokenizer](https://huggingface.co/) for words tokenization. The dataset preprocess file is in the:

+ src/datasets_glove.py
+ src/datasets.py

For GloVe download:

```shell
cd data
wget wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

For BERT-based models:

+ Install [Huggingface Transformers](https://huggingface.co/docs/transformers/installation):

```shell
conda install -c huggingface transformers
pip install emoji
pip install -r requirements.txt
```

## Dataset

| Topic   |  #Unlabeled |   #Labeled (Favor/Against/None) |
| :------------- | :----------: | :------------: |
| Stay at Home (SH) |   778   | 420 (194/113/113) |
| Wear Masks (WM)       |    1030     |  756 (173/288/295)  |
| Vaccination (VA)       |    1535     |  526 (106/194/226) |

> To obtain the raw tweet data, please contact: xiefeng@nudt.edu.cn

> For more detailed descriptions about the dataset, please clike [here](https://github.com/Xiefeng69/stance-detection-for-covid19-related-health-policies/tree/main/data)!

## Quick Start

### Running

+ cross-target setting:
```angular2html
python run.py --topic vaccination,face_masks --model mymodel --batch 16 --epoch 100 --hidden 128 --p_lambda 0.1 --alpha 0.01 --backbone bert_base
```

+ zero-shot setting:
```angular2html
python run.py --topic zeroshot,face_masks --model mymodel --batch 16 --epoch 100 --hidden 256 --p_lambda 0.1 --alpha 0.01 --backbone bert_base --lr 0.00002
```

### Parameters
| Parameter   |  Description |   Default |  Values. |
| :------------- | :----------: | :------------: |:------------: |
| --model | the running models   | mymodel | bilstm, bicond, textcnn, crossnet, tan, bert_base, mymodel |
| --topic | the running tasks    |  -  | cross-target setting or zero-shot setting |
| --batch | batch size     |  16 | - |
| --epoch | the number of epochs of traning process | 100 | - |
| --patience | we conduct early stop with fixed patience | 5 | - |
| --max_len | the maximum length of tokens | 100 | 100, 150|
| --hidden | the hidden dimension of model if it need it | 128 | 128, 256 |
| --alpha | the trade-off parameter of objectives | 0.01 | - |
| --p_lambda | negative constant in Gradient Reversal Layer | 0.1 | - |

### Baselines
All implemented methods are stored in the folder of `src/baselines`.

+ **BiLSTM** (1997): Long short-term memory.
+ **BiCond** (2016): Stance Detection with Bidirectional Conditional Encoding.
+ **TAN** (2017): Stance Classification with Target-Specific Neural Attention Networks.
+ **CrossNet** (2018): Cross-Target Stance Classification with Self-Attention Networks.
+ **SiamNet** (2019): Can siamese networks help in stance detection?
+ **Bert** (2019): BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
+ **Bertweet** (2020): BERTweet: A pre-trained language model for English Tweets.
+ **Covid-Tweet-Bert** (2020): COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter.
+ **WS-Bert** (2022): Infusing Knowledge from Wikipedia to Enhance Stance Detection.

## Implementation Details
All programs are implemented using 3.6.13 and PyTorch 1.10.2 with 11.3 on a personal workstation with an NVIDIA GeForce RTX 3090 GPU. The reported results are the averaged score of 5 runs with different random initialization.

### Training settings
In cross-target setting, the models are trained and validated on one topic and evaluated on another. There can be categorized into six source->destination tasks for cross-target evaluation: SH->WM, SH->VA, WM->SH, WMVA, VA->SH, and VA->WM. In zero-shot setting, the models are trained and validated on multiple topics and tested on one unseen topic. We use the unseen topic's name as the task's name, thus, the zero-shot evaluation can be set into: SH, WM, and VA. For all tasks, the batch size is set to 16, the dropout rate is set to 0.1, and the input texts are truncated or padded to a maximum of 100 tokens. We train all models using AdamW optimizer with weight decay 5e-5 for a maximum of 100 epochs with patience of 10 epochs, and the learning rate is chosen in {1e-5, 2e-5}.

### Models configuration
For BiLSTM, BiCond, TAN, CrossNet, TextCNN, the word embeddings are initialized with the pre-trained word vectors from [GloVe](https://github.com/stanfordnlp/GloVe), and the hidden dimension is optimized in {128, 256}. For BERT, we fine-tune the pre-trained language model from the [Hugging Face Transformer Library](https://huggingface.co/) to predict the stance by appending a linear classification layer to the hidden representation of the *[CLS]* token. In terms of WS-BERT-S and WS-BERT-D, considering the computational resource and fair comparison, the maximum length of Wikipedia summaries is set to 100 tokens and we use the pre-trained uncased BERT-base as encoder, in which each word is mapped to a 768-dimensional embedding. To speed up the training process, we only finetune the top layers of the Wikipedia encoder in WS-BERT-D, which is consistent with [paper](https://arxiv.org/abs/2204.03839). In our model, we also adopt the pre-trained uncased BERT-base as encoder. The maximum length of policy description is fixed at 50, the layer number *l* of GCN is set to 2, the trade-off parameter *alpha* is set to 0.01, the GRL's parameter *lambda* is set to 0.1, and the hidden dimension of GeoEncoder is optimized in {128, 256}.

## Code of Conduct and Ethical Statement
The tweet set for each policy contained a good mixture of pro, con, and neutral categories, as well as tweets with implicit and explicit opinions about the target. We removed the hashtags that appeared at the end of a tweet to exclude obvious cues, without making the tweet syntactically ambiguous. Each tweet was annotated by three annotators to avoid subjective errors of judgment. At present, we only collect social content on Twitter, without considering other social platforms, such as Weibo.

Our dataset does not provide any personally identifiable information as only the tweet IDs and human-annotated stance labels will be shared. Thus, the dataset complies with Twitter’s information privacy policy.

## Acknowledgement
We refer to the codes of these repos: [WS-BERT](https://github.com/zihaohe123/wiki-enhanced-stance-detection), [GVB](https://github.com/cuishuhao/GVB), [DANN](https://github.com/wogong/pytorch-dann). Thanks for their great contributions!

