import os

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, context

from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy
from mindnlp.dataset.transforms import PadTransform

from mindnlp.utils import ModelOutput
from mindnlp.transformers.modeling_outputs import *
from mindnlp.transformers.modeling_outputs import SequenceClassifierOutput

from mindnlp.models import BertConfig, BertModel

from mindspore import nn

import faiss

# prepare dataset
class SentimentDataset:
    """Sentiment Dataset"""

    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            text_a, label = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)

import numpy as np

def process_dataset(source, tokenizer, max_seq_len=64, batch_size=32, shuffle=True):
    column_names = ["label", "text_a"]
    
    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # import pdb; pdb.set_trace()
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int32)
    def tokenize_and_pad(text):
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
        return tokenized[0], tokenized[2]
    # map dataset
    dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a", output_columns=['input_ids', 'attention_mask'])
    dataset = dataset.map(operations=[type_cast_op], input_columns="label", output_columns='labels')
    # batch dataset
    dataset = dataset.batch(batch_size)

    return dataset

class RAG(nn.Cell):
    def __init__(self, num_labels, faiss_index, all_embeddings):
        super().__init__()
        self.num_labels = num_labels
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)
        # self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size//2, batch_first=True, bidirectional=True)
        self.crf_hidden_fc = nn.Dense(config.hidden_size, self.num_labels)
        self.faiss_index = faiss_index
        self.all_embeddings = all_embeddings
        # self.crf = CRF(self.num_labels, batch_first=True, reduction='mean')

    def construct(self, input_ids, attention_mask=None, labels=None):
        # import pdb; pdb.set_trace()
        attention_mask = (input_ids > 0)
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = output[-1]
        output_ = output
        # lstm_feat, _ = self.bilstm(output[0])
        D, I = self.faiss_index.search(output.numpy(), 2)
        D = D/(D.sum(-1).reshape(-1,1))
        searched_embedding = []
        for bs in range(output.shape[0]):
            for i in range(len(I[bs])):
                # import pdb; pdb.set_trace()
                # output_[bs] += D[bs][i] * self.all_embeddings[I[bs][i]][0]
                output_[bs] += self.all_embeddings[I[bs][i]][0]

        # import pdb; pdb.set_trace()
        emissions = self.crf_hidden_fc(output)
        loss = nn.CrossEntropyLoss()
        # loss_ranking = (emissions/emissions.sum(-1).reshape(-1,1) - I/I.sum(-1).reshape(-1,1) + I/I.sum(-1).reshape(-1,1)*1).reshpe(-1).sum(0)
        # import pdb; pdb.set_trace()
        # loss_crf = self.crf(emissions, tags=labels, seq_length=seq_length)
        loss_ = loss(emissions, labels)
        # return loss_.lgits loss_.loss
        return SequenceClassifierOutput(
            loss=loss_,
            logits=emissions,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
        # return loss_

def process_dataset_kn(source, tokenizer):
    dataset = GeneratorDataset(source, ["label", "text_a"], shuffle=False)
    def tokenize_and_pad(text):
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=64)
        return tokenized[0], tokenized[2]
    dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a", output_columns=['input_ids', 'attention_mask'])
    dataset = dataset.batch(batch_size=1)
    return dataset

import pandas as pd

'''def get_knowledge_embed(source, tokenizer, embedder):
    dataset = GeneratorDataset(source, ["label", "text_a"], shuffle=False)
    def tokenize_and_pad(text):
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=64)
        return tokenized[0], tokenized[2]
    dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a", output_columns=['input_ids', 'attention_mask'])
    dataset = dataset.batch(batch_size=1)
    import pdb; pdb.set_trace()
    for data in dataset:

    df = pd.read_csv(source, sep='\t')
    text_a_column = df['text_a']
    input_ids, attention_mask, embeddings = [], [], []
    for text in text_a_column:
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=64)
        # import pdb; pdb.set_trace()
        input_ids.append(tokenized['input_ids'])
        attention_mask.append(tokenized['attention_mask'])
        output = embedder(input_ids=mindspore.tensor(tokenized['input_ids']), attention_mask=mindspore.tensor(tokenized['attention_mask']))
        embeddings.append(output[-1])
    return embeddings'''
    
    # for 
    # tokenized_text_a = text_a_column.apply(lambda x: tokenizer(str(x)))

from mindnlp.transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

dataset_train = process_dataset(SentimentDataset("data/sst/train.tsv"), tokenizer)
dataset_val = process_dataset(SentimentDataset("data/sst/dev.tsv"), tokenizer)
dataset_test = process_dataset(SentimentDataset("data/sst/test.tsv"), tokenizer, shuffle=False)
# import pdb; pdb.set_trace()
dataset_train.get_col_names()

from mindnlp.transformers import BertForSequenceClassification, BertModel
from mindnlp._legacy.amp import auto_mixed_precision

# set bert config and define parameters for training
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
# model = RAG(num_labels=len(Entity)*2+1)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
# model = auto_mixed_precision(model, 'O1')
'''knowledge_path = 'data/sst/train.tsv'
config = BertConfig.from_pretrained('bert-base-uncased')
knowledge_bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)
knowledge_embeddings = get_knowledge_embed(knowledge_path, tokenizer, knowledge_bert_model)
import pdb; pdb.set_trace()'''
config = BertConfig.from_pretrained('bert-base-uncased')
knowledge_bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)
knowledge = process_dataset_kn(SentimentDataset("data/sst/train.tsv"), tokenizer)
cpu_index = faiss.IndexFlatIP(768)
all_embeddings = []
for kn in knowledge: 
    embeddings = knowledge_bert_model(input_ids=kn[0], attention_mask=kn[1])[-1]
    all_embeddings.append(embeddings)
    # cpu_index.add(embeddings.cpu().numpy())
    cpu_index.add(embeddings.numpy())

model = RAG(2, cpu_index, all_embeddings)
optimizer = nn.Adam(model.trainable_params(), learning_rate=2e-5)

metric = Accuracy()
# define callbacks to save checkpoints
ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='bert_emotect', epochs=1, keep_checkpoint_max=2)
best_model_cb = BestModelCallback(save_path='checkpoint', ckpt_name='bert_emotect_best', auto_load=True)


trainer = Trainer(network=model, train_dataset=dataset_train,
                  eval_dataset=dataset_val, metrics=metric,
                  epochs=5, optimizer=optimizer, callbacks=[ckpoint_cb, best_model_cb])

# start training
trainer.run(tgt_columns="labels")

evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
evaluator.run(tgt_columns="labels")

# dataset_infer = SentimentDataset("data/sst/infer.tsv")

# def predict(text, label=None):
#     label_map = {0: "消极", 1: "中性", 2: "积极"}

#     text_tokenized = Tensor([tokenizer(text).input_ids])
#     logits = model(text_tokenized)
#     predict_label = logits[0].asnumpy().argmax()
#     info = f"inputs: '{text}', predict: '{label_map[predict_label]}'"
#     if label is not None:
#         info += f" , label: '{label_map[label]}'"
#     print(info)

# from mindspore import Tensor

# for label, text in dataset_infer:
#     predict(text, label)

# predict(" ")