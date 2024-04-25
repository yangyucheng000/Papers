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

class Static(nn.Cell):
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
