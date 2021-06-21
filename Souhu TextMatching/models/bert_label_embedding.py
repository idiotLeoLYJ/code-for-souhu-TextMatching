# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from .GHM_Loss import GHMC_Loss
from .Focal_Loss import FocalLoss
from torch.cuda.amp import autocast

class BertTextClassificationLabelEmbedding(nn.Module):
    def __init__(self, args, bert_path, label_num=2, dropout=0.2):
        super(BertTextClassificationLabelEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, self.bert.pooler.dense.out_features),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss
        
        self.label_num = label_num


    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        seq_emb, pooler = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,return_dict=False)

        con_label_hiddens = seq_emb[:, :(1+self.label_num), :]
        con_label_hiddens = self.fc(con_label_hiddens)
        con_label_hiddens = self.dropout(con_label_hiddens)
        
        cls_hidden = con_label_hiddens[:, 0, :]
        label_hidden = con_label_hiddens[:, 1:(1+self.label_num), :]

        logits = torch.bmm(label_hidden,
                           cls_hidden.view(cls_hidden.size()[0], -1, 1)).squeeze()  # [batch_size, 2]

        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits, labels)

        return (loss, logits,)