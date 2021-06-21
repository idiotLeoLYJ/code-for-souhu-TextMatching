# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from .GHM_Loss import GHMC_Loss
from .Focal_Loss import FocalLoss

class BertTextClassification(nn.Module):
    def __init__(self, args, bert_path, label_num=2, dropout=0.2):
        super(BertTextClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, label_num),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        seq_emb, pooler = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,return_dict=False)
        cls_emb = self.dropout(pooler)
        logits = self.fc(cls_emb)
        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits,)
