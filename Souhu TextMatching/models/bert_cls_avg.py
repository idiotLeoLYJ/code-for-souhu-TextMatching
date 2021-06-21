# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from .GHM_Loss import GHMC_Loss
from .Focal_Loss import FocalLoss


class BertTextClassificationWithClsAveragePooling(nn.Module):
    def __init__(self, args, bert_path, label_num=2, dropout=0.2):
        super(BertTextClassificationWithClsAveragePooling, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.cls_fc = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, self.bert.pooler.dense.out_features),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features * 2, label_num),
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss


    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        seq_output, pooler = self.bert(input_ids, token_type_ids=token_type_ids, 
                                       attention_mask=attention_mask,return_dict=False)
        cls_token = self.cls_fc(pooler)
        average = self.dropout_1(torch.mean(seq_output, dim=1))
        cls_avg = torch.cat([cls_token, average], 1)

        pooled_output = self.dropout_2(cls_avg)
        logits = self.fc(pooled_output)
        
        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits,)