# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import torch
import numpy as np
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from .GHM_Loss import GHMC_Loss
from .Focal_Loss import FocalLoss


class BertTextClassificationWithAveragePooling(nn.Module):
    def __init__(self, args, bert_path, label_num=2, dropout=0.2):
        super(BertTextClassificationWithAveragePooling, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc_a = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, label_num),
        )
        self.fc_b = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, label_num),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss

    def forward(self, input_ids, token_type_ids, attention_mask, labels, type_index_a, type_index_b):
        hiddens, pooler = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,return_dict=False)
        average = self.dropout(torch.mean(hiddens, dim=1))
        
        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
            
        if type_index_a is not None:
            a_avg_emb = average[type_index_a]
            logits_a = self.fc_a(a_avg_emb)
            
        if type_index_b is not None:
            b_avg_emb = average[type_index_b]
            logits_b = self.fc_b(b_avg_emb)
            
        logits = torch.cat([logits_a, logits_b])
        
        index = torch.cat([type_index_a, type_index_b])
        reversed_index = torch.zeros_like(index)
        for key, i in enumerate(np.array(index.detach().cpu())):
            reversed_index[i] = key
        
        logits = torch.index_select(logits, dim=0, index=reversed_index)
        
        loss = loss_fct(logits, labels)
        return (loss, logits,)