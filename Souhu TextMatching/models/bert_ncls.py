# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from .GHM_Loss import GHMC_Loss
from .Focal_Loss import FocalLoss

class BertTextClassificationNCls(nn.Module):
    def __init__(self, args, bert_path, label_num=2, dropout=0.2, n=3):
        super(BertTextClassificationNCls, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features * n, label_num),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss
        
        self.n = n

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, 
                            output_hidden_states=True, return_dict=True)
        n_hidden_states = outputs.hidden_states[-self.n:]
        n_cls = [hidden_state[:,:1,:] for hidden_state in n_hidden_states]
        n_cls = torch.cat(n_cls,dim=1)
        n_cls = n_cls.reshape(n_cls.shape[0], -1)
        
        pooled_output = self.dropout(n_cls)
        logits = self.fc(pooled_output)
        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits,)
