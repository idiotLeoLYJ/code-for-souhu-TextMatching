# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import torch
import numpy as np
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss, KLDivLoss
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
        self.fc_a = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features * 2, label_num),
        )
        self.fc_b = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features * 2, label_num),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        
        self.dropout = nn.Dropout(dropout)
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss
        
        self.lamb = args.lamb
        self.T = args.temperature


    def forward(self, input_ids, token_type_ids, attention_mask, labels, type_index_a, type_index_b, logits):
        seq_output, pooler = self.bert(input_ids, token_type_ids=token_type_ids, 
                                       attention_mask=attention_mask,return_dict=False)
        cls_token = self.cls_fc(pooler)
        average = self.dropout(torch.mean(seq_output, dim=1))
        cls_avg = torch.cat([cls_token, average], 1)

        pooled_output = self.dropout(cls_avg)
        
        if type_index_a is not None:
            a_emb = pooled_output[type_index_a]
            logits_a = self.fc_a(a_emb)
            
        if type_index_b is not None:
            b_emb = pooled_output[type_index_b]
            logits_b = self.fc_b(b_emb)
            
        s_logits = torch.cat([logits_a, logits_b])

        index = torch.cat([type_index_a, type_index_b])
        reversed_index = torch.zeros_like(index)
        for key, i in enumerate(np.array(index.detach().cpu())):
            reversed_index[i] = key
        
        s_logits = torch.index_select(s_logits, dim=0, index=reversed_index) # 模型的logits
        
        # 1、KL Loss
        teacher_logits = logits.div(self.T)
        teacher_probs = self.softmax(teacher_logits)
        
        student_logits = s_logits.div(self.T)
        student_logprobs = self.logsoftmax(student_logits) # 输入需要是logsoftmax的结果
        
        kl_loss = KLDivLoss()
        loss_kl = kl_loss(student_logprobs, teacher_probs)
        
        # 2、BCE Loss
        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
        loss_hard = loss_fct(s_logits, labels)
        
        loss = self.lamb * self.T * self.T * loss_kl + (1-self.lamb) * loss_hard
        
        return (loss, s_logits,)