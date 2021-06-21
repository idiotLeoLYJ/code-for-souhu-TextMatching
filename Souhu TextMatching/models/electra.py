# -*- coding:utf-8 -*-

from torch import nn
from transformers import ElectraModel
from torch.nn import CrossEntropyLoss
from .GHM_Loss import GHMC_Loss
from .Focal_Loss import FocalLoss

# class ElectraClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, hidden_size, label_num=2,dropout=0.2):
#         super().__init__()
#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(dropout)
#         self.out_proj = nn.Linear(hidden_size, num_labels)

#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

# https://github.com/huggingface/transformers/blob/master/src/transformers/models/electra/modeling_electra.py
class ElectraTextClassification(nn.Module):
    def __init__(self, args, model_path, num_labels=2, dropout=0.2):
        super(ElectraTextClassification, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_path)
        hidden_size = self.electra.encoder.layer[-1].output.dense.out_features
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss
        
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        discriminator_hidden_states  = self.electra(
            input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            return_dict=False
        )
        sequence_output = discriminator_hidden_states[0][:,0,:]
        print(sequence_output.size())
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.fc(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.out_proj(sequence_output)
        
        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits,)
        
        