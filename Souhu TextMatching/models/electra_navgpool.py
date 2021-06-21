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
class ElectraTextClassificationWithNAveragePooling(nn.Module):
    def __init__(self, args, model_path, num_labels=2, dropout=0.2, n=3):
        super(ElectraTextClassificationWithNAveragePooling, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_path)
        hidden_size = self.electra.encoder.layer[-1].output.dense.out_features

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size * n, num_labels)
        
        self.n = n
        
        self.use_ghmloss = args.use_ghmloss
        self.ghmloss_bins = args.ghmloss_bins
        self.ghmloss_alpha = args.ghmloss_alpha
        self.use_focalloss = args.use_focalloss
        
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs  = self.electra(
            input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True, 
            return_dict=True
        )
        n_hidden_states = outputs.hidden_states[-self.n:]
        n_avgs = [hidden_state[:, 0].reshape((-1, 1, 768)) for hidden_state in n_hidden_states]
        
        n_avgs_cls = n_avgs.append(outputs.pooler_output)
        all_h = torch.cat(n_avgs_cls, 1)
        mean_pool = torch.mean(all_h, 1)
        
        pooled_output = self.dropout(mean_pool)
        logits = self.fc(pooled_output)

        if self.use_ghmloss:
            loss_fct = GHMC_Loss(self.ghmloss_bins, self.ghmloss_alpha)
        elif self.use_focalloss:
            loss_fct = FocalLoss(logits=True)
        else:
            loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits,)
        
        