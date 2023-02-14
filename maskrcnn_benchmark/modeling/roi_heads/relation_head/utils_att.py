import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, num_attention_heads=8, input_size=4096, hidden_size=4096):
        super(Self_Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        prod_rep, num_rels = x
        prod_rep_enc = list(prod_rep.split(num_rels, dim=0))
        for i in range(len(num_rels)):
            x = prod_rep_enc[i].unsqueeze(0)
            key = self.key_layer(x)
            query = self.query_layer(x)
            value = self.value_layer(x)

            key_heads = self.trans_to_multiple_heads(key)
            query_heads = self.trans_to_multiple_heads(query)
            value_heads = self.trans_to_multiple_heads(value)

            attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_probs = F.softmax(attention_scores, dim=-1)

            context = torch.matmul(attention_probs, value_heads)
            context = context.permute(0, 2, 1, 3).contiguous()
            new_size = context.size()[: -2] + (self.all_head_size , )
            context = context.view(*new_size).squeeze(0)

            prod_rep_enc[i] = context
        prod_rep_enc = torch.cat(prod_rep_enc, dim=0)
        res1 = prod_rep_enc + prod_rep
        res2 = res1 + self.fc(res1)
        return [res2, num_rels]

class Multi_Self_Attention(nn.Module):
    def __init__(self, sa_num=4, num_attention_heads=8, input_size=4096, hidden_size=512):
        super(Multi_Self_Attention, self).__init__()
        self.MSA = [Self_Attention(num_attention_heads=num_attention_heads, input_size=input_size, hidden_size=hidden_size) for _ in range(sa_num)]
        models = [Self_Attention(num_attention_heads=num_attention_heads, input_size=input_size, hidden_size=hidden_size) for _ in range(sa_num)]
        self.MSA = nn.Sequential(*models)

    def forward(self, x):
        x = self.MSA(x)
        return x[0]
