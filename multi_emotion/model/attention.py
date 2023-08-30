"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import math
import torch
from torch import nn


class TokenAttention(nn.Module):
    def __init__(self, query_size, num_heads, hidden_size):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = query_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.ModuleList([nn.Linear(self.attention_head_size, self.attention_head_size) for _ in range(self.num_attention_heads)])
        self.key = nn.Linear(hidden_size, query_size)
        self.value = nn.Linear(hidden_size, query_size)

        self.dense = nn.Linear(self.all_head_size, query_size)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, query_states):
        """

        Args:
            hidden_states: the word vec of the covered terminal words: [Batch_size, Seq_length, Hidden_size]
            query_states: the phrase vec of the covered terminal words: [Batch_size, Num_of_heads x Seq_length, Query_size]

        Returns:

        """
        new_query_shape = query_states.size()[:-2] + (hidden_states.size()[1], self.num_attention_heads, self.attention_head_size)
        query_states = query_states.view(*new_query_shape) # [Batch_size x Seq_length x Num_of_heads x Head_size]
        mixed_query_layer = torch.stack([self.query[i](query_states[:, :, i]) for i in range(self.num_attention_heads)])  # [Num_of_heads x Batch_size x Seq_length x Head_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Head_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Head_size]

        attention_scores = torch.stack([torch.matmul(mixed_query_layer[i], mixed_key_layer.transpose(-1,
                                                                            -2)) for i in range(self.num_attention_heads)])  # [Num_of_heads x Batch_size x Seq_length x Seq_length]

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Num_of_heads x Batch_size x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Num_of_heads x Batch_size x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     mixed_value_layer)  # [Num_of_heads x Batch_size x Seq_length x Head_size]

        context_layer = context_layer.permute(1, 2, 0, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)
        # pick the one with the highest attention score
        output = torch.max(output, dim=1)[0]

        return output