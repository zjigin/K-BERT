# -*- encoding:utf-8 -*-
import torch.nn as nn

from uer.layers.multi_headed_attn import MultiHeadedAttention


class AttnEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(AttnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.dropout
        )
        self.self_attn = nn.ModuleList([
            MultiHeadedAttention(
                args.hidden_size, args.heads_num, args.dropout
            )
            for _ in range(self.layers_num)
        ])

    def forward(self, emb, sequence):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            sequence: [batch_size x 1 x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        mask = (sequence > 0). \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.self_attn[i](hidden, hidden, hidden, mask)

        return hidden
