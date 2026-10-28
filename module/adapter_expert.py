import torch
from torch import nn


class BottleneckAdapterExpert(nn.Module):
    def __init__(self, embed_dims, bottleneck_dim, activation, dropout, zero_init):
        super().__init__()

        if bottleneck_dim < 1:
            raise ValueError('Invalid adapter bottleneck or dropout')

        dropout_in_range = (dropout >= 0) and (dropout < 1)
        if not dropout_in_range:
            raise ValueError('Invalid adapter bottleneck or dropout')

        if activation == 'GELU':
            activation_layer = nn.GELU()
        elif activation == 'ReLU':
            activation_layer = nn.ReLU()
        elif activation == 'SiLU':
            activation_layer = nn.SiLU()
        else:
            raise ValueError(f'Unsupported adapter activation: {activation}')

        self.down = nn.Conv2d(embed_dims, bottleneck_dim, 1)
        self.activation = activation_layer
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Conv2d(bottleneck_dim, embed_dims, 1)
        self.zero_init = zero_init

        self.reset_residual()

    def reset_residual(self):
        if not self.zero_init:
            return

        with torch.no_grad():
            for out_index in range(self.up.weight.shape[0]):
                self.up.weight[out_index].zero_()

            if self.up.bias is not None:
                for out_index in range(self.up.bias.shape[0]):
                    self.up.bias[out_index].zero_()

    def forward(self, feature):
        # breakpoint()
        hidden = self.down(feature)
        activated = self.activation(hidden)
        dropped = self.dropout(activated)
        output = self.up(dropped)
        # breakpoint()
        return output
