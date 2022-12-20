import torch
from torch import nn
from torch.autograd import Variable


class LinearZeroInit(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks_per_layer):
        super(LSTM, self).__init__()
        self.input_dim = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_blocks_per_layer

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            batch_first=True
        ).double()

        # Define the output layer
        self.linear = LinearZeroInit(self.hidden_channels, out_channels).double()
    
    def init_hidden(self, batch_size):
        """a"""
        weight = next(self.parameters()).data
        self.hidden = (
            Variable(weight.new(self.num_layers, batch_size, self.hidden_channels).zero_()),
            Variable(weight.new(self.num_layers, batch_size, self.hidden_channels).zero_())
        )

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (batch_size, num_layers, hidden_dim).
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        return self.linear(lstm_out)