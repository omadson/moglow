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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LSTM, self).__init__()
        num_layers = 2
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_channels, self.num_layers, batch_first=True).double()

        # Define the output layer
        self.linear = LinearZeroInit(self.hidden_channels, out_channels).double()

        # do_init
        self.do_init = True

        # self.lstm.flatten_parameters()

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, inputs, context=None):
        # dtype = torch.DoubleTensor
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (batch_size, num_layers, hidden_dim).

        # h_0 = Variable(torch.zeros(self.num_layers, inputs.size(0), self.hidden_channels).type(dtype).to(inputs.device)) # hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, inputs.size(0), self.hidden_channels).type(dtype).to(inputs.device)) # internal state
        # Propagate input through LSTM
        # lstm_out, (hn, cn) = self.lstm(inputs, (h_0, c_0)) # lstm with input, hidden, and internal state

        if self.do_init:
            lstm_out, self.hidden = self.lstm(inputs)
            self.do_init = False
        else:
            lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        
        # print(lstm_out.shape)
        return self.linear(lstm_out.double())