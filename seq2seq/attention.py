import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim,
                 dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(
            torch.cat((hidden, encoder_outputs), dim=2)
        ))  # energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)  # attention= [batch size, src len]
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)