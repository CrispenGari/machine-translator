from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout,
                 attention
                 ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0) # input = [1, batch size]
        embedded = self.dropout(self.embedding(input)) # embedded = [1, batch size, emb dim]
        a = self.attention(hidden, encoder_outputs, mask  )# a = [batch size, src len]
        a = a.unsqueeze(1) # a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # encoder_outputs = [batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs) # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2) # weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2) # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1)) # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0), a.squeeze(1)