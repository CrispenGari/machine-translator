from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout
                 ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim=emb_dim)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))
        packed_outputs, hidden = self.gru(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


