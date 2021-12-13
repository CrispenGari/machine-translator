
import random, torch
from utils.main import Constants
from torch import nn
random.seed(Constants.SEED)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_idx = src_pad_idx

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        src = [src len, batch size]
        src_len = [batch size]
        trg = [trg len, batch size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        """
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        """
        encoder_outputs is all hidden states of the input sequence, back and forwards
        hidden is the final forward and backward hidden states, passed through a linear layer
        """
        encoder_outputs, hidden = self.encoder(src, src_len)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        mask = self.create_mask(src)  # mask = [batch size, src len]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder- hidden states and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs