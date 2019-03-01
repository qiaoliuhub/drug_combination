import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer
from Sublayers import Norm, OutputFeedForward
import copy
import setting


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)
        self.out = OutputFeedForward(d_model, n_feature_type)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        flat_d_output = d_output.view(-1, d_output.size(-2)*d_output.size(-1))
        output = self.out(flat_d_output)
        return output


def get_model():
    assert setting.d_model % setting.attention_heads == 0
    assert setting.attention_dropout < 1

    model = Transformer(setting.d_model, setting.n_feature_type, setting.n_layers, setting.attention_heads, setting.attention_dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # if setting.device == 0:
    #     model = model.cuda()

    return model