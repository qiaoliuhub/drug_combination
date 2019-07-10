import torch.nn as nn
from torch import cat
import torch
import torch.nn.functional as F
from Layers import EncoderLayer, DecoderLayer, OutputAttentionLayer
from Sublayers import Norm, OutputFeedForward
import copy
import setting
from attention_main import use_cuda, device2
from CustomizedLinear import CustomizedLinear

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None, low_dim=False):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask, low_dim=low_dim)
        return x if low_dim else self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None, low_dim = False):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask, low_dim=low_dim)
        return x if low_dim else self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)

    def forward(self, src, trg, src_mask=None, trg_mask=None, low_dim = False):
        e_outputs = self.encoder(src, src_mask, low_dim = low_dim)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, low_dim=low_dim)
        flat_d_output = d_output.view(-1, d_output.size(-2)*d_output.size(-1))
        return flat_d_output

class TransformerPlusLinear(Transformer):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__(d_model, N, heads, dropout)
        self.input_linear = nn.Linear(d_input, d_model)
        self.out = OutputFeedForward(d_model, n_feature_type, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src, trg, src_mask=None, trg_mask=None, low_dim = False):
        src = self.input_linear(src)
        trg = self.input_linear(trg)
        flat_d_output = super().forward(src, trg)
        output = self.out(flat_d_output)
        return output

class FlexibleTransformer(Transformer):

    def __init__(self, inputs_lengths, d_model, n_feature_type_list, N, heads, dropout):
        super().__init__(d_model, N, heads, dropout)
        self.final_inputs = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(inputs_lengths)):
            self.linear_layers.append(nn.Linear(inputs_lengths[i], d_model))
        out_input_length = d_model * sum(n_feature_type_list)
        self.out = OutputFeedForward(out_input_length, 1, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None, low_dim = False):

        assert len(self.linear_layers) == len(src_list), "Features and sources length are different"
        final_srcs = []
        final_trgs = []
        for i in range(len(self.linear_layers)):
            final_srcs.append(self.linear_layers[i](src_list[i]))
            final_trgs.append(self.linear_layers[i](trg_list[i]))
        final_src = cat(tuple(final_srcs), 1)
        final_trg = cat(tuple(final_trgs), 1)
        flat_d_output = super().forward(final_src, final_trg)
        output = self.out(flat_d_output)
        return output

class MultiTransformers(nn.Module):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout):
        super().__init__()

        assert len(d_input_list) == len(n_feature_type_list) and len(d_input_list) == len(d_model_list),\
            "claimed inconsistent number of transformers"
        self.linear_layers = nn.ModuleList()
        for i in range(len(d_input_list)):
            self.linear_layers.append(nn.Linear(d_input_list[i], d_model_list[i]))
        self.transformer_list = nn.ModuleList()
        self.n_feature_type_list = n_feature_type_list
        for i in range(len(d_input_list)):
            self.transformer_list.append(Transformer(d_model_list[i], N, heads, dropout))

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None):

        assert len(src_list) == len(self.transformer_list), "inputs length is not same with input length for model"
        src_list_linear = []
        trg_list_linear = []
        for i in range(len(self.linear_layers)):
            src_list_linear.append(self.linear_layers[i](src_list[i]))
            trg_list_linear.append(self.linear_layers[i](trg_list[i]))
        output_list = []
        for i in range(len(self.transformer_list)):
            output_list.append(self.transformer_list[i](src_list_linear[i], trg_list_linear[i]))

        return output_list

class TransposeMultiTransformers(nn.Module):

    def __init__(self,  d_input_list, d_model_list, n_feature_type_list, N, heads, dropout, masks = None):
        super().__init__()

        assert len(d_input_list) == len(n_feature_type_list) and len(d_input_list) == len(d_model_list),\
            "claimed inconsistent number of transformers"
        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(d_input_list)):
            if masks[i] is not None:
                if setting.seperate_drug_cellline:
                    for j in range(setting.n_feature_type[i]):
                        self.linear_layers.append(CustomizedLinear(masks[i]))
                        self.norms.append(Norm(d_model_list[i]))
                        self.dropouts.append(nn.Dropout(p=0))
                #assert masks[i].shape[0] ==
                else:
                    self.linear_layers.append(CustomizedLinear(masks[i]))
                    self.norms.append(Norm(d_model_list[i]))
                    self.dropouts.append(nn.Dropout(dropout))
            else:
                self.linear_layers.append(nn.Linear(d_input_list[i], d_model_list[i]))
                self.norms.append(Norm(d_model_list[i]))
                self.dropouts.append(nn.Dropout(dropout))
        self.transformer_list = nn.ModuleList()
        self.n_feature_type_list = n_feature_type_list
        for i in range(len(d_input_list)):
            self.transformer_list.append(Transformer(n_feature_type_list[i] * setting.d_model_i, N, heads, dropout))

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None, low_dim = False):

        # x = F.relu(self.linear_1(x))
        # x = x if low_dim else self.norm(x)
        # x = self.dropout(x)
        assert len(src_list) == len(self.transformer_list), "inputs length is not same with input length for model"
        src_list_linear = []
        trg_list_linear = []
        if not setting.apply_var_filter:
            cur_linear = 0
            for i in range(len(self.transformer_list)):
                if setting.seperate_drug_cellline:
                    src_list_dim = []
                    trg_list_dim = []
                    for j in range(src_list[i].size(1)):
                        cur_src_dim = src_list[i].narrow_copy(1,j,1)
                        cur_trg_dim = trg_list[i].narrow_copy(1,j,1)
                        cur_src_processed_dim = self.dropouts[cur_linear](F.relu(self.linear_layers[cur_linear](cur_src_dim)))
                        cur_trg_processed_dim = self.dropouts[cur_linear](F.relu(self.linear_layers[cur_linear](cur_trg_dim)))
                        src_list_dim.append(cur_src_processed_dim.contiguous().view([-1, setting.d_model_i, setting.d_model_j]))
                        trg_list_dim.append(cur_trg_processed_dim.contiguous().view([-1, setting.d_model_i, setting.d_model_j]))
                        cur_linear += 1
                    src_list_linear.append(cat(tuple(src_list_dim), dim = 1))
                    trg_list_linear.append(cat(tuple(trg_list_dim), dim = 1))
                elif low_dim:
                    src_list_linear.append(self.dropouts[i](F.relu(self.linear_layers[i](src_list[i]))))
                    trg_list_linear.append(self.dropouts[i](F.relu(self.linear_layers[i](trg_list[i]))))
                else:
                    src_list_linear.append(F.relu(self.linear_layers[i](src_list[i])))
                    trg_list_linear.append(F.relu(self.linear_layers[i](trg_list[i])))
        else:
            for i in range(len(src_list)):
                src_list_linear.append(src_list[i].clone())
                trg_list_linear.append(trg_list[i].clone())

        output_list = []
        for i in range(len(self.transformer_list)):
            src_list_linear[i] = torch.transpose(src_list_linear[i], -1, -2)
            trg_list_linear[i] = torch.transpose(trg_list_linear[i], -1, -2)
            output_list.append(self.transformer_list[i](src_list_linear[i], trg_list_linear[i], low_dim=low_dim))
        return output_list

class MultiTransformersPlusLinear(MultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout):

        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout)
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list))])
        self.out = OutputFeedForward(out_input_length, 1, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None):

        output_list = super().forward(src_list, trg_list)
        cat_output = cat(tuple(output_list), dim=1)
        output = self.out(cat_output)
        return output, cat_output

class TransposeMultiTransformersPlusLinear(TransposeMultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout, masks=None):

        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout, masks=masks)
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list))]) \
                           + setting.single_repsonse_feature_length
        self.out = OutputFeedForward(out_input_length, 1, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None, low_dim = True):

        input_src_list = src_list[:-1] if setting.single_repsonse_feature_length != 0 else src_list
        input_trg_list = trg_list[:-1] if setting.single_repsonse_feature_length != 0 else trg_list
        output_list = super().forward(input_src_list, input_trg_list, low_dim=low_dim)
        single_response_feature_list = []
        if setting.single_repsonse_feature_length != 0:
            single_response_feature_list = [src_list[-1].contiguous().view(-1, setting.single_repsonse_feature_length)]
        cat_output = cat(tuple(output_list + single_response_feature_list), dim=1)
        output = self.out(cat_output)
        return output, cat_output

class MultiTransformersPlusSDPAttention(MultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout):

        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout)
        self.n_feature_type_list = n_feature_type_list
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list)-1)])
        self.output_attn = OutputAttentionLayer(d_model_list[0], d_model_list[-1])
        H = sum(self.n_feature_type_list)-1
        self.out = OutputFeedForward(1, d_model_list[-1], d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None):
        output_list = super().forward(src_list, trg_list)
        bs = output_list[0].size(0)
        for i, output_tensor in enumerate(output_list):
            output_list[i] = output_tensor.contiguous().view(bs, self.n_feature_type_list[i], -1)
        cat_output = cat(tuple(output_list[:-1]), dim=1)
        attn_output = self.output_attn(output_list[-1], cat_output)
        attn_output = attn_output.contiguous().view(bs, -1)
        output = self.out(attn_output)
        return output, cat_output

class TransposeMultiTransformersPlusRNN(TransposeMultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout):

        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout)
        self.n_feature_type_list = n_feature_type_list
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list)-1)])
        self.hidden_size = 200
        self.rnn = nn.LSTM(input_size=d_model_list[0], hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.out = OutputFeedForward(sum(self.n_feature_type_list), self.hidden_size, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None, low_dim = True):
        output_list = super().forward(src_list, trg_list, low_dim=low_dim)
        bs = output_list[0].size(0)
        for i, output_tensor in enumerate(output_list):
            output_list[i] = output_tensor.contiguous().view(bs, self.n_feature_type_list[i], -1)
        cat_output = cat(tuple(output_list), dim=1)
        h_s, c_s = torch.randn(1, bs, self.hidden_size), torch.randn(1, bs, self.hidden_size)
        if use_cuda:
            h_s = h_s.to(device2)
            c_s = c_s.to(device2)
        rnn_output, hidden = self.rnn(cat_output, (h_s, c_s))
        attn_output = rnn_output.contiguous().view(bs, -1)
        output = self.out(attn_output, low_dim = low_dim)
        return output, cat_output


class MultiTransformersPlusRNN(MultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout):

        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout)
        self.n_feature_type_list = n_feature_type_list
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list)-1)])
        self.hidden_size = 200
        self.rnn = nn.LSTM(input_size=d_model_list[0], hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.out = OutputFeedForward(sum(self.n_feature_type_list), 2 * self.hidden_size, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None):
        output_list = super().forward(src_list, trg_list)
        bs = output_list[0].size(0)
        for i, output_tensor in enumerate(output_list):
            output_list[i] = output_tensor.contiguous().view(bs, self.n_feature_type_list[i], -1)
        cat_output = cat(tuple(output_list), dim=1)
        h_s, c_s = torch.randn(2, bs, self.hidden_size), torch.randn(2, bs, self.hidden_size)
        if use_cuda:
            h_s = h_s.to(device2)
            c_s = c_s.to(device2)
        rnn_output, hidden = self.rnn(cat_output, (h_s, c_s))
        attn_output = rnn_output.contiguous().view(bs, -1)
        output = self.out(attn_output)
        return output, cat_output

class MultiTransformersPlusMulAttention(MultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout):

        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout)
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list)-1)])
        self.hidden_size = 20
        self.linear = nn.Linear(d_model_list[-1], self.hidden_size)
        H = sum(self.n_feature_type_list)-1
        self.out = OutputFeedForward(H*self.hidden_size, d_model_list[-1], d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None):
        output_list = super().forward(src_list, trg_list)
        bs = output_list[0].size(0)
        for i, output_tensor in enumerate(output_list):
            output_list[i] = output_tensor.contiguous().view(bs, self.n_feature_type_list[i], -1)
        cat_output = cat(tuple(output_list[:-1]), dim=1)
        cat_output = self.linear(cat_output)
        mul_output = torch.matmul(cat_output.contiguous().view(bs,-1,1), output_list[-1].contiguous().view(bs,1,-1))
        output = self.out(mul_output.contiguous().view(bs, -1))
        return output, cat_output

class LastLSTM(nn.Module):

    def __init__(self, d_model_list, dropout):

        super(LastLSTM, self).__init__()
        self.hidden_size = 100
        self.rnn = nn.LSTM(input_size=d_model_list[0], hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.out = OutputFeedForward(3 * len(setting.catoutput_intput_type), 2*self.hidden_size, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, input):

        #cat_input = cat(tuple(input), dim=1)
        bs = input.size(0)
        h_s, c_s = torch.randn(2, bs, self.hidden_size), torch.randn(2, bs, self.hidden_size)
        if use_cuda:
            h_s = h_s.to(device2)
            c_s = c_s.to(device2)
        rnn_output, hidden = self.rnn(input, (h_s, c_s))
        attn_output = rnn_output.contiguous().view(bs, -1)
        output = self.out(attn_output, low_dim = True)
        return output

class LastFC(nn.Module):

    def __init__(self, d_model_list, dropout):

        super(LastFC, self).__init__()
        self.hidden_size = 100
        self.out = OutputFeedForward(3 * len(setting.catoutput_intput_type) * setting.d_model + 2 * sum(list(setting.dir_input_type.values())),
                                     1, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, input):

        #cat_input = cat(tuple(input), dim=1)
        bs = input.size(0)
        attn_output = input.contiguous().view(bs, -1)
        output = self.out(attn_output, low_dim = True)
        return output

def get_retrain_model():

    if not isinstance(setting.d_model, list):
        d_models = [setting.d_model] * 3 * len(setting.catoutput_intput_type)
    else:
        d_models = setting.d_model

    #model = LastLSTM(d_models, setting.attention_dropout)
    model = LastFC(d_models, setting.attention_dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if use_cuda:
        model.to(device2)

    return model

def get_model(inputs_lengths):
    assert setting.d_model % setting.attention_heads == 0
    assert setting.attention_dropout < 1

    #model = TransformerPlusLinear(setting.d_input, setting.d_model, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    model = FlexibleTransformer(inputs_lengths, setting.d_model, setting.n_layers, setting.attention_heads, setting.attention_dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def get_multi_models(inputs_lengths, input_masks = None):

    if not isinstance(setting.d_model, list):
        d_models = [setting.d_model] * len(inputs_lengths)
    else:
        d_models = setting.d_model

    if not isinstance(setting.n_feature_type, list):
        n_feature_types = [setting.n_feature_type] * len(inputs_lengths)
    else:
        n_feature_types = setting.n_feature_type

    for d_model in d_models:
        assert d_model % setting.attention_heads == 0
    assert setting.attention_dropout < 1

    if not isinstance(input_masks, list):
        input_masks = [input_masks] * len(inputs_lengths)
    else:
        input_masks = input_masks

    final_inputs_lengths = [inputs_lengths[i]//n_feature_types[i] for i in range(len(inputs_lengths))]
    #model = FlexibleTransformer(final_inputs_lengths, setting.d_model, setting.n_feature_type, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    #model = TransformerPlusLinear(final_inputs_lengths, d_models, setting.n_feature_type, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    #model = MultiTransformersPlusLinear(final_inputs_lengths, final_inputs_lengths, n_feature_types, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    #model = MultiTransformersPlusLinear(final_inputs_lengths, d_models, n_feature_types, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    #model = MultiTransformersPlusSDPAttention(final_inputs_lengths, d_models, n_feature_types, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    #model = MultiTransformersPlusMulAttention(final_inputs_lengths, d_models, n_feature_types, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    model = TransposeMultiTransformersPlusLinear(final_inputs_lengths, d_models, n_feature_types, setting.n_layers, setting.attention_heads, setting.attention_dropout, input_masks)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model