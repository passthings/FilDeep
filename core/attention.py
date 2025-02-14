import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, emb_dim = 128, seq_length = 60, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(
            torch.arange(1, seq_length + 1, 1)[None, :, None] / emb_dim,
            requires_grad = True
        )

    def forward(self, x):
        return self.dropout(
            x + self.pos_embedding.to(x.device)
        )


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    """
    Q: n * d
    K: m * d
    V: m * v
    Q * K_T * V
    """

    def forward(self, queries, keys, values, flag):
        d = queries.size()[-1]
        attention_weights = nn.functional.softmax(
            torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d), dim = -1)
        scores = torch.bmm(attention_weights, values)
        return scores


def transpose_qkv(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_input, ffn_hidden, ffn_output, activate_fun = "relu", **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_input, ffn_hidden)
        if "relu" == activate_fun:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.GELU()
        self.dense2 = nn.Linear(ffn_hidden, ffn_output)

    def forward(self, x):
        return self.dense2(self.activate(self.dense1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            query_size,
            key_size,
            value_size,
            feature_dims,
            num_heads,
            dropout,
            bias = False,
            flag = "cross",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.flag = flag
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        self.W_q = nn.Linear(query_size, feature_dims, bias = bias)
        self.W_k = nn.Linear(key_size, feature_dims, bias = bias)
        self.W_v = nn.Linear(value_size, feature_dims, bias = bias)

        self.W_o = nn.Linear(feature_dims, feature_dims, bias = bias)

    def forward(self, x, y = None):
        queries = None
        keys = None
        values = None
        if self.flag == "cross":
            queries = transpose_qkv(self.W_q(x), self.num_heads)
            keys = transpose_qkv(self.W_k(y), self.num_heads)
            values = transpose_qkv(self.W_v(y), self.num_heads)
        elif self.flag == "self":
            queries = transpose_qkv(self.W_q(x), self.num_heads)
            keys = transpose_qkv(self.W_k(x), self.num_heads)
            values = transpose_qkv(self.W_v(x), self.num_heads)
        elif self.flag == "acf_self":
            queries = transpose_qkv(self.W_q(x), self.num_heads)
            keys = transpose_qkv(self.W_k(x), self.num_heads)
            values = transpose_qkv(self.W_v(x), self.num_heads)
        elif self.flag == "acf_cross":
            queries = transpose_qkv(self.W_q(x), self.num_heads)
            keys = transpose_qkv(self.W_k(y), self.num_heads)
            values = transpose_qkv(self.W_v(y), self.num_heads)
        else:
            print("flag error in MultiHeadAttention")

        output = self.attention(queries, keys, values, self.flag)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class DeformationLayer(nn.Module):
    def __init__(
            self,
            dropout,
            num_heads,
            fea_len,
            bias,
            flag = "cross",
            scale_factor = 2
    ):
        super().__init__()
        if flag == "acf":
            self.attention_self = MultiHeadAttention(
                fea_len,
                fea_len,
                fea_len,
                fea_len,
                num_heads,
                dropout,
                bias,
                flag + "_self"
            )
            self.attention_cross = MultiHeadAttention(
                fea_len,
                fea_len,
                fea_len,
                fea_len,
                num_heads,
                dropout,
                bias,
                flag + "_cross"
            )
        else:
            self.attention = MultiHeadAttention(
                fea_len,
                fea_len,
                fea_len,
                fea_len,
                num_heads,
                dropout,
                bias,
                flag
            )
        self.flag = flag
        self.add_norm = AddNorm(fea_len, dropout)
        self.fc = PositionWiseFFN(
            fea_len,
            fea_len * num_heads // scale_factor,
            fea_len,
            "relu"
        )
        self.add_norm_fc = AddNorm(fea_len, dropout)

    def forward(self, x, y = None):
        if self.flag == "cross":
            y = self.add_norm(y, self.attention(x, y))
            y = self.add_norm_fc(y, self.fc(y))
            return y
        elif self.flag == "self":
            x = self.add_norm(x, self.attention(x))
            x = self.add_norm_fc(x, self.fc(x))
            return x
        elif self.flag == "acf":
            y = self.add_norm(y, self.attention_cross(x, y))
            y = self.add_norm(y, self.attention_self(y))
            y = self.add_norm_fc(y, self.fc(y))
            return y
        else:
            print("flag error in DeformationLayer")


class DeformationModule(nn.Module):
    def __init__(
            self,
            num_layers = 1,
            dropout = 0.5,
            num_heads = 8,
            fea_len = 64,
            bias = False,
            flag = "cross"
    ):
        super().__init__()
        self.fea_len = fea_len
        if flag == "cross":
            self.pos_encoding_x = PositionalEncoding(dropout)
            self.pos_encoding_y = PositionalEncoding(dropout)
        elif flag == "self":
            self.pos_encoding = PositionalEncoding(dropout)
        elif flag == "acf":
            self.pos_encoding_x = PositionalEncoding(dropout)
            self.pos_encoding_y = PositionalEncoding(dropout)
        self.flag = flag
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(
                "DeformationLayer" + str(i),
                DeformationLayer(
                    dropout, num_heads, fea_len, bias, flag
                )
            )

    def forward(self, x, y = None):
        if self.flag == "cross":
            x = self.pos_encoding_x(x)
            y = self.pos_encoding_y(y)
            for layer in self.layers:
                y = layer(x, y)
            return y
        elif self.flag == "self":
            x = self.pos_encoding(x)
            for layer in self.layers:
                x = layer(x)
            return x
        elif self.flag == "acf":
            x = self.pos_encoding_x(x)
            y = self.pos_encoding_y(y)
            for layer in self.layers:
                y = layer(x, y)
            return y
        else:
            print("flag error in DeformationModule")


class CrossModule(nn.Module):
    def __init__(
            self,
            num_layers = 3,
            dropout = 0,
            num_heads = 8,
            fea_len = 128,
            bias = False
    ):
        super().__init__()
        self.cross_module = DeformationModule(
            num_layers,
            dropout,
            num_heads,
            fea_len,
            bias,
            "cross"
        )

    def forward(self, external_condition, strip):
        return self.cross_module(external_condition, strip)


class SelfModule(nn.Module):
    def __init__(
            self,
            num_layers = 3,
            dropout = 0,
            num_heads = 8,
            fea_len = 128,
            bias = False
    ):
        super().__init__()
        self.self_module = DeformationModule(
            num_layers,
            dropout,
            num_heads,
            fea_len,
            bias,
            "self"
        )

    def forward(self, strip):
        return self.self_module(strip)

class Attention_Cross_Fidelity(nn.Module):
    def __init__(
            self,
            num_layers = 3,
            dropout = 0,
            num_heads = 8,
            fea_len = 128,
            bias = False
    ):
        super().__init__()
        self.acf = DeformationModule(
            num_layers,
            dropout,
            num_heads,
            fea_len,
            bias,
            "acf"
        )

    def forward(self, external_condition, strip):
        return self.acf(external_condition, strip)
