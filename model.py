import torch
import torch.nn as nn
import numpy as np


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_padding_idx, trg_padding_idx,
                 max_seq_len=5000, n=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, p_drop=0.1):
        super(TransformerModel, self).__init__()
        print(f"Init TransformerModel : {src_vocab_size=}, {trg_vocab_size=}, {src_padding_idx=}, {trg_padding_idx=},"
              f" {n=}, {d_model=}, {d_ff=}, {h=}, {d_k=}, {d_v=}, {p_drop=}")
        self.d_model = d_model
        self.src_padding_idx = src_padding_idx
        self.trg_padding_idx = trg_padding_idx
        self.src_embedding = Embedding(src_vocab_size, d_model, src_padding_idx)
        self.trg_embedding = Embedding(trg_vocab_size, d_model, trg_padding_idx)
        self.dropout = nn.Dropout(p_drop)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.encoder_stack = EncoderStack(n, d_model, d_k, d_v, h, d_ff, p_drop)
        self.decoder_stack = DecoderStack(n, d_model, d_k, d_v, h, d_ff, p_drop)
        self.projection = nn.Linear(d_model, trg_vocab_size, bias=False)

        self.register_buffer('leftward_mask', torch.triu(torch.ones((max_seq_len, max_seq_len)), diagonal=1).bool())

    @classmethod
    def _mask_paddings(cls, x, padding_idx):
        return x.eq(padding_idx).unsqueeze(-2).unsqueeze(1)

    def forward(self, src, trg):
        # src : [BATCH * SRC_SEQ_LEN], trg : [BATCH * TRG_SEQ_LEN]

        src_padding_mask = self._mask_paddings(src, self.src_padding_idx)
        src = self.src_embedding(src)
        src = self.dropout(src + self.positional_encoding(src))

        x = self.encoder_stack(src, padding_mask=src_padding_mask)

        trg_padding_mask = self._mask_paddings(trg, self.trg_padding_idx)
        trg_self_attn_mask = trg_padding_mask | self.leftward_mask[:trg.size(-1), :trg.size(-1)]
        # print(trg_self_attn_mask[0])
        trg = self.trg_embedding(trg)
        trg = self.dropout(trg + self.positional_encoding(trg))

        x = self.decoder_stack(trg, x, self_attn_mask=trg_self_attn_mask, enc_dec_attn_mask=src_padding_mask)
        x = self.projection(x)
        return x


class EncoderStack(nn.Module):
    def __init__(self, n, d_model, d_k, d_v, h, d_ff, p_drop):
        super(EncoderStack, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, p_drop) for _ in range(n)])

    def forward(self, x, padding_mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, p_drop):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.add_and_norm_1 = AddAndNorm(d_model, p_drop)
        self.pos_ff = PositionWiseFeedForward(d_model, d_ff)
        self.add_and_norm_2 = AddAndNorm(d_model, p_drop)

    def forward(self, x, padding_mask):
        x = self.add_and_norm_1(x, self.self_attention(x, x, x, mask=padding_mask))
        x = self.add_and_norm_2(x, self.pos_ff(x))
        return x


class DecoderStack(nn.Module):
    def __init__(self, n, d_model, d_k, d_v, h, d_ff, p_drop):
        super(DecoderStack, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, p_drop) for _ in range(n)])

    def forward(self, x, x_enc, self_attn_mask, enc_dec_attn_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, x_enc, self_attn_mask, enc_dec_attn_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, p_drop):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.add_and_norm_1 = AddAndNorm(d_model, p_drop)
        self.enc_dec_attention = MultiHeadAttention(d_model, d_v, d_v, h)
        self.add_and_norm_2 = AddAndNorm(d_model, p_drop)
        self.pos_ff = PositionWiseFeedForward(d_model, d_ff)
        self.add_and_norm_3 = AddAndNorm(d_model, p_drop)

    def forward(self, x, x_enc, self_attn_mask, enc_dec_attn_mask):
        x = self.add_and_norm_1(x, self.self_attention(x, x, x, mask=self_attn_mask))
        x = self.add_and_norm_2(x, self.enc_dec_attention(x, x_enc, x_enc, mask=enc_dec_attn_mask))
        x = self.add_and_norm_3(x, self.pos_ff(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, h * d_k, bias=False)
        self.w_k = nn.Linear(d_model, h * d_k, bias=False)
        self.w_v = nn.Linear(d_model, h * d_v, bias=False)
        self.w_o = nn.Linear(h * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k)

    def _split_into_heads(self, *xs):
        # x : [BATCH * SEQ_LEN * D_MODEL] -> [BATCH * H * SEQ_LEN * D]
        return [x.view(x.size(0), x.size(1), self.h, -1).transpose(1, 2) for x in xs]

    def forward(self, q, k, v, mask=None):
        # q, k, v : [BATCH * SEQ_LEN * D_MODEL]

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self._split_into_heads(q, k, v)  # -> q, k, v : [BATCH * H * SEQ_LEN * D]

        x = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)  # -> x : [BATCH * SEQ_LEN * D_MODEL]
        x = self.w_o(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = d_k ** -0.5

    def forward(self, q, k, v, mask):
        # q, k, v : [BATCH * H * SEQ_LEN * D_K(D_V)]

        x = torch.matmul(q, k.transpose(-2, -1))  # -> x : BATCH * H * SEQ_LEN * SEQ_LEN

        x = x if mask is None else x.masked_fill(mask, float('-inf'))
        x = torch.matmul(torch.softmax(self.scale * x, dim=-1), v)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = d_model ** 0.5

    def forward(self, x):
        x = self.embedding(x)
        return x * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        # torch.sin/cos 함수가 계산이 더 빠르지만, 계산값은 numpy와 비교시 오차가 있기 때문에 numpy로 계산하고 tensor로 변환
        angles = np.array([[pos / (10000 ** (2 * i / d_model)) for i in range(d_model // 2)] for pos in range(max_seq_len)])
        sinusoids = np.empty((max_seq_len, d_model))
        sinusoids[:, 0::2], sinusoids[:, 1::2] = np.sin(angles), np.cos(angles)
        self.register_buffer('sinusoids', torch.FloatTensor(sinusoids).unsqueeze(0))

    def forward(self, x):
        return self.sinusoids[:, :x.size(1)]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(torch.relu(self.linear_1(x)))


class AddAndNorm(nn.Module):
    def __init__(self, d_model, p_drop):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, inputs, x):
        return self.layer_norm(inputs + self.dropout(x))
