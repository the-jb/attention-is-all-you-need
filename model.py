import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_sizes, n=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, p_drop=0.1, e_ls=0.1):
        super(TransformerModel, self).__init__()
        print(f"Init TransformerModel : {n=}, {d_model=}, {d_ff=}, {h=}, {d_k=}, {d_v=}, {p_drop=}, {e_ls=}")
        self.d_model = d_model
        self.src_embedding = Embedding(vocab_sizes[0], d_model)
        self.encoder_stack = EncoderStack()
        self.trg_embedding = Embedding(vocab_sizes[1], d_model)
        self.decoder_stack = DecoderStack()
        self.linear = nn.Linear(d_model, vocab_sizes[1], bias=False)

    def forward(self, src, trg):
        embedded_src = self.src_embedding(src)
        embedded_trg = self.trg_embedding(trg)

        x = self.encoder_stack(src)
        x = self.decoder_stack(embedded_src, embedded_trg, x)
        return self.linear(x)


class EncoderStack(nn.Module):
    def __init__(self):
        super(EncoderStack, self).__init__()


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()


class DecoderStack(nn.Module):
    def __init__(self):
        super(DecoderStack, self).__init__()


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding()

    def forward(self, x):
        return self.embedding(x) + self.positional_encoding(x)


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()


class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
