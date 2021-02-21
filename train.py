import torch
from torchtext.datasets.translation import WMT14
from torchtext.data import Field, BucketIterator
from model import TransformerModel


def train(epoch=1, batch_size=16, **model_params):
    print(f"Train : {epoch=}, {batch_size=}")
    model = TransformerModel(**model_params)

    source = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='en_core_web_sm')
    target = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='de_core_news_sm')

    train_data, valid_data, _ = WMT14.splits(exts=('.en', '.de'), fields=(source, target))
    train_it, valid_it = BucketIterator.splits((train_data, valid_data), batch_size=batch_size)

    source.build_vocab(train_data)
    target.build_vocab(train_data)

    for e in range(epoch):
        for b in train_it:
            pass

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # model args
    parser.add_argument('--n', type=int, nargs='?')
    parser.add_argument('--d-model', type=int, nargs='?')
    parser.add_argument('--d-ff', type=int, nargs='?')
    parser.add_argument('--h', type=int, nargs='?')
    parser.add_argument('--d-k', type=int, nargs='?')
    parser.add_argument('--d-v', type=int, nargs='?')
    parser.add_argument('--p-drop', type=int, nargs='?')
    parser.add_argument('--e-ls', type=int, nargs='?')

    # main args
    parser.add_argument('--model-path', type=str, default='.data/model.pb')

    # train args
    parser.add_argument('--epoch', type=int, nargs='?')
    parser.add_argument('--batch-size', type=int, nargs='?')

    args = vars(parser.parse_args())
    path = args.pop('model_path')

    trained_model = train(**args)
    torch.save(trained_model, path)
