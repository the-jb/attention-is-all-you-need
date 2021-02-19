import torch
from model import TransformerModel


def train(**kwargs):
    epoch = kwargs.pop('epoch')
    model = TransformerModel(**kwargs)

    for e in range(epoch):
        pass

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--n', type=int, nargs='?')
    parser.add_argument('--d-model', type=int, nargs='?')
    parser.add_argument('--d-ff', type=int, nargs='?')
    parser.add_argument('--h', type=int, nargs='?')
    parser.add_argument('--d-k', type=int, nargs='?')
    parser.add_argument('--d-v', type=int, nargs='?')
    parser.add_argument('--p-drop', type=int, nargs='?')
    parser.add_argument('--e-ls', type=int, nargs='?')

    parser.add_argument('--model-path', type=str, default='./model.pb')
    parser.add_argument('--epoch', type=int, nargs='?', default=1)

    args = vars(parser.parse_args())
    path = args.pop('model_path')

    trained_model = train(**args)
    torch.save(trained_model, path)
