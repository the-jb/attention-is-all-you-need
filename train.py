import torch
from torchtext.datasets.translation import WMT14, Multi30k
from torchtext.data import Field, BucketIterator
from model import TransformerModel


def train(epoch=1, batch_size=16, warmup_steps=4000, data_cls=WMT14, root_data_path='.data', **model_params):
    print(f"Train : {epoch=}, {batch_size=}")
    model = TransformerModel(**model_params)

    source = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='en_core_web_sm')
    target = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='de_core_news_sm')

    train_data, valid_data, _ = data_cls.splits(exts=('.en', '.de'), fields=(source, target), root=root_data_path)
    train_it, valid_it = BucketIterator.splits((train_data, valid_data), batch_size=batch_size)

    source.build_vocab(train_data)
    target.build_vocab(train_data)

    k = model.d_model ** -0.5
    opt = torch.optim.Adam(model.parameters(), lr=k * warmup_steps ** -1.5, betas=(0.9, 0.98), eps=1e-9)
    lr = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: k * min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5)))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=target.vocab.stoi['<pad>'])

    for e in range(epoch):
        epoch_loss = 0
        for b in train_it:
            output = model(b.src)
            loss = criterion(output, b.trg)

            opt.zero_grad()
            loss.backward()

            opt.step(), lr.step()

            epoch_loss += loss.item()
        print(epoch_loss)
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
    parser.add_argument('--model-path', type=str, nargs=1, default='.data/model.pb')
    parser.add_argument('--use-small-data', action='store_true', default=False)

    # train args
    parser.add_argument('--epoch', type=int, nargs='?')
    parser.add_argument('--batch-size', type=int, nargs='?')
    parser.add_argument('--warmup-steps', type=int, nargs='?')
    parser.add_argument('--root-data-path', type=str, nargs='?')

    args = vars(parser.parse_args())
    path = args.pop('model_path')
    if args.pop('use_small_data'):
        args['data_cls'] = Multi30k

    trained_model = train(**args)
    torch.save(trained_model, path)
