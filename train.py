import torch
from torchtext.datasets.translation import WMT14, Multi30k
from torchtext.data import Field, BucketIterator
from model import TransformerModel


def train(epoch=1, batch_size=128, warmup_steps=4000, data_cls=WMT14, root_data_path='.data', e_ls=0.1, **model_params):
    print(f"Train : {epoch=}, {batch_size=}")
    source = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='en_core_web_sm', batch_first=True)
    target = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='de_core_news_sm', batch_first=True)

    train_data, valid_data, _ = data_cls.splits(exts=('.en', '.de'), fields=(source, target), root=root_data_path)

    source.build_vocab(train_data)
    target.build_vocab(train_data)
    src_vocab_size = len(source.vocab)
    trg_vocab_size = len(target.vocab)
    src_padding_idx = source.vocab.stoi[source.pad_token]
    trg_padding_idx = target.vocab.stoi[target.pad_token]

    model = TransformerModel(src_vocab_size, trg_vocab_size, src_padding_idx, trg_padding_idx, **model_params)

    train_it, valid_it = BucketIterator.splits((train_data, valid_data), batch_size=batch_size)

    def _lrate(step_num):
        return (model.d_model ** -0.5) * min((step_num + 1) ** -0.5, (step_num + 1) * (warmup_steps ** -1.5))

    opt = torch.optim.Adam(model.parameters(), lr=_lrate(0), betas=(0.9, 0.98), eps=1e-9)
    lr = torch.optim.lr_scheduler.LambdaLR(opt, _lrate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_padding_idx, reduction='sum')

    model.train()

    total_loss = 0
    n = 0
    for e in range(epoch):
        epoch_loss = 0
        for batch in train_it:
            opt.zero_grad()

            output = model(batch.src, batch.trg[:, :-1])
            loss = criterion(output.view(-1, trg_vocab_size), batch.trg[:, 1:].flatten())

            loss.backward()
            opt.step(), lr.step()

            epoch_loss += loss.item()
            n += batch.trg.ne(trg_padding_idx).sum().item()

        epoch_loss = epoch_loss / n
        print(f"Epoch {e + 1}/{epoch} Loss : {epoch_loss}")
        total_loss += epoch_loss
    print(f"Total Loss : {total_loss}")
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

    # main args
    parser.add_argument('--model-path', type=str, nargs='?', default='.data/model.pb')
    parser.add_argument('--root-data-path', type=str, nargs='?')
    parser.add_argument('--use-small-data', action='store_true', default=False)

    # train args
    parser.add_argument('--e-ls', type=int, nargs='?')
    parser.add_argument('--epoch', type=int, nargs='?')
    parser.add_argument('--batch-size', type=int, nargs='?')
    parser.add_argument('--warmup-steps', type=int, nargs='?')

    args = vars(parser.parse_args())
    path = args.pop('model_path')
    if args.pop('use_small_data'):
        args['data_cls'] = Multi30k

    trained_model = train(**args)
    torch.save(trained_model, path)
