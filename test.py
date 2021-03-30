import torch
from torchtext.datasets.translation import WMT14, Multi30k
from torchtext.data import Field, Iterator


def test(model, max_seq_len=5000, root_data_path='.data', data_cls=WMT14):
    source = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='en_core_web_sm', batch_first=True)
    target = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', tokenizer_language='de_core_news_sm', batch_first=True)
    train_data, _, test_data = data_cls.splits(exts=('.en', '.de'), fields=(source, target), root=root_data_path)
    test_it = Iterator(test_data, batch_size=1, train=False)

    source.build_vocab(train_data)
    target.build_vocab(train_data)

    trg_init_idx = target.vocab.stoi[target.init_token]
    trg_eos_idx = target.vocab.stoi[target.eos_token]

    initial_output = torch.LongTensor([[trg_init_idx]])
    for (src, trg), _ in test_it:
        output = initial_output
        for i in range(max_seq_len):
            output = model(src, output)
            output = output.argmax(dim=-1)
            output = torch.cat((initial_output, output), dim=-1)
            if output[0, -1].item() == trg_eos_idx:
                print("End : ", output)
                break
        print("Source : ", ''.join(map(lambda x: source.vocab.itos[x] + ' ', src[0].numpy())))
        print("Target : ", ''.join(map(lambda x: target.vocab.itos[x] + ' ', trg[0].numpy())))
        print("Output : ", ''.join(map(lambda x: target.vocab.itos[x] + ' ', output[0].numpy())))
        print()
        # TODO : add BLEU score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seq-len', nargs='?', type=int)
    parser.add_argument('--model-path', nargs='?', type=str, default='./model.pb')
    parser.add_argument('--root-data-path', type=str, nargs='?')
    parser.add_argument('--use-small-data', action='store_true', default=False)
    args = vars(parser.parse_args())

    if args.pop('use_small_data'):
        args['data_cls'] = Multi30k
    model_path = args.pop('model_path')

    test(model=torch.load(model_path), **args)
