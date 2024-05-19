import argparse

import datasets
import tokenizers
import torch
import tqdm
from torch.utils.data import DataLoader

from model import TransformerModel


def construct_tokenizer(data):
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tokenizer.train_from_iterator(
        data,
        trainer=tokenizers.trainers.BpeTrainer(
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            vocab_size=37000,
            min_frequency=2,
        ),
    )
    tokenizer.enable_padding()
    return tokenizer


class BatchProcessor:
    def __init__(
        self,
        src_tokenizer,
        trg_tokenizer,
        src_key="en",
        trg_key="de",
        max_seq_len=5000,
    ):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_key = src_key
        self.trg_key = trg_key
        self.max_seq_len = max_seq_len

    def __call__(self, items: list[dict]) -> dict:
        return {
            "src": torch.as_tensor(
                [
                    enc.ids
                    for enc in self.src_tokenizer.encode_batch(
                        [item["translation"][self.src_key] for item in items]
                    )
                ][: self.max_seq_len]
            ),
            "trg": torch.as_tensor(
                [
                    enc.ids
                    for enc in self.trg_tokenizer.encode_batch(
                        [item["translation"][self.trg_key] for item in items]
                    )
                ][: self.max_seq_len]
            ),
        }


def train(
    epoch=1,
    batch_size=128,
    warmup_steps=4000,
    dataset="wmt/wmt14",
    subset="de-en",
    e_ls=0.1,
    **model_params,
):
    print(f"Train : {epoch=}, {batch_size=}")
    dataset = datasets.load_dataset(dataset, subset)
    src_tokenizer = construct_tokenizer(
        map(lambda x: x["translation"]["en"], dataset["train"])
    )
    trg_tokenizer = construct_tokenizer(
        map(lambda x: x["translation"]["de"], dataset["train"])
    )

    src_padding_idx = src_tokenizer.token_to_id("<pad>")
    trg_padding_idx = src_tokenizer.token_to_id("<pad>")

    src_vocab_size = src_tokenizer.get_vocab_size()
    trg_vocab_size = trg_tokenizer.get_vocab_size()

    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_padding_idx=src_padding_idx,
        trg_padding_idx=trg_padding_idx,
        **model_params,
    )

    def _lrate(step_num):
        return (model.d_model**-0.5) * min(
            (step_num + 1) ** -0.5, (step_num + 1) * (warmup_steps**-1.5)
        )

    opt = torch.optim.Adam(
        model.parameters(), lr=_lrate(0), betas=(0.9, 0.98), eps=1e-9
    )
    lr = torch.optim.lr_scheduler.LambdaLR(opt, _lrate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_padding_idx, reduction="sum")

    model.train()
    train_dl = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=BatchProcessor(src_tokenizer, trg_tokenizer),
    )
    for e in range(epoch):
        epoch_loss = n = 0
        for batch in tqdm.tqdm(train_dl):
            opt.zero_grad()

            output = model(batch["src"], batch["trg"][:, :-1])
            loss = criterion(
                output.view(-1, trg_vocab_size), batch["trg"][:, 1:].flatten()
            )

            loss.backward()
            opt.step(), lr.step()

            epoch_loss += loss.item()
            n += batch["trg"].ne(trg_padding_idx).sum().item()

        epoch_loss = epoch_loss / n
        print(f"Epoch {e + 1}/{epoch} Train Loss : {epoch_loss}")

    model.eval()
    eval_loss = n = 0
    valid_dl = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchProcessor(src_tokenizer, trg_tokenizer),
    )
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_dl):
            oo = model(batch["src"], batch["trg"][:, :-1])
            loss = criterion(oo.view(-1, trg_vocab_size), batch["trg"][:, 1:].flatten())

            eval_loss += loss.item()
            n += batch["trg"].ne(trg_padding_idx).sum().item()
    eval_loss /= n
    print(f"Total Evaluation Loss : {eval_loss}")

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # model params
    parser.add_argument("--n", type=int, nargs="?")
    parser.add_argument("--d-model", type=int, nargs="?")
    parser.add_argument("--d-ff", type=int, nargs="?")
    parser.add_argument("--h", type=int, nargs="?")
    parser.add_argument("--d-k", type=int, nargs="?")
    parser.add_argument("--d-v", type=int, nargs="?")
    parser.add_argument("--p-drop", type=int, nargs="?")

    # main args
    parser.add_argument("--model-path", type=str, default=".data/model.pb")
    parser.add_argument("--root-data-path", type=str, nargs="?")
    parser.add_argument("--dataset", type=str, nargs="?")
    parser.add_argument("--subset", type=str, nargs="?")

    # train hyperparams
    parser.add_argument("--e-ls", type=int, nargs="?")
    parser.add_argument("--epoch", type=int, nargs="?")
    parser.add_argument("--batch-size", type=int, nargs="?")
    parser.add_argument("--warmup-steps", type=int, nargs="?")

    args = vars(parser.parse_args())
    print(args)
    model_path = args.pop("model_path")

    trained_model = train(**args)
    torch.save(trained_model, model_path)
