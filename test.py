import torch


def test(model):
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', nargs='?', type=str, default='./model.pb')
    args = parser.parse_args()

    model = torch.load(args.model_path)
    test(model)
