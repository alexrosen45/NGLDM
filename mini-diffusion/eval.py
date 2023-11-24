"""
python3 eval.py --fake_data_dir path/to/fake
"""

import torch
# from torch import nn, optim
import torch.nn.functional as F
import argparse
import blobfile as bf
import numpy as np
import torch.multiprocessing

from ignite.engine import *
from ignite.metrics import *
import data
torch.multiprocessing.set_sharing_strategy('file_system')

#TODO: Add better inception model for this domain

def eval_step(engine, batch):
    return batch


def main():
    args = create_argparser().parse_args()
    # Image dimension
    d = args.image_size

    _, _, true_data, _ = data.load_all_data('data')

    fake_data = load_fake_data(args.fake_data_dir)

    # Will extract this into a function later
    x = torch.from_numpy(true_data[:args.num_samples])
    # Change to square image
    x = np.reshape(x, (x.size()[0], 1, d, d))
    # Repeat in three channels
    x = x.repeat(1, 3, 1, 1)
    # Normalize brightness values
    x = x / 127.5 - 1
    # Expand to 299x299
    all_true = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

    x = torch.from_numpy(fake_data[:args.num_samples])
    # Change to square image
    x = np.reshape(x, (x.size()[0], 1, d, d))
    # Repeat in three channels
    x = x.repeat(1, 3, 1, 1)
    # Normalize brightness values
    x = x / 127.5 - 1
    # Expand to 299x299
    all_fake = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

    default_evaluator = Engine(eval_step)
    metric = FID()
    metric.attach(default_evaluator, "fid")

    state = default_evaluator.run([[all_true.float(), all_fake.float()]])
    print(state.metrics["fid"])


def load_fake_data(base_samples):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
    return image_arr


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        fake_data_dir="",
        image_size=8,
        num_samples=1000
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
