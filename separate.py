#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU
# Edited by: yoonsanghyu

import argparse
import os
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch

from FaSNet import FaSNet_TAC
from data import AudioDataset, EvalAudioDataLoader

parser = argparse.ArgumentParser('Separate speech using FaSNet + TAC')
parser.add_argument('--model_path', type=str, default='exp/tmp/temp_best.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--out_dir', type=str, default='C:/output_files/exp1',
                    help='Directory putting separated wav files')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')

# Network architecture
parser.add_argument('--enc_dim', default=64, type=int, help='Number of filters in autoencoder')
parser.add_argument('--win_len', default=4, type=int, help='Number of convolutional blocks in each repeat')
parser.add_argument('--context_len', default=16, type=int, help='context window size')
parser.add_argument('--feature_dim', default=64, type=int, help='feature dimesion')
parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden dimension')
parser.add_argument('--layer', default=4, type=int, help='Number of layer in dprnn step')
parser.add_argument('--segment_size', default=50, type=int, help="segment_size")
parser.add_argument('--nspk', default=2, type=int, help='Maximum number of speakers')
parser.add_argument('--mic', default=6, type=int, help='number of microphone')


def separate(args):
    # Load model
    model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer,
                       segment_size=args.segment_size,
                       nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)

    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()

    model_info = torch.load(args.model_path)
    try:
        model.load_state_dict(model_info['model_state_dict'])
    except KeyError:
        state_dict = OrderedDict()
        for k, v in model_info['model_state_dict'].items():
            name = k.replace("module.", "")  # remove 'module.'
            state_dict[name] = v
        model.load_state_dict(state_dict)

    print(model)
    model.eval()

    # Load data
    dataset = AudioDataset('test', batch_size=1, sample_rate=args.sample_rate, nmic=args.mic)
    eval_loader = EvalAudioDataLoader(dataset, batch_size=1, num_workers=8)

    os.makedirs(args.out_dir, exist_ok=True)

    def write(inputs, filename, sr=args.sample_rate):
        sf.write(filename, inputs, sr)  # norm=True)

    with torch.no_grad():
        # t = tqdm(total=len(eval_dataset), mininterval=0.5)
        for i, data in enumerate(eval_loader):

            padded_mixture, mixture_lengths, padded_source = data

            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            x = torch.rand(2, 6, 32000)
            none_mic = torch.zeros(1).type(x.type())
            # Forward
            estimate_source = model(padded_mixture, none_mic.long())  # [M, C, T]

            for j in range(estimate_source.size()[0]):

                scs = estimate_source[j].cpu().numpy()

                power = np.sqrt((padded_mixture.cpu().numpy() ** 2).sum() / len(padded_mixture.cpu().numpy()))
                for k, src in enumerate(scs):
                    this_dir = os.path.join(args.out_dir, 'utt{0}'.format(i + 1))
                    if not os.path.exists(this_dir):
                        os.makedirs(this_dir)
                    source = src * (power / np.sqrt((src ** 2).sum() / len(padded_mixture)))
                    write(source, os.path.join(this_dir, 's{0}.wav'.format(k + 1)))

            # t.update()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)
