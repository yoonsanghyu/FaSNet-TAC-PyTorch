#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:09:09 2020

@author: yoonsanghyu
"""

# Created on 2018/12
# Author: Kaituo XU

import argparse
from collections import OrderedDict

import numpy as np
import torch
from mir_eval.separation import bss_eval_sources

from FaSNet import FaSNet_TAC
from data import AudioDataset, EvalAudioDataLoader
from pit_criterion import calc_loss
from utility.metrics import calc_SISNRi, calc_SDRi


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3:  # [B, C, T]
            results.append(input[:, :length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]  
            results.append(input[:length].view(-1).cpu().numpy())
    return results


parser = argparse.ArgumentParser('Evaluate separation performance using FaSNet + TAC')
parser.add_argument('--model_path', type=str, default='exp/tmp/temp_best.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cal_sdr', type=int, default=1,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=1, help='Whether use GPU to separate speech')

# General config
# Task related
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')

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


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model

    model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer,
                       segment_size=args.segment_size,
                       nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)

    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()

    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

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
    data_loader = EvalAudioDataLoader(dataset, batch_size=1, num_workers=8)

    sisnr_array = []
    sdr_array = []
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data

            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            x = torch.rand(2, 6, 32000)
            none_mic = torch.zeros(1).type(x.type())
            # Forward
            estimate_source = model(padded_mixture, none_mic.long())  # [M, C, T]

            loss, max_snr, estimate_source, reorder_estimate_source = \
                calc_loss(padded_source, estimate_source, mixture_lengths)

            M, _, T = padded_mixture.shape
            mixture_ref = torch.chunk(padded_mixture, args.mic, dim=1)[0]  # [M, ch, T] -> [M, 1, T]
            mixture_ref = mixture_ref.view(M, T)  # [M, 1, T] -> [M, T]

            mixture = remove_pad(mixture_ref, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)

            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = calc_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi
                    sdr_array.append(avg_SDRi)
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                # Compute SI-SNRi
                avg_SISNRi = calc_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                sisnr_array.append(avg_SISNRi)
                total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))

    np.save('sisnr.npy', np.array(sisnr_array))
    np.save('sdr.npy', np.array(sdr_array))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))




if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
