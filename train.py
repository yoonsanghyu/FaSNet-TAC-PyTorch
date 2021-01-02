# -*- coding: utf-8 -*-
"""
Created on 2018/12
Author: Kaituo XU

Edited by: yoonsanghyu  2020/04

"""

import argparse
import torch

from data import AudioDataset, AudioDataLoader

from FaSNet import FaSNet_TAC
from solver import Solver


parser = argparse.ArgumentParser( "FaSNet + TAC model")

# General config
# Task related
parser.add_argument('--tr_json', type=str, default=None, help='path to tr.json')
parser.add_argument('--cv_json', type=str, default=None, help='path to cv.json')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--segment', default=4, type=float, help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=20, type=float, help='max audio length (seconds) in cv, to avoid OOM issue.')

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

# Training config
parser.add_argument('--use_cuda', type=int, default=1, help='Whether use GPU')
parser.add_argument('--epochs', default=120, type=int, help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=1, type=int, help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int, help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float, help='Gradient norm threshold to clip')


# minibatch
parser.add_argument('--shuffle', default=1, type=int, help='reshuffle the data at every epoch')
parser.add_argument('--drop', default=0, type=int, help='drop files shorter than this')
parser.add_argument('--batch_size', default=3, type=int, help='Batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to generate minibatch')


# optimizer
parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam'], help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float, help='weight decay (L2 penalty)')


# save and load model
parser.add_argument('--save_folder', default='exp/tmp', help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int, help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--tseed', default=-1, help='Torch random seed', type=int)
parser.add_argument('--nseed', default=-1, help='Numpy random seed', type=int)


# logging
parser.add_argument('--print_freq', default=1000, type=int, help='Frequency of printing training infomation')

def main(args):

    
    # data
    tr_dataset = AudioDataset('tr', batch_size = args.batch_size, sample_rate= args.sample_rate, nmic = args.mic)
    cv_dataset = AudioDataset('val', batch_size = args.batch_size, sample_rate= args.sample_rate, nmic = args.mic)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1, shuffle=args.shuffle, num_workers=0) #num_workers=0 for PC
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1, num_workers=0) #num_workers=0 for PC

    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    # model
    model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size, 
                           nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)
    
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)
    
    #print(model)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
        
    
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
