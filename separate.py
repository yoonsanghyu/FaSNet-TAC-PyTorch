#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from data_eachbatch import AudioDataset, EvalAudioDataLoader
from model_2d_real3 import MCTasNet
from pit_criterion import cal_loss


parser = argparse.ArgumentParser('Separate speech using Conv-TasNet')
parser.add_argument('--model_path', type=str, default='temp_best.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture wav files')
parser.add_argument('--mix_json', type=str, default=None,
                    help='Json file including mixture wav files')
parser.add_argument('--out_dir', type=str, default='D:/exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
# parser.add_argument('--wav_out', default=1, type=int,
#                     help='Save separated wav file')

parser.add_argument('--N', default=512, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=32, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=128, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--Sc', default=128, type=int,
                    help="Number of channels in skip-connection paths' 1 x 1-conv blocks")
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=3, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Maximum number of speakers')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mic', default=6, type=int, help='number of microphone')



def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    model = MCTasNet(args.N, args.L, args.B, args.Sc, args.H, args.X, args.R, args.P, args.C, args.mic)
    
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
            name = k.replace("module.", "")    # remove 'module.'
            state_dict[name] = v
        model.load_state_dict(state_dict)
    
    print(model)
    model.eval()

    eval_dataset = AudioDataset('test', batch_size = 1, sample_rate = args.sample_rate, nmic = args.mic)
    eval_loader = EvalAudioDataLoader(eval_dataset, batch_size=1, num_workers=8)
    # Load data

    os.makedirs(args.out_dir, exist_ok=True)

    def write(inputs, filename, sr=args.sample_rate):
        librosa.output.write_wav(filename, inputs, sr)# norm=True)

    with torch.no_grad():
        t = tqdm(total=len(eval_dataset), mininterval=0.5)
        for i, data in enumerate(eval_loader):
            
            #mix_batch, infos = data
            # info = [_info[:-1] for _info in infos]
            #mix_batch = torch.stack(mix_batch)
            padded_mixture, mixture_lengths, padded_source = data
            
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            '''
            if args.use_cuda:
                mix_batch = mix_batch.cuda()
            '''
            estimate_source = model(padded_mixture)
            
            for j in range(estimate_source.size()[0]):
                #padded, sample_info, file_name = infos[j]
                scs = estimate_source[j].cpu().numpy()
                '''
                if padded:
                    mix = mix_batch[j][:, :-sample_info[1]].cpu().numpy()
                    scs = scs[:, :-sample_info[1]]
                
                
                mix = mix_batch[j].cpu().numpy()
                
                power = np.sqrt((mix**2).sum()/len(mix))
                
                # if args.wav_out:
                file_name = os.path.join(args.out_dir,
                                         os.path.basename(file_name).strip('.wav'))
                write(mix, file_name + '.wav')
                '''
                
                power = np.sqrt((padded_mixture.cpu().numpy()**2).sum()/len(padded_mixture.cpu().numpy()))
                for k, src in enumerate(scs):
                    source = src*(power/np.sqrt((src**2).sum()/len(padded_mixture)))
                    write(source, 'utt{0}_s{1}.wav'.format(i+1,k+1))
            
            t.update()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)

