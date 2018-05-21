import argparse
import json
import math
import os
import random as rn
import shutil
import sys
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm, trange
import torch.nn.init as weightinit
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataGen import FaceLandmarksDataset
from model import SPCH2FLM

#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = '128'
np.random.seed(128)
rn.seed(128)
torch.manual_seed(128)
#-----------------------------------------#

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-file", type=str, help="Input file containing train data", default=None)
    parser.add_argument("-o", "--out-fold", type=str, help="output folder", default='../models/def')
    parser.add_argument("-gpu", "--gpu_id", type=str, help="gpu id", default='3')
    args = parser.parse_args()

    params = {}
    params['LEARNING_RATE'] = 1e-04
    params['BATCHSIZE'] = 64
    params['NUMFRAMES'] = 7
    params['INCREMENT'] = int(0.04*8000)
    params['MID'] = int(math.floor(params['NUMFRAMES']/2.0))
    params['LMARKDIM'] = 3
    params['OUT_SHAPE'] = 6
    params['IN_PATH'] = args.in_file
    params['OUT_PATH'] = args.out_fold
    params['GPU_ID'] = args.gpu_id
    params['NUM_IT'] = int(2108771/params['BATCHSIZE'])
    params['NUM_EPOCH'] = 200
    params['SEQ'] = False
   
    if not os.path.exists(params['OUT_PATH']):
        os.makedirs(params['OUT_PATH'])
    else:
        shutil.rmtree(params['OUT_PATH'])
        os.mkdir(params['OUT_PATH'])

    with open(os.path.join(params['OUT_PATH'], 'params.txt'), 'w') as file:
        file.write(json.dumps(params, sort_keys=True, separators=('\n', ':')))

    params['CUDA'] = torch.cuda.is_available()
    params['DEVICE'] = torch.device("cuda" if params['CUDA'] else "cpu") 
    params['kwargs'] = {'num_workers': 10, 'pin_memory': True} if params['CUDA'] else {}

    return params

def train():
    params = initParams()

    os.environ["CUDA_VISIBLE_DEVICES"] = params['GPU_ID']
    print params['GPU_ID']
    print params['CUDA'] 

    model = SPCH2FLM().to(device=params['DEVICE'])
    print model
 
    dataset = FaceLandmarksDataset(params)

    
    optimizer = optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=params['BATCHSIZE'], 
                                               shuffle=True, 
                                               **params['kwargs'])

    prev_loss = np.inf
    model.train()
    print '++++='
    for epoch in tqdm(range(params['NUM_EPOCH'])):
        lossDict = defaultdict(list)
        print '++++='
        with trange(len(train_loader)) as t:
            for i in t:
                print '++++='
                (data, target) = next(iter(train_loader))
                print '++++='
                data, target = data.to(device=params['DEVICE']), target.to(device=params['DEVICE'])
                print '++++='
                optimizer.zero_grad()
                output = model(data)
                loss = F.l1_loss(output, target, reduce=True)
                lossDict['abs'].append(loss.item())
                loss.backward()
                print '++++='
                optimizer.step()
                t.set_description("loss: %.5f, cur_loss: %.5f" % (np.mean(lossDict['abs']), lossDict['abs'][-1]))

        cur_loss = loss.item()
        if prev_loss > cur_loss:
            print("Loss has improved from %.5f to %.5f" % (prev_loss, cur_loss))
            prev_loss = cur_loss
            torch.save(model.state_dict(), os.path.join(params['OUT_PATH'], 'SPCH2FLM.pt'))


if __name__ == "__main__":
    train()

# for batch_idx, (data, target) in tqdm(enumerate(train_loader)):