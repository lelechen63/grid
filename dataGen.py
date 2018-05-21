import os
import numpy as np
import h5py
import random as rn

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
np.random.seed(128)
rn.seed(128)
#-----------------------------------------#

class FaceLandmarksDataset(Dataset):

    def __init__(self, params):
        self.params = params


        self.lmarkDSET =  h5py.File(os.path.join(self.params['IN_PATH'],'flmark.h5' ) , 'r')['flmark'][:,:,:]
        self.speechDSET =  h5py.File(os.path.join(self.params['IN_PATH'],'speech.h5' ) , 'r')['speech']

        self.num_samples = self.lmarkDSET.shape[0]
        self.idxList = [(i, j) for i in range(self.num_samples) for j in range(75-self.params['NUMFRAMES']-1)]
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
        rn.shuffle(self.idxList)
        
    def __len__(self):
        return len(self.idxList)

    def __getitem__(self, idx):
       i, j = self.idxList[idx]

       rnd_dset = np.random.randint(0, high=5, size=[1, ])[0]
       rnd_dB = np.random.randint(0, high=len(self.augList), size=[1, ])[0]

       cur_lmark = self.lmarkDSET[i, j:j+self.params['NUMFRAMES'], :]
       cur_speech = np.reshape(self.speechDSET[rnd_dset][i, j*self.params['INCREMENT']:(j+self.params['NUMFRAMES'])*self.params['INCREMENT']],
                                [1, self.params['NUMFRAMES']*self.params['INCREMENT']])

       cur_speech = cur_speech*np.power(10.0, self.augList[rnd_dB]/20.0)

       return cur_speech, cur_lmark[self.params['MID'], :]