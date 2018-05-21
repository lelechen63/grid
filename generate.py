import argparse
import math
import os
import random as rn
import shutil
import subprocess

import h5py
import librosa
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from plot_face import facePainter
import torch
import utils
from model import SPCH2FLM
#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = '128'
np.random.seed(128)
rn.seed(128)
torch.manual_seed(128)
#-----------------------------------------#

batchsize = 1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-folder", type=str, help="input speech folder")
parser.add_argument("-m", "--model", type=str, help="DNN model to use")
parser.add_argument("-n", "--num-frames", type=int, help="Number of frames", default=7)
parser.add_argument("-d", "--lmark-dim", type=int, help="Dimension of lmarks", default=3)
parser.add_argument("-s", "--is-seq", type=bool, help="Is sequence?", default=False)
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()

sequence = args.is_seq
lmark_dim = args.lmark_dim

MS = np.load('../../data/AAM_DATA/data_6/mean_shape.npy')
EIGVECS = np.load('../../data/AAM_DATA/data_6/S.npy')
from model import SPCH2FLM
def generateFace(root, filename):
    speech, sr = librosa.load(os.path.join(root, filename), sr=fs)
    speech = speech / np.max(np.abs(speech))
    # speech = librosa.effects.pitch_shift(speech, sr, n_steps=4)
    speech_orig = speech[:]

    increment = int(0.04*fs)
    upper_limit = speech.shape[0]
    lower = 0
    predicted = np.zeros((0, num_features_Y))
    flag = 0

    speech = np.insert(speech, 0, np.zeros((int(increment*num_frames/2))))
    speech = np.append(speech, np.zeros((int(increment*num_frames/2))))

    while True:
        cur_features = np.zeros((1, 1, num_frames*increment))

        local_speech = speech[lower:lower+num_frames*increment]
        # print local_speech.shape
        # exit()
        if local_speech.shape[0] < num_frames*increment:
            local_speech = np.append(local_speech, np.zeros((num_frames*increment-local_speech.shape[0])))
            flag = 1
        # print local_speech.shape
        # local_speech = preemphasis(local_speech)
        # local_speech = RMSNorm(local_speech)

        cur_features[0, 0, :] = local_speech
        lower += increment
        if flag:
            break

        # print cur_features.shape
        cur_features = torch.from_numpy(cur_features).float()

        pred = model(cur_features).data.numpy()

        # ---------------- BOOST 6 COMP ------------------ #
        pred[0, 1:6] *= 2*np.array([1.2, 1.4, 1.6, 1.8, 2.0])
        # ------------------------------------------------ #

        # ---------------- BOOST 20 COMP ----------------- #
        # gain = np.ones((1, 20))
        # gain[0, 1:6] = 2*np.array([1.2, 1.4, 1.6, 1.8, 2.0])
        # gain = np.linspace(1.0, 3.0, 20)
        # print(gain)
        # exit()
        # pred *= gain
        # ------------------------------------------------ #

        print(np.mean(pred), np.min(pred), np.max(pred))

        # print pred.shape
        if sequence:
            pred = pred[:, mid, :]

        # pred[:, :15] = 0
        # print(pred)

        pred = (MS + np.dot(pred, EIGVECS))
        # print np.mean(pred)
        predicted = np.append(predicted, np.reshape(pred[0, :], (1, num_features_Y)), axis=0)
    print (predicted.shape, np.min(predicted[:]), np.max(predicted[:]))

    if len(predicted.shape) < 3:
        if lmark_dim == 2:
            predicted = np.reshape(predicted, (predicted.shape[0], int(predicted.shape[1]/2), 2))
        elif lmark_dim == 3:
            predicted = np.reshape(predicted, (predicted.shape[0], int(predicted.shape[1]/3), 3))

    np.save(os.path.join(output_path, os.path.splitext(filename)[0]+'.npy'), predicted)

    fp = facePainter(predicted, speech_orig)
    fp.paintFace(output_path, os.path.splitext(filename)[0]+'_painted')

    # utils.write_video(predicted, speech_orig, 8000, output_path, os.path.splitext(filename)[0]+'_2D',  [-0.25, 0.25], [-0.25, 0.25])
    # if lmark_dim == 3:
    #     utils.write_video3D(predicted, speech_orig, 8000, output_path, os.path.splitext(filename)[0]+'_3D', [-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25])
    # utils.write_video(predicted, speech_orig, fs, output_path, os.path.splitext(filename)[0], [-0.25, 0.25], [-0.25, 0.25])

output_path = args.out_fold
if lmark_dim == 2:
    num_features_Y = 68*2
elif lmark_dim == 3:
    num_features_Y = 68*3
num_frames = args.num_frames
mid = int(math.floor(num_frames/2.0))
fs = 8000

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)


params = {}
params['CUDA'] = torch.cuda.is_available()
params['DEVICE'] = torch.device("cuda" if params['CUDA'] else "cpu") 
params['kwargs'] = {'num_workers': 1, 'pin_memory': True} if params['CUDA'] else {}

model = SPCH2FLM().to(params['DEVICE'])
test_folder = args.in_folder
model.load_state_dict(torch.load(args.model, map_location="cuda" if params['CUDA'] else "cpu"))

for root, dirs, files in os.walk(test_folder):
    for filename in files:
        if filename.endswith(('.WAV', '.wav', '.flac')):
            print (os.path.join(root, filename))
            generateFace(root, filename)
