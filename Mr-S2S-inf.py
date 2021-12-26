"""
results saved at
https://drive.google.com/file/d/1nOuGR_CF-wV9YnsP9BmWpFSlVR6m907s/view?usp=sharing
"""


import json
import os

import h5py
import numpy as np
import tensorflow
from scipy.io import loadmat, savemat
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import model_from_json

WORKDIR = '/data/long/data/data_spvsd/'
SAVEDIR = '/data/long/data/data_spvsd/norm/'
MDIR = '/data/long/experiment/buda/'
N_PREDICTION = 500

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


class ConstMultiplierLayer(Layer):
    def __init__(self, val=0.1, **kwargs):
        self.val = val
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    # Create a function to initialize this layer with a specific value
    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer=Constant(value=self.val),
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return tensorflow.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def get_buda_mat(filename, DATA_DIR, tag):
    file = os.path.join(DATA_DIR, filename)
    try:
        with h5py.File(file, 'r') as f:
            keys = list(f.keys())[0]
            print('key-', keys)
            annots = np.array(f[tag])
    except:
        annots = loadmat(file)[tag]
        annots = np.transpose(annots, (3, 4, 2, 0, 1))
        return annots, False
    return annots, True


def run_norm(img, flag):
    print(img.shape)
    _, _, _, nrow, ncol = img.shape
    imgref = img.reshape(-1, nrow, ncol)
    imgref = np.array([imgref[nc] / np.mean(imgref[nc]) for nc in range(len(imgref))])
    if flag:
        imgref = np.array([np.rot90(imgref[nc], 2) for nc in range(len(imgref))])
    else:
        imgref = np.array([np.fliplr(np.rot90(imgref[nc], 1)) for nc in range(len(imgref))])
    nc = 10
    return imgref



def load_model(h5path, model_path, train_in):
    # get the shape of the data
    _, nrow, ncol, _ = train_in.shape
    print(train_in.shape)

    # read the model file
    with open(model_path, 'r') as file:
        string_model = json.load(file)

    model = model_from_json(json.dumps(string_model))

    model.load_weights(h5path)
    return model

def reverse_norm(img, imgref, flag):
    img = np.array([np.flipud(np.rot90(img[nc], 3)) for nc in range(len(img))])
    print ('img,', img.shape)
    nnum, nrow, ncol = img.shape
    ncase = int(nnum / 3 / 3)
    img = img.reshape((3, 3, ncase, nrow, ncol))
    outs = np.zeros(imgref.shape)
    for i in range(3):
        for j in range(3):
            for k in range(ncase):
                outs[i, j, k] = img[i, j, k] / np.mean(img[i, j, k]) * np.mean(imgref[i, j, k])
    savemat('outs', {'outs': outs})

def get_norm_v3(shot4, fullpath):
    # load the case
    annots, flag = get_buda_mat(shot4, fullpath, shot4.split('.')[0])
    inputs = run_norm(annots, flag)
    annots_out, flag = get_buda_mat('p3_img_buda_4shot_s2_out_v2.mat', WORKDIR, 'p3_img_buda_4shot_s2_out_v2')
    gt = run_norm(annots_out, flag)

    # load the model
    mid, cid = '3', '30'
    h5_path = MDIR + mid + '/' + mid + cid + '.ckpt'
    model_path = MDIR + mid + '/' + mid + '_model.json'
    model = load_model(h5_path, model_path, inputs[0:1, ..., np.newaxis])

    print('begin running prediction', inputs.shape)
    outs = np.zeros((N_PREDICTION, inputs.shape))
    for nc in N_PREDICTION:
        outs[nc] = model.predict(inputs[..., np.newaxis], batch_size=1, verbose=0)
    outs = np.mean(outs, axis=0)[0]
    print (outs.shape)
    reverse_norm(outs[...,0]*0.8+0.2*inputs, annots, flag)

if __name__ == "__main__":
    shot4 = 'img_buda_4shot.mat'
    fullpath = os.path.join(WORKDIR, 'subject1')
    get_norm_v3(shot4, fullpath,)
