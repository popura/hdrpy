
# coding: utf-8

# In[2]:

import numpy as np
import scipy.misc
import re
import sys
import math
import os
import random
import matplotlib.pyplot as plt
from pfm_format import PFMFormat as PFM
import imageio

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def readLDR(file, sz):
    try:
        x_buffer = scipy.misc.imread(file)
        x_buffer = scipy.misc.imresize(x_buffer, sz)
        x_buffer = x_buffer.astype(np.float32)/255.0
        x_buffer = x_buffer[np.newaxis,:,:,:]
        
        return x_buffer
    
    except Exception as e:
        raise IOException("Failed reading LDR image: %s"%e)

def writeLDR(img, file, exposure=0):
    sc = np.power(np.power(2.0, exposure), 0.5)

    try:
        scipy.misc.toimage(sc*np.squeeze(img), cmin=0.0, cmax=1.0).save(file)
    except Exception as e:
        raise IOException("Failed writing LDR image: %s"%e)

def load_data(train_number, test_number):
    ldr_dir = os.path.join("data", "ldr")
    hdr_dir = os.path.join("data", "hdr")
    
    ldr_file_names = []
    hdr_file_names = []
    for name in sorted(os.listdir(ldr_dir)):
        if os.path.isfile(os.path.join(ldr_dir, name)):
            ldr_file_names.append(os.path.join(ldr_dir, name))
    for name in sorted(os.listdir(hdr_dir)):
        if os.path.isfile(os.path.join(hdr_dir, name)):
            hdr_file_names.append(os.path.join(hdr_dir, name))
    
    index = np.arange(0, len(ldr_file_names), 1)
    random.seed(0)
    random.shuffle(index)
    
    x_train = np.zeros((train_number, 320, 320, 3))
    y_train = np.zeros((train_number, 320, 320, 3))
    for i in range(train_number):
        x_train[i] = readLDR(ldr_file_names[index[i]], (320, 320))
        y_train[i] = imageio.imread(hdr_file_names[index[i]])

    x_test = np.zeros((test_number, 320, 320, 3))
    y_test = np.zeros((test_number, 320, 320, 3))
    for i in range(train_number, train_number+test_number):
        x_test[i-train_number] = readLDR(ldr_file_names[index[i]], (320, 320))
        y_test[i-train_number] = imageio.imread(hdr_file_names[index[i]])

    return (x_train, y_train), (x_test, y_test)

def save_data(ldr_data, hdr_data, hdr_truth):
    output_dir = os.path.join("data_out")
    for i in range(len(ldr_data)):
        scipy.misc.toimage(ldr_data[i], cmin=0.0, cmax=1.0).save(os.path.join(output_dir, "input" + str(i) + ".png"))
        imageio.imwrite(os.path.join(output_dir, "predict" + str(i) + ".hdr"), \
                hdr_data[i].astype(np.float32))
        imageio.imwrite(os.path.join(output_dir, "groundtruth" + str(i) + ".hdr"), \
                hdr_truth[i].astype(np.float32))

def load_data2(train_number, test_number):
    ldr_dir = os.path.join("data", "ldr")
    reinhard_dir = os.path.join("data", "reinhard")
    
    ldr_file_names = []
    reinhard_file_names = []
    for name in sorted(os.listdir(ldr_dir)):
        if os.path.isfile(os.path.join(ldr_dir, name)):
            ldr_file_names.append(os.path.join(ldr_dir, name))
    for name in sorted(os.listdir(reinhard_dir)):
        if os.path.isfile(os.path.join(reinhard_dir, name)):
            reinhard_file_names.append(os.path.join(reinhard_dir, name))
    
    index = np.arange(0, len(ldr_file_names), 1)
    random.seed(0)
    random.shuffle(index)
    
    x_train = np.zeros((train_number, 320, 320, 3))
    y_train = np.zeros((train_number, 320, 320, 3))
    for i in range(train_number):
        x_train[i] = readLDR(ldr_file_names[index[i]], (320, 320))
        y_train[i] = readLDR(reinhard_file_names[index[i]], (320, 320))

    x_test = np.zeros((test_number, 320, 320, 3))
    y_test = np.zeros((test_number, 320, 320, 3))
    for i in range(train_number, train_number+test_number):
        x_test[i-train_number] = readLDR(ldr_file_names[index[i]], (320, 320))
        y_test[i-train_number] = readLDR(reinhard_file_names[index[i]], (320, 320))

    return (x_train, y_train), (x_test, y_test)

def save_data2(ldr_data, ldr_predict, ldr_truth):
    output_dir = os.path.join("data_out")
    for i in range(len(ldr_data)):
        scipy.misc.toimage(ldr_data[i], cmin=0.0, cmax=1.0).save(os.path.join(output_dir,\
                "input" + str(i) + ".png"))
        scipy.misc.toimage(ldr_predict[i], cmin=0.0, cmax=1.0).save(os.path.join(output_dir,\
                "predict" + str(i) + ".png"))
        scipy.misc.toimage(ldr_truth[i], cmin=0.0, cmax=1.0).save(os.path.join(output_dir,\
                "truth" + str(i) + ".png"))
