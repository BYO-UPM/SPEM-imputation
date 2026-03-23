import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


instances_dict_path_train = './instances_dict_train.json'
with open(instances_dict_path_train, 'r') as f:
    instances_dict_train = json.load(f)
    
instances_dict_path_test = './instances_dict_test.json'
with open(instances_dict_path_test, 'r') as f:
    instances_dict_test = json.load(f)

smoothpur_1_4_train = np.load('Data_in/Train/SmoothPur_1_4.npy')
smoothpur_5_8_train = np.load('Data_in/Train/SmoothPur_5_8.npy')
smoothpur_9_10_train = np.load('Data_in/Train/SmoothPur_9_10.npy')
smoothpur_11_12_train = np.load('Data_in/Train/SmoothPur_11_12.npy')

smoothpur_1_4_test = np.load('Data_in/Test/SmoothPur_1_4.npy')
smoothpur_5_8_test = np.load('Data_in/Test/SmoothPur_5_8.npy')
smoothpur_9_10_test = np.load('Data_in/Test/SmoothPur_9_10.npy')
smoothpur_11_12_test = np.load('Data_in/Test/SmoothPur_11_12.npy')

print('TRAIN')
print(instances_dict_train)
print('length of instances_dict : ' ,len(instances_dict_train))
print('-' * 100)
print('length of smoothpur_1_4 : ' ,len(smoothpur_1_4_train))
print('length of smoothpur_5_8 : ' ,len(smoothpur_5_8_train))
print('length of smoothpur_9_10 : ' ,len(smoothpur_9_10_train))
print('length of smoothpur_11_12 : ' ,len(smoothpur_11_12_train))
print('-' * 100)
print('type of smoothpur_1_4 : ' , type(smoothpur_1_4_train))
print('type of smoothpur_5_8 : ' , type(smoothpur_5_8_train))
print('type of smoothpur_9_10 : ' , type(smoothpur_9_10_train))
print('type of smoothpur_11_12 : ' , type(smoothpur_11_12_train))
print('-' * 100)
print('shape of smoothpur_1_4 : ' , smoothpur_1_4_train.shape)
print('shape of smoothpur_5_8 : ' , smoothpur_5_8_train.shape)
print('shape of smoothpur_9_10 : ' , smoothpur_9_10_train.shape)
print('shape of smoothpur_11_12 : ' , smoothpur_11_12_train.shape)

print('TEST')
print(instances_dict_test)
print('length of instances_dict : ' ,len(instances_dict_test))
print('-' * 100)
print('length of smoothpur_1_4 : ' ,len(smoothpur_1_4_test))
print('length of smoothpur_5_8 : ' ,len(smoothpur_5_8_test))
print('length of smoothpur_9_10 : ' ,len(smoothpur_9_10_test))
print('length of smoothpur_11_12 : ' ,len(smoothpur_11_12_test))
print('-' * 100)
print('type of smoothpur_1_4 : ' , type(smoothpur_1_4_test))
print('type of smoothpur_5_8 : ' , type(smoothpur_5_8_test))
print('type of smoothpur_9_10 : ' , type(smoothpur_9_10_test))
print('type of smoothpur_11_12 : ' , type(smoothpur_11_12_test))
print('-' * 100)
print('shape of smoothpur_1_4 : ' , smoothpur_1_4_test.shape)
print('shape of smoothpur_5_8 : ' , smoothpur_5_8_test.shape)
print('shape of smoothpur_9_10 : ' , smoothpur_9_10_test.shape)
print('shape of smoothpur_11_12 : ' , smoothpur_11_12_test.shape)


def flatten_smoothpur_1_8(original_arr):
    arr_shape = original_arr.shape
    flattened_list = []
    for i in range(arr_shape[0]):
      for j in range(arr_shape[1]):
        flattened_list.append(original_arr[i][j][0])
        flattened_list.append(original_arr[i][j][1])
    flattened_arr = np.array(flattened_list)
    return flattened_arr

def flatten_smoothpur_9_12(original_arr):
    arr_shape = original_arr.shape
    flattened_x_list = []
    flattened_y_list = []
    for i in range(arr_shape[0]):
      for j in range(arr_shape[1]):
        flattened_x_list.append(original_arr[i][j][0][0])
        flattened_x_list.append(original_arr[i][j][0][1])
        flattened_y_list.append(original_arr[i][j][1][0])
        flattened_y_list.append(original_arr[i][j][1][1])
    flattened_x_arr = np.array(flattened_x_list)
    flattened_y_arr = np.array(flattened_y_list)
    return flattened_x_arr, flattened_y_arr

def downsample(signals, new_len):
    downsampled_signals = []
    for signal in signals:
        signal = signal.reshape(-1)  # Flatten the signal
        downsample_factor = len(signal) // new_len
        indices = np.arange(0, len(signal), downsample_factor)
        downsampled_signal = signal[indices[:new_len]]  # Ensure the length matches new_len
        downsampled_signals.append(downsampled_signal)
    return np.array(downsampled_signals)


newL = 500

print("TRAIN")

flat_1_4_train = flatten_smoothpur_1_8(smoothpur_1_4_train)
flat_5_8_train = flatten_smoothpur_1_8(smoothpur_5_8_train)
flat_x_9_10_train, flat_y_9_10_train = flatten_smoothpur_9_12(smoothpur_9_10_train)
flat_x_11_12_train, flat_y_11_12_train = flatten_smoothpur_9_12(smoothpur_11_12_train)

mean_1_4_train = np.nanmean(flat_1_4_train)
std_1_4_train = np.nanstd(flat_1_4_train)
flat_1_4_normalized_train = (flat_1_4_train - mean_1_4_train) / std_1_4_train

mean_5_8_train = np.nanmean(flat_5_8_train)
std_5_8_train = np.nanstd(flat_5_8_train)
flat_5_8_normalized_train = (flat_5_8_train - mean_5_8_train) / std_5_8_train

mean_x_9_10_train = np.nanmean(flat_x_9_10_train)
std_x_9_10_train = np.nanstd(flat_x_9_10_train)
flat_x_9_10_normalized_train = (flat_x_9_10_train - mean_x_9_10_train) / std_x_9_10_train

mean_y_9_10_train = np.nanmean(flat_y_9_10_train)
std_y_9_10_train = np.nanstd(flat_y_9_10_train)
flat_y_9_10_normalized_train = (flat_y_9_10_train - mean_y_9_10_train) / std_y_9_10_train

mean_x_11_12_train = np.nanmean(flat_x_11_12_train)
std_x_11_12_train = np.nanstd(flat_x_11_12_train)
flat_x_11_12_normalized_train = (flat_x_11_12_train - mean_x_11_12_train) / std_x_11_12_train

mean_y_11_12_train = np.nanmean(flat_y_11_12_train)
std_y_11_12_train = np.nanstd(flat_y_11_12_train)
flat_y_11_12_normalized_train = (flat_y_11_12_train - mean_y_11_12_train) / std_y_11_12_train

down_1_4_train = downsample(flat_1_4_normalized_train, newL)
down_5_8_train = downsample(flat_5_8_normalized_train, newL)
down_x_9_10_train, down_y_9_10_train = downsample(flat_x_9_10_normalized_train, newL), downsample(flat_y_9_10_normalized_train, newL)
down_x_11_12_train, down_y_11_12_train = downsample(flat_x_11_12_normalized_train, newL), downsample(flat_y_11_12_normalized_train, newL)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_train.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_train.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_train.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_train.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_train.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_train.shape)
print('-' * 100)
print('shape of downsampled smoothpur_1_4 :', down_1_4_train.shape)
print('shape of downsampled smoothpur_5_8 :', down_5_8_train.shape)
print('shape of downsampled smoothpur_x_9_10 :', down_x_9_10_train.shape)
print('shape of downsampled smoothpur_y_9_10 :', down_y_9_10_train.shape)
print('shape of downsampled smoothpur_x_11_12 :', down_x_11_12_train.shape)
print('shape of downsampled smoothpur_y_11_12 :', down_y_11_12_train.shape)


print("TEST")

flat_1_4_test = flatten_smoothpur_1_8(smoothpur_1_4_test)
flat_5_8_test = flatten_smoothpur_1_8(smoothpur_5_8_test)
flat_x_9_10_test, flat_y_9_10_test = flatten_smoothpur_9_12(smoothpur_9_10_test)
flat_x_11_12_test, flat_y_11_12_test = flatten_smoothpur_9_12(smoothpur_11_12_test)

mean_1_4_test = np.nanmean(flat_1_4_test)
std_1_4_test = np.nanstd(flat_1_4_test)
flat_1_4_normalized_test = (flat_1_4_test - mean_1_4_test) / std_1_4_test

mean_5_8_test = np.nanmean(flat_5_8_test)
std_5_8_test = np.nanstd(flat_5_8_test)
flat_5_8_normalized_test = (flat_5_8_test - mean_5_8_test) / std_5_8_test

mean_x_9_10_test = np.nanmean(flat_x_9_10_test)
std_x_9_10_test = np.nanstd(flat_x_9_10_test)
flat_x_9_10_normalized_test = (flat_x_9_10_test - mean_x_9_10_test) / std_x_9_10_test

mean_y_9_10_test = np.nanmean(flat_y_9_10_test)
std_y_9_10_test = np.nanstd(flat_y_9_10_test)
flat_y_9_10_normalized_test = (flat_y_9_10_test - mean_y_9_10_test) / std_y_9_10_test

mean_x_11_12_test = np.nanmean(flat_x_11_12_test)
std_x_11_12_test = np.nanstd(flat_x_11_12_test)
flat_x_11_12_normalized_test = (flat_x_11_12_test - mean_x_11_12_test) / std_x_11_12_test

mean_y_11_12_test = np.nanmean(flat_y_11_12_test)
std_y_11_12_test = np.nanstd(flat_y_11_12_test)
flat_y_11_12_normalized_test = (flat_y_11_12_test - mean_y_11_12_test) / std_y_11_12_test

down_1_4_test = downsample(flat_1_4_normalized_test, newL)
down_5_8_test = downsample(flat_5_8_normalized_test, newL)
down_x_9_10_test, down_y_9_10_test = downsample(flat_x_9_10_normalized_test, newL), downsample(flat_y_9_10_normalized_test, newL)
down_x_11_12_test, down_y_11_12_test = downsample(flat_x_11_12_normalized_test, newL), downsample(flat_y_11_12_normalized_test, newL)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_test.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_test.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_test.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_test.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_test.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_test.shape)
print('-' * 100)
print('shape of downsampled smoothpur_1_4 :', down_1_4_test.shape)
print('shape of downsampled smoothpur_5_8 :', down_5_8_test.shape)
print('shape of downsampled smoothpur_x_9_10 :', down_x_9_10_test.shape)
print('shape of downsampled smoothpur_y_9_10 :', down_y_9_10_test.shape)
print('shape of downsampled smoothpur_x_11_12 :', down_x_11_12_test.shape)
print('shape of downsampled smoothpur_y_11_12 :', down_y_11_12_test.shape)


#TRAIN

# Process flat_1_4
long_signals_1_4_train = []
idxs_1_4_train = []
for idx, i in enumerate(flat_1_4_normalized_train):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_1_4_train.append(idx)
        long_signals_1_4_train.append(i)

short_signals_1_4_train = [down_1_4_train[idx] for idx in idxs_1_4_train]

# Convert lists to numpy arrays
long_signals_1_4_train = np.array(long_signals_1_4_train)
short_signals_1_4_train = np.array(short_signals_1_4_train)

# Process flat_5_8
long_signals_5_8_train = []
idxs_5_8_train = []
for idx, i in enumerate(flat_5_8_normalized_train):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_5_8_train.append(idx)
        long_signals_5_8_train.append(i)

short_signals_5_8_train = [down_5_8_train[idx] for idx in idxs_5_8_train]

# Convert lists to numpy arrays
long_signals_5_8_train = np.array(long_signals_5_8_train)
short_signals_5_8_train = np.array(short_signals_5_8_train)

# Process flat_x_9_10
long_signals_x_9_10_train = []
idxs_x_9_10_train = []
for idx, i in enumerate(flat_x_9_10_normalized_train):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_x_9_10_train.append(idx)
        long_signals_x_9_10_train.append(i)

short_signals_x_9_10_train = [down_x_9_10_train[idx] for idx in idxs_x_9_10_train]

# Convert lists to numpy arrays
long_signals_x_9_10_train = np.array(long_signals_x_9_10_train)
short_signals_x_9_10_train = np.array(short_signals_x_9_10_train)

# Process flat_y_9_10
long_signals_y_9_10_train = []
idxs_y_9_10_train = []
for idx, i in enumerate(flat_y_9_10_normalized_train):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_y_9_10_train.append(idx)
        long_signals_y_9_10_train.append(i)

short_signals_y_9_10_train = [down_y_9_10_train[idx] for idx in idxs_y_9_10_train]

# Convert lists to numpy arrays
long_signals_y_9_10_train = np.array(long_signals_y_9_10_train)
short_signals_y_9_10_train = np.array(short_signals_y_9_10_train)

# Process flat_x_11_12
long_signals_x_11_12_train = []
idxs_x_11_12_train = []
for idx, i in enumerate(flat_x_11_12_normalized_train):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_x_11_12_train.append(idx)
        long_signals_x_11_12_train.append(i)

short_signals_x_11_12_train = [down_x_11_12_train[idx] for idx in idxs_x_11_12_train]

# Convert lists to numpy arrays
long_signals_x_11_12_train = np.array(long_signals_x_11_12_train)
short_signals_x_11_12_train = np.array(short_signals_x_11_12_train)

# Process flat_y_11_12
long_signals_y_11_12_train = []
idxs_y_11_12_train = []
for idx, i in enumerate(flat_y_11_12_normalized_train):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_y_11_12_train.append(idx)
        long_signals_y_11_12_train.append(i)

short_signals_y_11_12_train = [down_y_11_12_train[idx] for idx in idxs_y_11_12_train]

# Convert lists to numpy arrays
long_signals_y_11_12_train = np.array(long_signals_y_11_12_train)
short_signals_y_11_12_train = np.array(short_signals_y_11_12_train)

# Append all long signals and short signals
long_signals_train = np.concatenate([
    long_signals_1_4_train,
    long_signals_5_8_train,
    long_signals_x_9_10_train,
    long_signals_y_9_10_train,
    long_signals_x_11_12_train,
    long_signals_y_11_12_train
], axis=0)

short_signals_train = np.concatenate([
    short_signals_1_4_train,
    short_signals_5_8_train,
    short_signals_x_9_10_train,
    short_signals_y_9_10_train,
    short_signals_x_11_12_train,
    short_signals_y_11_12_train
], axis=0)



#TEST

# Process flat_1_4
long_signals_1_4_test = []
idxs_1_4_test = []
for idx, i in enumerate(flat_1_4_normalized_test):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_1_4_test.append(idx)
        long_signals_1_4_test.append(i)

short_signals_1_4_test = [down_1_4_test[idx] for idx in idxs_1_4_test]

# Convert lists to numpy arrays
long_signals_1_4_test = np.array(long_signals_1_4_test)
short_signals_1_4_test = np.array(short_signals_1_4_test)

# Process flat_5_8
long_signals_5_8_test = []
idxs_5_8_test = []
for idx, i in enumerate(flat_5_8_normalized_test):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_5_8_test.append(idx)
        long_signals_5_8_test.append(i)

short_signals_5_8_test = [down_5_8_test[idx] for idx in idxs_5_8_test]

# Convert lists to numpy arrays
long_signals_5_8_test = np.array(long_signals_5_8_test)
short_signals_5_8_test = np.array(short_signals_5_8_test)

# Process flat_x_9_10
long_signals_x_9_10_test = []
idxs_x_9_10_test = []
for idx, i in enumerate(flat_x_9_10_normalized_test):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_x_9_10_test.append(idx)
        long_signals_x_9_10_test.append(i)

short_signals_x_9_10_test = [down_x_9_10_test[idx] for idx in idxs_x_9_10_test]

# Convert lists to numpy arrays
long_signals_x_9_10_test = np.array(long_signals_x_9_10_test)
short_signals_x_9_10_test = np.array(short_signals_x_9_10_test)

# Process flat_y_9_10
long_signals_y_9_10_test = []
idxs_y_9_10_test = []
for idx, i in enumerate(flat_y_9_10_normalized_test):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_y_9_10_test.append(idx)
        long_signals_y_9_10_test.append(i)

short_signals_y_9_10_test = [down_y_9_10_test[idx] for idx in idxs_y_9_10_test]

# Convert lists to numpy arrays
long_signals_y_9_10_test = np.array(long_signals_y_9_10_test)
short_signals_y_9_10_test = np.array(short_signals_y_9_10_test)

# Process flat_x_11_12
long_signals_x_11_12_test = []
idxs_x_11_12_test = []
for idx, i in enumerate(flat_x_11_12_normalized_test):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_x_11_12_test.append(idx)
        long_signals_x_11_12_test.append(i)

short_signals_x_11_12_test = [down_x_11_12_test[idx] for idx in idxs_x_11_12_test]

# Convert lists to numpy arrays
long_signals_x_11_12_test = np.array(long_signals_x_11_12_test)
short_signals_x_11_12_test = np.array(short_signals_x_11_12_test)

# Process flat_y_11_12
long_signals_y_11_12_test = []
idxs_y_11_12_test = []
for idx, i in enumerate(flat_y_11_12_normalized_test):
    if not np.isnan(i).sum():  # Check if there are no NaN values
        idxs_y_11_12_test.append(idx)
        long_signals_y_11_12_test.append(i)

short_signals_y_11_12_test = [down_y_11_12_test[idx] for idx in idxs_y_11_12_test]

# Convert lists to numpy arrays
long_signals_y_11_12_test = np.array(long_signals_y_11_12_test)
short_signals_y_11_12_test = np.array(short_signals_y_11_12_test)

# Append all long signals and short signals
long_signals_test = np.concatenate([
    long_signals_1_4_test,
    long_signals_5_8_test,
    long_signals_x_9_10_test,
    long_signals_y_9_10_test,
    long_signals_x_11_12_test,
    long_signals_y_11_12_test
], axis=0)

short_signals_test = np.concatenate([
    short_signals_1_4_test,
    short_signals_5_8_test,
    short_signals_x_9_10_test,
    short_signals_y_9_10_test,
    short_signals_x_11_12_test,
    short_signals_y_11_12_test
], axis=0)


def initial_upsampling(downsampled_signals, target_length):
    upsampled_signals = []
    for signal in downsampled_signals:
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)
        interpolator = interp1d(x_old, signal, kind='cubic')  # You can use 'linear' or 'cubic'
        upsampled_signal = interpolator(x_new)
        upsampled_signals.append(upsampled_signal)
    return np.array(upsampled_signals)

# Example data generation (replace with your actual data)
downsampled_signals_train = short_signals_train.astype(np.float32)
original_signals_train = long_signals_train.astype(np.float32)

downsampled_signals_test = short_signals_test.astype(np.float32)
original_signals_test = long_signals_test.astype(np.float32)

# Initial Upsampling
initial_upsampled_signals_train = initial_upsampling(downsampled_signals_train, target_length=15000)
initial_upsampled_signals_test = initial_upsampling(downsampled_signals_test, target_length=15000)

# # Convert numpy arrays to PyTorch tensors with float32
initial_upsampled_signals_train = torch.tensor(initial_upsampled_signals_train, dtype=torch.float32)
original_signals_train = torch.tensor(original_signals_train, dtype=torch.float32)

initial_upsampled_signals_test = torch.tensor(initial_upsampled_signals_test, dtype=torch.float32)
original_signals_test = torch.tensor(original_signals_test, dtype=torch.float32)

print(f"Initial upsampled signals shape train: {initial_upsampled_signals_train.shape}")
print(f"Original signals shape train: {original_signals_train.shape}")
print(f"Initial upsampled signals shape test: {initial_upsampled_signals_test.shape}")
print(f"Original signals shape test: {original_signals_test.shape}")

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define the model
class ImprovedAutoencoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoder, self).__init__()
        self.encoder_conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.encoder_bn1 = nn.BatchNorm1d(32)
        self.encoder_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_bn2 = nn.BatchNorm1d(64)
        self.encoder_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_bn3 = nn.BatchNorm1d(128)
        self.encoder_conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.encoder_bn4 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        
        self.decoder_convtrans1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_bn1 = nn.BatchNorm1d(128)
        self.decoder_convtrans2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_bn2 = nn.BatchNorm1d(64)
        self.decoder_convtrans3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_bn3 = nn.BatchNorm1d(32)
        self.decoder_conv1 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1 = self.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x2 = self.relu(self.encoder_bn2(self.encoder_conv2(x1)))
        x3 = self.relu(self.encoder_bn3(self.encoder_conv3(x2)))
        x4 = self.relu(self.encoder_bn4(self.encoder_conv4(x3)))
        
        x = self.relu(self.decoder_bn1(self.decoder_convtrans1(x4)))
        if x3.size(2) != x.size(2):
            x3 = F.interpolate(x3, size=x.size(2))
        x = x + x3
        
        x = self.relu(self.decoder_bn2(self.decoder_convtrans2(x)))
        if x2.size(2) != x.size(2):
            x2 = F.interpolate(x2, size=x.size(2))
        x = x + x2
        
        x = self.relu(self.decoder_bn3(self.decoder_convtrans3(x)))
        if x1.size(2) != x.size(2):
            x1 = F.interpolate(x1, size=x.size(2))
        x = x + x1
        
        x = self.decoder_conv1(x)
        return x


# ==============================================================================
# ADDED TRAINING LOOP WITH 10-FOLD CROSS-VALIDATION
# ==============================================================================

# Directory to save trained model weights
save_dir_models = 'saved_RAE_models'
os.makedirs(save_dir_models, exist_ok=True)

# Prepare full training dataset for CV
train_dataset_cv = TensorDataset(initial_upsampled_signals_train, original_signals_train)

# Initialize K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(initial_upsampled_signals_train)):
    print(f'\n--- Training Fold {fold + 1} ---')
    
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader_cv = DataLoader(train_dataset_cv, batch_size=32, sampler=train_subsampler)
    val_loader_cv = DataLoader(train_dataset_cv, batch_size=32, sampler=val_subsampler)
    
    # Initialize a new model for each fold
    model_cv = ImprovedAutoencoder().to(device)
    criterion_cv = nn.MSELoss()
    optimizer_cv = optim.Adam(model_cv.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    patience = 20  # Early stopping patience
    trigger_times = 0
    
    for epoch in range(1, 301):
        model_cv.train()
        train_loss = 0.0
        for upsampled, original in train_loader_cv:
            upsampled = upsampled.unsqueeze(1).to(device)
            original = original.unsqueeze(1).to(device)
            
            optimizer_cv.zero_grad()
            outputs = model_cv(upsampled)
            loss = criterion_cv(outputs, original)
            loss.backward()
            optimizer_cv.step()
            
            train_loss += loss.item() * upsampled.size(0)
            
        train_loss /= len(train_idx)
        
        # Validation Loop
        model_cv.eval()
        val_loss = 0.0
        with torch.no_grad():
            for upsampled, original in val_loader_cv:
                upsampled = upsampled.unsqueeze(1).to(device)
                original = original.unsqueeze(1).to(device)
                outputs = model_cv(upsampled)
                loss = criterion_cv(outputs, original)
                val_loss += loss.item() * upsampled.size(0)
                
        val_loss /= len(val_idx)
        
        print(f"Epoch [{epoch}/300], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early Stopping and Saving Model Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss decreased ({val_loss:.6f}). Saving model...")
            torch.save(model_cv.state_dict(), os.path.join(save_dir_models, f'best_model_fold_{fold + 1}.pth'))
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

# ==============================================================================
# END ADDED TRAINING LOOP 
# ==============================================================================


print("\n--- Starting Evaluation on Test Set ---")

# Load all 10 pre-trained models from the newly created saved_models folder
models = []
for fold in range(1, 11):
    model = ImprovedAutoencoder()
    model.load_state_dict(torch.load(os.path.join(save_dir_models, f'best_model_fold_{fold}.pth')))
    model.to(device)
    models.append(model)

# Prepare test dataset
test_dataset = TensorDataset(initial_upsampled_signals_test, original_signals_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Compute test loss for each model
criterion = nn.MSELoss()
test_losses = []
for idx, model in enumerate(models, 1):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for upsampled, original in test_loader:
            upsampled = upsampled.unsqueeze(1).to(device)
            original = original.unsqueeze(1).to(device)
            outputs = model(upsampled)
            loss = criterion(outputs, original)
            test_loss += loss.item() * upsampled.size(0)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'Model fold {idx} Test Loss: {test_loss:.6f}')

avg_test_loss = np.mean(test_losses)
print(f'Average Test Loss across 10 folds: {avg_test_loss:.6f}')

# Use the last model (fold 10) for plotting
model = models[-1]

# Create directories for plots
signal_plot_dir = 'signal_plots'
spectrum_plot_dir = 'spectrum_plots'
os.makedirs(signal_plot_dir, exist_ok=True)
os.makedirs(spectrum_plot_dir, exist_ok=True)

# Define function to plot spectra in subplots
def plot_spectra_subplots(original, upsampled, refinement, sample_idx, save_dir):
    sample_rate = 1000  # Adjust if necessary
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    for ax, signal, title in zip(axes, [original, upsampled, refinement], ["Original", "Upsampled", "Refined"]):
        fft_size = 1024
        num_rows = len(signal) // fft_size
        spectrogram = np.zeros((num_rows, fft_size))
        
        for i in range(num_rows):
            segment = signal[i*fft_size:(i+1)*fft_size]
            fft_vals = np.fft.fft(segment)
            fft_vals_shifted = np.fft.fftshift(fft_vals)
            power_spectrum = 10 * np.log10(np.abs(fft_vals_shifted) ** 2 + 1e-12)
            spectrogram[i, :] = power_spectrum
        
        im = ax.imshow(spectrogram, aspect='auto', 
                       extent=[-sample_rate/2/1e6, sample_rate/2/1e6, num_rows*fft_size/sample_rate, 0],
                       cmap='viridis')
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Time [s]")
        ax.set_title(f"{title} Signal Spectrum")
        fig.colorbar(im, ax=ax, label="Power (dB)")
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"spectra_sample_{sample_idx}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot signals and spectra for all test samples
sample_idx = 0
for batch_idx, (upsampled, original) in enumerate(test_loader):
    upsampled = upsampled.unsqueeze(1).to(device)
    original = original.unsqueeze(1).to(device)
    outputs = model(upsampled)
    
    upsampled_np = upsampled.cpu().detach().numpy().squeeze()
    original_np = original.cpu().detach().numpy().squeeze()
    outputs_np = outputs.cpu().detach().numpy().squeeze()
    
    batch_size = upsampled_np.shape[0]
    for i in range(batch_size):
        # Plot signals
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(original_np[i][4100:4200], label="Original Signal", alpha=0.7)
        ax.plot(upsampled_np[i][4100:4200], label="Upsampled Signal", alpha=0.7)
        ax.plot(outputs_np[i][4100:4200], label="Refined Signal", alpha=0.7)
        ax.set_title(f"Test Sample {sample_idx}")
        ax.legend()
        ax.grid(True)
        signal_filename = os.path.join(signal_plot_dir, f"sample_{sample_idx}.png")
        plt.savefig(signal_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot spectra
        plot_spectra_subplots(original_np[i], upsampled_np[i], outputs_np[i], sample_idx, spectrum_plot_dir)
        
        sample_idx += 1

print(f"Signal plots saved in '{signal_plot_dir}'")
print(f"Spectrum plots saved in '{spectrum_plot_dir}'")