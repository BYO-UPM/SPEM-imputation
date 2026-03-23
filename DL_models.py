import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, seq_missing, block_missing
from pypots.imputation import SAITS, BRITS, CSDI
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from sklearn.impute import KNNImputer
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib.widgets import Slider
import ipywidgets as widgets
from ipywidgets import interact
from scipy.fft import fft
from scipy.ndimage import gaussian_filter
from pypots.optim import Adam
from sklearn.model_selection import KFold

# ==========================================
#  DEBUGGING FLAGS
# ==========================================
RUN_SAITS = False   # Set to False to skip SAITS
RUN_BRITS = True   # Set to False to skip BRITS
RUN_CSDI  = False   # Set to False to skip CSDI
# ==========================================

# Load JSON dictionaries for training and testing
instances_dict_path_train = './instances_dict_train.json'
with open(instances_dict_path_train, 'r') as f:
    instances_dict_train = json.load(f)
    
instances_dict_path_test = './instances_dict_test.json'
with open(instances_dict_path_test, 'r') as f:
    instances_dict_test = json.load(f)

# Load numpy arrays (processed and original)
smoothpur_1_4_train = np.load('Data_in/Train/SmoothPur_1_4_Ar.npy')
smoothpur_5_8_train = np.load('Data_in/Train/SmoothPur_5_8_Ar.npy')
smoothpur_9_10_train = np.load('Data_in/Train/SmoothPur_9_10_Ar.npy')
smoothpur_11_12_train = np.load('Data_in/Train/SmoothPur_11_12_Ar.npy')

smoothpur_1_4_test = np.load('Data_in/Test/SmoothPur_1_4_Ar.npy')
smoothpur_5_8_test = np.load('Data_in/Test/SmoothPur_5_8_Ar.npy')
smoothpur_9_10_test = np.load('Data_in/Test/SmoothPur_9_10_Ar.npy')
smoothpur_11_12_test = np.load('Data_in/Test/SmoothPur_11_12_Ar.npy')

smoothpur_1_4_train_ori = np.load('Data_in/Train/SmoothPur_1_4.npy')
smoothpur_5_8_train_ori = np.load('Data_in/Train/SmoothPur_5_8.npy')
smoothpur_9_10_train_ori = np.load('Data_in/Train/SmoothPur_9_10.npy')
smoothpur_11_12_train_ori = np.load('Data_in/Train/SmoothPur_11_12.npy')

smoothpur_1_4_test_ori = np.load('Data_in/Test/SmoothPur_1_4.npy')
smoothpur_5_8_test_ori = np.load('Data_in/Test/SmoothPur_5_8.npy')
smoothpur_9_10_test_ori = np.load('Data_in/Test/SmoothPur_9_10.npy')
smoothpur_11_12_test_ori = np.load('Data_in/Test/SmoothPur_11_12.npy')

# Print information
print('TRAIN')
print(instances_dict_train)
print('length of instances_dict : ', len(instances_dict_train))
print('-' * 100)
print('length of smoothpur_1_4 : ', len(smoothpur_1_4_train))
print('length of smoothpur_5_8 : ', len(smoothpur_5_8_train))
print('length of smoothpur_9_10 : ', len(smoothpur_9_10_train))
print('length of smoothpur_11_12 : ', len(smoothpur_11_12_train))
print('-' * 100)
print('shape of smoothpur_1_4 : ', smoothpur_1_4_train.shape)
print('shape of smoothpur_5_8 : ', smoothpur_5_8_train.shape)

print('TEST')
print(instances_dict_test)
print('length of instances_dict : ', len(instances_dict_test))
print('-' * 100)
print('shape of smoothpur_1_4 : ', smoothpur_1_4_test.shape)

# Function definitions
def downsample(signals, new_len):
    downsampled_signals = []
    for signal in signals:
        signal = signal.reshape(-1)  # Flatten the signal
        downsample_factor = len(signal) // new_len
        indices = np.arange(0, len(signal), downsample_factor)
        downsampled_signal = signal[indices[:new_len]]  # Ensure the length matches new_len
        downsampled_signals.append(downsampled_signal)
    return np.array(downsampled_signals)

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

def compute_all_metrics(imputed_signal, original_signal, indicating_mask):
    num_signals, signal_length = original_signal.shape
    high_freq = 30
    low_freq = 0.1
    sampling_rate = 1000
    MAE_list = []
    MRE_list = []
    RMSE_list = []
    Sim_list = []
    FSD_list = []
    RMSE_F_list = []
    RMSE_F_low_list = []
    RMSE_F_high_list = []
    freq_bins = np.fft.rfftfreq(signal_length, d=1/sampling_rate)
    low_freq_indices = np.where(freq_bins <= low_freq)[0]
    high_freq_indices = np.where(freq_bins >= high_freq)[0]
    for i in range(num_signals):
        orig_sig = np.ravel(original_signal[i, :])
        imp_sig = np.ravel(imputed_signal[i, :])
        mask = np.ravel(indicating_mask[i, :])
        if not np.any(mask):
            continue
        orig_values_at_imputed = orig_sig[mask]
        imp_values_at_imputed = imp_sig[mask]
        mae = np.mean(np.abs(orig_values_at_imputed - imp_values_at_imputed))
        MAE_list.append(mae)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs((orig_values_at_imputed - imp_values_at_imputed) / orig_values_at_imputed)
            relative_errors = np.nan_to_num(relative_errors, nan=0.0, posinf=0.0, neginf=0.0)
        mre = np.mean(relative_errors)
        MRE_list.append(mre)
        mse = np.mean((orig_values_at_imputed - imp_values_at_imputed) ** 2)
        rmse = np.sqrt(mse)
        RMSE_list.append(rmse)
        orig_mean = np.mean(orig_values_at_imputed)
        imp_mean = np.mean(imp_values_at_imputed)
        numerator = np.sum((orig_values_at_imputed - orig_mean) * (imp_values_at_imputed - imp_mean))
        denominator = np.sqrt(np.sum((orig_values_at_imputed - orig_mean) ** 2) * np.sum((imp_values_at_imputed - imp_mean) ** 2))
        sim = numerator / denominator if denominator != 0 else 0
        Sim_list.append(sim)
        std_orig = np.std(orig_values_at_imputed)
        fsd = rmse / std_orig if std_orig != 0 else 0
        FSD_list.append(fsd)
        orig_sig_fft = fft(orig_sig)
        imp_sig_fft = fft(imp_sig)
        mse_f = np.mean(np.abs(orig_sig_fft - imp_sig_fft) ** 2)
        rmse_f = np.sqrt(mse_f)
        RMSE_F_list.append(rmse_f)
        orig_sig_fft_low = orig_sig_fft[low_freq_indices]
        imp_sig_fft_low = imp_sig_fft[low_freq_indices]
        mse_f_low = np.mean(np.abs(orig_sig_fft_low - imp_sig_fft_low) ** 2)
        rmse_f_low = np.sqrt(mse_f_low)
        RMSE_F_low_list.append(rmse_f_low)
        orig_sig_fft_high = orig_sig_fft[high_freq_indices]
        imp_sig_fft_high = imp_sig_fft[high_freq_indices]
        mse_f_high = np.mean(np.abs(orig_sig_fft_high - imp_sig_fft_high) ** 2)
        rmse_f_high = np.sqrt(mse_f_high)
        RMSE_F_high_list.append(rmse_f_high)
    metrics = {
        'MAE_mean': np.mean(MAE_list) if MAE_list else None,
        'MRE_mean': np.mean(MRE_list) if MRE_list else None,
        'RMSE_mean': np.mean(RMSE_list) if RMSE_list else None,
        'Sim_mean': np.mean(Sim_list) if Sim_list else None,
        'FSD_mean': np.mean(FSD_list) if FSD_list else None,
        'RMSE_F_mean': np.mean(RMSE_F_list) if RMSE_F_list else None,
        'RMSE_F_Low_mean': np.mean(RMSE_F_low_list) if RMSE_F_low_list else None,
        'RMSE_F_High_mean': np.mean(RMSE_F_high_list) if RMSE_F_high_list else None
    }
    return metrics

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


print("TRAIN (Original)")
flat_1_4_train_ori = flatten_smoothpur_1_8(smoothpur_1_4_train_ori)
flat_5_8_train_ori = flatten_smoothpur_1_8(smoothpur_5_8_train_ori)
flat_x_9_10_train_ori, flat_y_9_10_train_ori = flatten_smoothpur_9_12(smoothpur_9_10_train_ori)
flat_x_11_12_train_ori, flat_y_11_12_train_ori = flatten_smoothpur_9_12(smoothpur_11_12_train_ori)

mean_1_4_train_ori = np.nanmean(flat_1_4_train_ori)
std_1_4_train_ori = np.nanstd(flat_1_4_train_ori)
flat_1_4_normalized_train_ori = (flat_1_4_train_ori - mean_1_4_train_ori) / std_1_4_train_ori

mean_5_8_train_ori = np.nanmean(flat_5_8_train_ori)
std_5_8_train_ori = np.nanstd(flat_5_8_train_ori)
flat_5_8_normalized_train_ori = (flat_5_8_train_ori - mean_5_8_train_ori) / std_5_8_train_ori

mean_x_9_10_train_ori = np.nanmean(flat_x_9_10_train_ori)
std_x_9_10_train_ori = np.nanstd(flat_x_9_10_train_ori)
flat_x_9_10_normalized_train_ori = (flat_x_9_10_train_ori - mean_x_9_10_train_ori) / std_x_9_10_train_ori

mean_y_9_10_train_ori = np.nanmean(flat_y_9_10_train_ori)
std_y_9_10_train_ori = np.nanstd(flat_y_9_10_train_ori)
flat_y_9_10_normalized_train_ori = (flat_y_9_10_train_ori - mean_y_9_10_train_ori) / std_y_9_10_train_ori

mean_x_11_12_train_ori = np.nanmean(flat_x_11_12_train_ori)
std_x_11_12_train_ori = np.nanstd(flat_x_11_12_train_ori)
flat_x_11_12_normalized_train_ori = (flat_x_11_12_train_ori - mean_x_11_12_train_ori) / std_x_11_12_train_ori

mean_y_11_12_train_ori = np.nanmean(flat_y_11_12_train_ori)
std_y_11_12_train_ori = np.nanstd(flat_y_11_12_train_ori)
flat_y_11_12_normalized_train_ori = (flat_y_11_12_train_ori - mean_y_11_12_train_ori) / std_y_11_12_train_ori

print("TEST (Original)")
flat_1_4_test_ori = flatten_smoothpur_1_8(smoothpur_1_4_test_ori)
flat_5_8_test_ori = flatten_smoothpur_1_8(smoothpur_5_8_test_ori)
flat_x_9_10_test_ori, flat_y_9_10_test_ori = flatten_smoothpur_9_12(smoothpur_9_10_test_ori)
flat_x_11_12_test_ori, flat_y_11_12_test_ori = flatten_smoothpur_9_12(smoothpur_11_12_test_ori)

mean_1_4_test_ori = np.nanmean(flat_1_4_test_ori)
std_1_4_test_ori = np.nanstd(flat_1_4_test_ori)
flat_1_4_normalized_test_ori = (flat_1_4_test_ori - mean_1_4_test_ori) / std_1_4_test_ori

mean_5_8_test_ori = np.nanmean(flat_5_8_test_ori)
std_5_8_test_ori = np.nanstd(flat_5_8_test_ori)
flat_5_8_normalized_test_ori = (flat_5_8_test_ori - mean_5_8_test_ori) / std_5_8_test_ori

mean_x_9_10_test_ori = np.nanmean(flat_x_9_10_test_ori)
std_x_9_10_test_ori = np.nanstd(flat_x_9_10_test_ori)
flat_x_9_10_normalized_test_ori = (flat_x_9_10_test_ori - mean_x_9_10_test_ori) / std_x_9_10_test_ori

mean_y_9_10_test_ori = np.nanmean(flat_y_9_10_test_ori)
std_y_9_10_test_ori = np.nanstd(flat_y_9_10_test_ori)
flat_y_9_10_normalized_test_ori = (flat_y_9_10_test_ori - mean_y_9_10_test_ori) / std_y_9_10_test_ori

mean_x_11_12_test_ori = np.nanmean(flat_x_11_12_test_ori)
std_x_11_12_test_ori = np.nanstd(flat_x_11_12_test_ori)
flat_x_11_12_normalized_test_ori = (flat_x_11_12_test_ori - mean_x_11_12_test_ori) / std_x_11_12_test_ori

mean_y_11_12_test_ori = np.nanmean(flat_y_11_12_test_ori)
std_y_11_12_test_ori = np.nanstd(flat_y_11_12_test_ori)
flat_y_11_12_normalized_test_ori = (flat_y_11_12_test_ori - mean_y_11_12_test_ori) / std_y_11_12_test_ori

flats_normalized_train = np.concatenate([
    flat_1_4_normalized_train, flat_5_8_normalized_train,
    flat_x_9_10_normalized_train, flat_y_9_10_normalized_train,
    flat_x_11_12_normalized_train, flat_y_11_12_normalized_train
], axis=0)

flats_normalized_test = np.concatenate([
    flat_1_4_normalized_test, flat_5_8_normalized_test,
    flat_x_9_10_normalized_test, flat_y_9_10_normalized_test,
    flat_x_11_12_normalized_test, flat_y_11_12_normalized_test
], axis=0)

flats_normalized_train_ori = np.concatenate([
    flat_1_4_normalized_train_ori, flat_5_8_normalized_train_ori,
    flat_x_9_10_normalized_train_ori, flat_y_9_10_normalized_train_ori,
    flat_x_11_12_normalized_train_ori, flat_y_11_12_normalized_train_ori
], axis=0)

flats_normalized_test_ori = np.concatenate([
    flat_1_4_normalized_test_ori, flat_5_8_normalized_test_ori,
    flat_x_9_10_normalized_test_ori, flat_y_9_10_normalized_test_ori,
    flat_x_11_12_normalized_test_ori, flat_y_11_12_normalized_test_ori
], axis=0)

print("TRAIN - Downsampling")
newL = 500
downs_train = downsample(flats_normalized_train, newL)
downs_train_ori = downsample(flats_normalized_train_ori, newL)

print("TEST - Downsampling")
downs_test = downsample(flats_normalized_test, newL)
downs_test_ori = downsample(flats_normalized_test_ori, newL)

print("TRAIN - Expanding Dimensions")
exps_train = np.expand_dims(downs_train, axis=-1)
exps_train_ori = np.expand_dims(downs_train_ori, axis=-1)

print("TEST - Expanding Dimensions")
exps_test = np.expand_dims(downs_test, axis=-1)
exps_test_ori = np.expand_dims(downs_test_ori, axis=-1)

custom_optimizer = Adam(lr=0.0004)

np.random.seed(42)
torch.manual_seed(42)

n_splits = 10

# ==============================================================================
# 1. SAITS EVALUATION
# ==============================================================================
if RUN_SAITS:
    print("\n" + "="*50)
    print("Evaluating SAITS (10-Fold CV Training & Testing)")
    print("="*50)
    
    os.makedirs('saits_weights', exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # ------------------ TRAINING ------------------
    fold = 1
    for train_index, val_index in kf.split(exps_train):
        print(f"\n--- Training SAITS Fold {fold}/{n_splits} ---")
        X_train, X_val = exps_train[train_index], exps_train[val_index]
        X_ori_train, X_ori_val = exps_train_ori[train_index], exps_train_ori[val_index]

        dataset_train = {"X": X_train, "X_ori": X_ori_train}
        dataset_val = {"X": X_val, "X_ori": X_ori_val}

        
        saits = SAITS(n_steps=newL, n_features=1, n_layers=2, d_model=256, n_heads=4, 
                      d_k=64, d_v=64, d_ffn=128, dropout=0.2, epochs=500, 
                      optimizer=custom_optimizer, model_saving_strategy='better')

        # Train model within the fold
        saits.fit(train_set=dataset_train, val_set=dataset_val)
        
        # Save model weights
        saits.save(f'saits_weights/saits_all_artificial_4_kfolds_{fold}.pypots', overwrite=True)            
        
        # Validate to show progress
        imputation = saits.impute(dataset_val)
        indicating_mask = np.isnan(X_val) & (~np.isnan(X_ori_val))
        fold_metrics = compute_all_metrics(imputation.squeeze(), X_ori_val.squeeze(), indicating_mask.squeeze())
        print(f"Fold {fold} Val metrics: {fold_metrics}")
        
        fold += 1

    # ------------------ TESTING ------------------
    print("\n--- Testing SAITS on Held-Out Set ---")
    X_test = exps_test       
    X_test_ori = exps_test_ori
    test_dataset = {"X": X_test}
    metrics_list = []

    for fold in range(1, n_splits + 1):
        print(f"Processing Test for fold {fold}...")  
        model_path = f"saits_weights/saits_all_artificial_4_kfolds_{fold}.pypots"
        
        saits = SAITS(n_steps=newL, n_features=1, n_layers=2, d_model=256, n_heads=4, 
                      d_k=64, d_v=64, d_ffn=128, dropout=0.2, epochs=500, 
                      optimizer=custom_optimizer, model_saving_strategy='better')
        saits.load(model_path)
        
        imputation_test = saits.impute(test_dataset)
        indicating_mask_test = np.isnan(X_test) & (~np.isnan(X_test_ori))
        
        fold_metrics = compute_all_metrics(imputation_test.squeeze(), X_test_ori.squeeze(), indicating_mask_test.squeeze())
        metrics_list.append(fold_metrics)
        print(f"Fold {fold} Test metrics: {fold_metrics}")

    averaged_metrics = {
        key: np.mean([m[key] for m in metrics_list if m[key] is not None]) if any(m[key] is not None for m in metrics_list) else None
        for key in metrics_list[0].keys()}

    print("\nSAITS - FINAL K-Fold Cross Validation Results (Test Set)")
    for key, value in averaged_metrics.items():
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: None")


# ==============================================================================
# 2. BRITS EVALUATION
# ==============================================================================
if RUN_BRITS:
    print("\n" + "="*50)
    print("Evaluating BRITS (10-Fold CV Training & Testing)")
    print("="*50)
    
    os.makedirs('brits_weights', exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # ------------------ TRAINING ------------------
    fold = 1
    for train_index, val_index in kf.split(exps_train):
        print(f"\n--- Training BRITS Fold {fold}/{n_splits} ---")
        X_train, X_val = exps_train[train_index], exps_train[val_index]
        X_ori_train, X_ori_val = exps_train_ori[train_index], exps_train_ori[val_index]

        dataset_train = {"X": X_train, "X_ori": X_ori_train}
        dataset_val = {"X": X_val, "X_ori": X_ori_val}

        brits = BRITS(n_steps=newL, n_features=1, rnn_hidden_size=64, batch_size=128, epochs=500, 
                      optimizer=custom_optimizer, model_saving_strategy='better')

        brits.fit(train_set=dataset_train, val_set=dataset_val)
        
        brits.save(f'brits_weights/brits_all_artificial_4_kfolds_{fold}.pypots', overwrite=True)            
        
        imputation = brits.impute(dataset_val)
        indicating_mask = np.isnan(X_val) & (~np.isnan(X_ori_val))
        fold_metrics = compute_all_metrics(imputation.squeeze(), X_ori_val.squeeze(), indicating_mask.squeeze())
        print(f"Fold {fold} Val metrics: {fold_metrics}")
        
        fold += 1

    # ------------------ TESTING ------------------
    print("\n--- Testing BRITS on Held-Out Set ---")
    X_test = exps_test       
    X_test_ori = exps_test_ori
    test_dataset = {"X": X_test}
    metrics_list = []

    for fold in range(1, n_splits + 1):
        print(f"Processing Test for fold {fold}...")  
        model_path = f"brits_weights/brits_all_artificial_4_kfolds_{fold}.pypots"
        
        brits = BRITS(n_steps=newL, n_features=1, rnn_hidden_size=64, batch_size=128, epochs=500, 
                      optimizer=custom_optimizer, model_saving_strategy='better')
        brits.load(model_path)
        
        imputation_test = brits.impute(test_dataset)
        indicating_mask_test = np.isnan(X_test) & (~np.isnan(X_test_ori))
        
        fold_metrics = compute_all_metrics(imputation_test.squeeze(), X_test_ori.squeeze(), indicating_mask_test.squeeze())
        metrics_list.append(fold_metrics)
        print(f"Fold {fold} Test metrics: {fold_metrics}")

    averaged_metrics = {
        key: np.mean([m[key] for m in metrics_list if m[key] is not None]) if any(m[key] is not None for m in metrics_list) else None
        for key in metrics_list[0].keys()}

    print("\nBRITS - FINAL K-Fold Cross Validation Results (Test Set)")
    for key, value in averaged_metrics.items():
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: None")


# ==============================================================================
# 3. CSDI EVALUATION
# ==============================================================================
if RUN_CSDI:
    print("\n" + "="*50)
    print("Evaluating CSDI (10-Fold CV Training & Testing)")
    print("="*50)
    
    os.makedirs('csdi_weights', exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # ------------------ TRAINING ------------------
    fold = 1
    for train_index, val_index in kf.split(exps_train):
        print(f"\n--- Training CSDI Fold {fold}/{n_splits} ---")
        X_train, X_val = exps_train[train_index], exps_train[val_index]
        X_ori_train, X_ori_val = exps_train_ori[train_index], exps_train_ori[val_index]

        dataset_train = {"X": X_train, "X_ori": X_ori_train}
        dataset_val = {"X": X_val, "X_ori": X_ori_val}

        csdi = CSDI(
        n_steps=newL, 
        n_features=1, 
        n_layers=2, 
        n_heads=4, 
        n_channels=32, 
        d_time_embedding=64,      # <--- ADDED
        d_feature_embedding=3,   # <--- ADDED
        d_diffusion_embedding=128, # <--- ADDED
        n_diffusion_steps=20, 
        epochs=500, 
        optimizer=Adam(lr=0.0004),
        model_saving_strategy='better'
    )

        # csdi.fit(train_set=dataset_train, val_set=dataset_val)
        
        # csdi.save(f'csdi_weights/csdi_all_artificial_4_kfolds_{fold}.pypots', overwrite=True)
        model_path = f"csdi_weights/csdi_all_artificial_4_kfolds_{fold}.pypots"
        csdi.load(model_path)            
        
        imputation = csdi.impute(dataset_val)
        indicating_mask = np.isnan(X_val) & (~np.isnan(X_ori_val))
        fold_metrics = compute_all_metrics(imputation.squeeze(), X_ori_val.squeeze(), indicating_mask.squeeze())
        print(f"Fold {fold} Val metrics: {fold_metrics}")
        
        fold += 1

    # ------------------ TESTING ------------------
    print("\n--- Testing CSDI on Held-Out Set ---")
    X_test = exps_test       
    X_test_ori = exps_test_ori
    test_dataset = {"X": X_test}
    metrics_list = []

    for fold in range(1, n_splits + 1):
        print(f"Processing Test for fold {fold}...")  
        model_path = f"csdi_weights/csdi_all_artificial_4_kfolds_{fold}.pypots"
        
        csdi = CSDI(
        n_steps=newL, 
        n_features=1, 
        n_layers=2, 
        n_heads=4, 
        n_channels=32, 
        d_time_embedding=64,      # <--- ADDED
        d_feature_embedding=3,   # <--- ADDED
        d_diffusion_embedding=128, # <--- ADDED
        n_diffusion_steps=20, 
        epochs=500, 
        optimizer=Adam(lr=0.0004),
        model_saving_strategy='better')

        csdi.load(model_path)
        
        imputation_test = csdi.impute(test_dataset)
        indicating_mask_test = np.isnan(X_test) & (~np.isnan(X_test_ori))
        
        fold_metrics = compute_all_metrics(imputation_test.squeeze(), X_test_ori.squeeze(), indicating_mask_test.squeeze())
        metrics_list.append(fold_metrics)
        print(f"Fold {fold} Test metrics: {fold_metrics}")

    averaged_metrics = {
        key: np.mean([m[key] for m in metrics_list if m[key] is not None]) if any(m[key] is not None for m in metrics_list) else None
        for key in metrics_list[0].keys()}

    print("\nCSDI - FINAL K-Fold Cross Validation Results (Test Set)")
    for key, value in averaged_metrics.items():
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: None")