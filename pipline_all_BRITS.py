import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, seq_missing, block_missing
from pypots.imputation import BRITS

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

from sklearn.model_selection import KFold
from pypots.optim import Adam

instances_dict_path_train = './instances_dict_train.json'
with open(instances_dict_path_train, 'r') as f:
    instances_dict_train = json.load(f)
    
instances_dict_path_test = './instances_dict_test.json'
with open(instances_dict_path_test, 'r') as f:
    instances_dict_test = json.load(f)

smoothpur_1_4_train_Ar = np.load('Data_in/Train/SmoothPur_1_4_Ar.npy')
smoothpur_5_8_train_Ar = np.load('Data_in/Train/SmoothPur_5_8_Ar.npy')
smoothpur_9_10_train_Ar = np.load('Data_in/Train/SmoothPur_9_10_Ar.npy')
smoothpur_11_12_train_Ar = np.load('Data_in/Train/SmoothPur_11_12_Ar.npy')

smoothpur_1_4_test_Ar = np.load('Data_in/Test/SmoothPur_1_4_Ar.npy')
smoothpur_5_8_test_Ar = np.load('Data_in/Test/SmoothPur_5_8_Ar.npy')
smoothpur_9_10_test_Ar = np.load('Data_in/Test/SmoothPur_9_10_Ar.npy')
smoothpur_11_12_test_Ar = np.load('Data_in/Test/SmoothPur_11_12_Ar.npy')

smoothpur_1_4_train_ori = np.load('Data_in/Train/SmoothPur_1_4.npy')
smoothpur_5_8_train_ori = np.load('Data_in/Train/SmoothPur_5_8.npy')
smoothpur_9_10_train_ori = np.load('Data_in/Train/SmoothPur_9_10.npy')
smoothpur_11_12_train_ori = np.load('Data_in/Train/SmoothPur_11_12.npy')

smoothpur_1_4_test_ori = np.load('Data_in/Test/SmoothPur_1_4.npy')
smoothpur_5_8_test_ori = np.load('Data_in/Test/SmoothPur_5_8.npy')
smoothpur_9_10_test_ori = np.load('Data_in/Test/SmoothPur_9_10.npy')
smoothpur_11_12_test_ori = np.load('Data_in/Test/SmoothPur_11_12.npy')

print('TRAIN')
print(instances_dict_train)
print('length of instances_dict : ' ,len(instances_dict_train))
print('-' * 100)
print('length of smoothpur_1_4 : ' ,len(smoothpur_1_4_train_ori))
print('length of smoothpur_5_8 : ' ,len(smoothpur_5_8_train_ori))
print('length of smoothpur_9_10 : ' ,len(smoothpur_9_10_train_ori))
print('length of smoothpur_11_12 : ' ,len(smoothpur_11_12_train_ori))
print('-' * 100)
print('type of smoothpur_1_4 : ' , type(smoothpur_1_4_train_ori))
print('type of smoothpur_5_8 : ' , type(smoothpur_5_8_train_ori))
print('type of smoothpur_9_10 : ' , type(smoothpur_9_10_train_ori))
print('type of smoothpur_11_12 : ' , type(smoothpur_11_12_train_ori))
print('-' * 100)
print('shape of smoothpur_1_4 : ' , smoothpur_1_4_train_ori.shape)
print('shape of smoothpur_5_8 : ' , smoothpur_5_8_train_ori.shape)
print('shape of smoothpur_9_10 : ' , smoothpur_9_10_train_ori.shape)
print('shape of smoothpur_11_12 : ' , smoothpur_11_12_train_ori.shape)

print('TEST')
print(instances_dict_test)
print('length of instances_dict : ' ,len(instances_dict_test))
print('-' * 100)
print('length of smoothpur_1_4 : ' ,len(smoothpur_1_4_test_ori))
print('length of smoothpur_5_8 : ' ,len(smoothpur_5_8_test_ori))
print('length of smoothpur_9_10 : ' ,len(smoothpur_9_10_test_ori))
print('length of smoothpur_11_12 : ' ,len(smoothpur_11_12_test_ori))
print('-' * 100)
print('type of smoothpur_1_4 : ' , type(smoothpur_1_4_test_ori))
print('type of smoothpur_5_8 : ' , type(smoothpur_5_8_test_ori))
print('type of smoothpur_9_10 : ' , type(smoothpur_9_10_test_ori))
print('type of smoothpur_11_12 : ' , type(smoothpur_11_12_test_ori))
print('-' * 100)
print('shape of smoothpur_1_4 : ' , smoothpur_1_4_test_ori.shape)
print('shape of smoothpur_5_8 : ' , smoothpur_5_8_test_ori.shape)
print('shape of smoothpur_9_10 : ' , smoothpur_9_10_test_ori.shape)
print('shape of smoothpur_11_12 : ' , smoothpur_11_12_test_ori.shape)

def flatten_smoothpur_1_8(original_arr):
    arr_shape = original_arr.shape
    flattened_list = []
    for i in range(arr_shape[0]):
      for j in range(arr_shape[1]):
        flattened_list.append(original_arr[i][j][0])
        flattened_list.append(original_arr[i][j][1])
    flattened_arr = np.array(flattened_list)
    return flattened_arr

def reform_smoothpur_1_8(flattened_arr, original_arr):
    arr_shape = original_arr.shape
    reshaped_arr = flattened_arr.reshape((154, 4, 2, 15000))
    target_values = np.empty((154, 4, 1, 15000))
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            target = original_arr[i][j][2]
            target_values[i][j][0] = target
    reformed_arr = np.concatenate((reshaped_arr, target_values), axis=2)
    return reformed_arr

def flatten_Target_1_8(original_arr):
    arr_shape = original_arr.shape
    flattened_list = []
    for i in range(arr_shape[0]):
      for j in range(arr_shape[1]):
        flattened_list.append(original_arr[i][j][2])
        flattened_list.append(original_arr[i][j][2])
    flattened_arr = np.array(flattened_list)
    return flattened_arr

def flatten_Target_9_12(original_arr):
    arr_shape = original_arr.shape
    flattened_x_list = []
    flattened_y_list = []
    for i in range(arr_shape[0]):
      for j in range(arr_shape[1]):
        flattened_x_list.append(original_arr[i][j][2][0])
        flattened_x_list.append(original_arr[i][j][2][0])
        flattened_y_list.append(original_arr[i][j][2][1])
        flattened_y_list.append(original_arr[i][j][2][1])
    flattened_x_arr = np.array(flattened_x_list)
    flattened_y_arr = np.array(flattened_y_list)
    return flattened_x_arr, flattened_y_arr

def reform_smoothpur_1_8_500(flattened_arr):
    reshaped_arr = flattened_arr.reshape((154, 4, 2, 500))
    return reshaped_arr

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

def reform_smoothpur_9_12(flattened_x_arr,flattened_y_arr, original_arr):
    arr_shape = original_arr.shape
    reshaped_x_arr = flattened_x_arr.reshape((154, 2, 1, 2, 15000))
    reshaped_y_arr = flattened_y_arr.reshape((154, 2, 1, 2, 15000))
    reformed_signal_arr = np.concatenate((reshaped_x_arr, reshaped_y_arr), axis=2)
    target_x_values = np.empty((154, 2, 1, 1, 15000))
    target_y_values = np.empty((154, 2, 1, 1, 15000))
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            targetx = original_arr[i][j][2][0]
            targety = original_arr[i][j][2][1]
            target_x_values[i][j][0][0] = targetx
            target_y_values[i][j][0][0] = targety
    reformed_target_arr = np.concatenate((target_x_values, target_y_values), axis=3)
    total_reformed_arr = np.concatenate((reformed_signal_arr, reformed_target_arr), axis=2)
    return total_reformed_arr

def reform_smoothpur_9_12_500(flattened_x_arr,flattened_y_arr):
    reshaped_x_arr = flattened_x_arr.reshape((154, 2, 1, 2, 500))
    reshaped_y_arr = flattened_y_arr.reshape((154, 2, 1, 2, 500))
    reformed_signal_arr = np.concatenate((reshaped_x_arr, reshaped_y_arr), axis=2)
    return reformed_signal_arr

def downsample(signals, new_len):
    downsampled_signals = []
    for signal in signals:
        signal = signal.reshape(-1)  # Flatten the signal
        downsample_factor = len(signal) // new_len
        indices = np.arange(0, len(signal), downsample_factor)
        downsampled_signal = signal[indices[:new_len]]  # Ensure the length matches new_len
        downsampled_signals.append(downsampled_signal)
    return np.array(downsampled_signals)


def interpolation(downsampled_signals, target_length):
    upsampled_signals = []
    for signal in downsampled_signals:
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)
        interpolator = interp1d(x_old, signal, kind='cubic')
        upsampled_signal = interpolator(x_new)
        upsampled_signals.append(upsampled_signal)
    return np.array(upsampled_signals)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Define the Autoencoder Model
import torch.nn.functional as F

class ImprovedAutoencoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoder, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.encoder_bn1 = nn.BatchNorm1d(32)
        
        self.encoder_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)  # Downsample
        self.encoder_bn2 = nn.BatchNorm1d(64)
        
        self.encoder_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)  # Downsample
        self.encoder_bn3 = nn.BatchNorm1d(128)
        
        self.encoder_conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # Downsample
        self.encoder_bn4 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
    
        # Decoder
        self.decoder_convtrans1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)  # Upsample
        self.decoder_bn1 = nn.BatchNorm1d(128)
        
        self.decoder_convtrans2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample
        self.decoder_bn2 = nn.BatchNorm1d(64)
        
        self.decoder_convtrans3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample
        self.decoder_bn3 = nn.BatchNorm1d(32)
        
        self.decoder_conv1 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)  # Output layer
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)
        x1 = self.encoder_bn1(x1)
        x1 = self.relu(x1)
        # x1 = self.dropout(x1)
        
        x2 = self.encoder_conv2(x1)
        x2 = self.encoder_bn2(x2)
        x2 = self.relu(x2)
        # x2 = self.dropout(x2)
        
        x3 = self.encoder_conv3(x2)
        x3 = self.encoder_bn3(x3)
        x3 = self.relu(x3)
        # x3 = self.dropout(x3)
        
        x4 = self.encoder_conv4(x3)
        x4 = self.encoder_bn4(x4)
        x4 = self.relu(x4)
        # x4 = self.dropout(x4)
        
        # Decoder
        x = self.decoder_convtrans1(x4)
        x = self.decoder_bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        # Add skip connection from x3
        if x3.size(2) != x.size(2):
            x3 = F.interpolate(x3, size=x.size(2))
        x = x + x3
        
        x = self.decoder_convtrans2(x)
        x = self.decoder_bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        # Add skip connection from x2
        if x2.size(2) != x.size(2):
            x2 = F.interpolate(x2, size=x.size(2))
        x = x + x2
        
        x = self.decoder_convtrans3(x)
        x = self.decoder_bn3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        # Add skip connection from x1
        if x1.size(2) != x.size(2):
            x1 = F.interpolate(x1, size=x.size(2))
        x = x + x1
        
        # Output layer
        x = self.decoder_conv1(x)
        return x


def plot_signal_comparison(original_signals, interpolated_signals, refined_signals):
    def plot_signal(signal_idx):
        original_signal = original_signals[signal_idx]
        interpolated_signal = interpolated_signals[signal_idx]
        refined_signal = refined_signals[signal_idx]

        plt.figure(figsize=(18, 8))
        plt.plot(original_signal.flatten(), label='Original Signal', alpha=0.7, color='blue', linewidth=2)
        plt.plot(interpolated_signal.flatten(), label='Interpolated Signal', alpha=0.7, color='orange')
        plt.plot(refined_signal.flatten(), label='Refined Signal', alpha=0.7, color='green')

        plt.title(f'Signal Comparison for Index {signal_idx}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    interact(plot_signal, signal_idx=widgets.IntSlider(min=0, max=len(original_signals)-1, step=1, value=0))


def mask_nan_values(original_signals, replacement_signals):
    sigma=1
    masked_signals = []
    for orig_signal, repl_signal in zip(original_signals, replacement_signals):
        if orig_signal.shape != repl_signal.shape:
            raise ValueError("Original and replacement signals must have the same shape.")
        
        masked_signal = np.array(orig_signal)
        mask = np.isnan(orig_signal)
        masked_signal[mask] = repl_signal[mask]
        
        # Apply Gaussian smoothing
        masked_signal = gaussian_filter(masked_signal, sigma=sigma)
        
        masked_signals.append(masked_signal)
        
    return np.array(masked_signals)


def cubic_interpolation_imputation(signals):
    """
    Impute missing values in signals using cubic interpolation.

    Parameters:
    - signals: numpy array of shape (n_signals, n_samples) containing NaNs at positions to impute.

    Returns:
    - imputed_signals: numpy array of same shape as signals, with NaNs replaced by imputed values.
    """
    n_signals, n_samples = signals.shape
    imputed_signals = np.empty_like(signals)

    for i in range(n_signals):
        signal = signals[i, :]
        x = np.arange(n_samples)
        y = signal

        # Indices where y is not NaN
        valid_mask = ~np.isnan(y)
        invalid_mask = np.isnan(y)

        if np.sum(valid_mask) < 2:
            # Not enough valid points to interpolate; copy the original signal
            imputed_signals[i, :] = signal
            continue

        # Check if there are enough points for cubic interpolation
        if np.sum(valid_mask) < 4:
            # Use linear interpolation if not enough points for cubic
            interp_kind = 'linear'
        else:
            interp_kind = 'cubic'

        # Perform interpolation
        interp_func = interp1d(
            x[valid_mask], y[valid_mask], kind=interp_kind, bounds_error=False, fill_value="extrapolate"
        )

        # Interpolate at all positions
        y_interp = interp_func(x)

        # Replace NaNs with interpolated values
        imputed_signal = np.copy(signal)
        imputed_signal[invalid_mask] = y_interp[invalid_mask]

        imputed_signals[i, :] = imputed_signal

    return imputed_signals


def refine_signals(interpolated_results, device):

    
    # Convert the interpolated results to a tensor
        
    interpolated_tensor = torch.tensor(interpolated_results, dtype=torch.float32)
    
    # Create a dataset and dataloader
    interpolated_dataset = TensorDataset(interpolated_tensor)
    interpolated_dataloader = DataLoader(interpolated_dataset, batch_size=32, shuffle=False)


    # Parámetros iniciales
    n_folds = 10

    # Lista para almacenar las reconstrucciones
    reconstructions = []

    fold_index = 0
    while fold_index < n_folds:
        fold_index += 1

        # Instantiate the model
        model = ImprovedAutoencoder()

        model.load_state_dict(torch.load(f'best_model_fold_{fold_index}.pth'))
        model.to(device)
        # Model evaluation mode
        model.eval()


        refined_signals = []
    
        with torch.no_grad():
            # Process the data using the model
            for data in interpolated_dataloader:
                data = data[0].unsqueeze(1).to(device)
                outputs = model(data)
                refined_signals.append(outputs.cpu().numpy())



        # Concatenate and squeeze the refined signals
        refined_signals = np.concatenate(refined_signals, axis=0)
        refined_signals = np.squeeze(refined_signals, axis=1)

        # Guardar la reconstrucción de este fold
        reconstructions.append(refined_signals)

    # Calcular la media de las 10 reconstrucciones
    mean_reconstruction = np.mean(reconstructions, axis=0)

    # Devolver la media de las reconstrucciones
    return reconstructions


def Refinement_nan_pipeline(original_data, imputation_method, Target, device, oDATA):
    # Flatten and downsample the input data


    interpolated_data=[]
    refined_data=[]  

    downsampled_data = downsample(original_data, newL)
    down_oDATA = downsample(oDATA, newL)

    # Impute missing values using the specified imputation method
    if imputation_method == 'SSA':
        imputed_results = ssa_imputation_for_signal(downsampled_data, Target)
        interpolated_data = interpolation(imputed_results, 15000)
        refined_data = refine_signals(interpolated_data, device)

        print(imputation_method)
        print(f"Imputed results{np.array(imputed_results).shape}")
        print(f"Interpoleted data{np.array(interpolated_data).shape}")    
        print(f"Refined data{np.array(refined_data).shape}")

        

    elif imputation_method == 'Cubic':
        imputed_results = cubic_interpolation_imputation(downsampled_data)
        interpolated_data = interpolation(imputed_results, 15000)
        refined_data = refine_signals(interpolated_data, device)

        print(imputation_method)
        print(f"Imputed results{np.array(imputed_results).shape}")
        print(f"Interpoleted data{np.array(interpolated_data).shape}")    
        print(f"Refined data{np.array(refined_data).shape}")

    elif imputation_method == 'brits':
        imputed_results = imputation_brits(downsampled_data)

        imputed_results = np.array(imputed_results)

        for i in range(imputed_results.shape[0]):
            print(i)
            # Extraer la fila actual
            current_row = imputed_results[i, :, :]
    
            # Aplicar interpolación
            interpolated_row = interpolation(current_row, 15000)
    
            # Refinar señales
            refined_row = refine_signals(interpolated_row, device)
    
            # Agregar los resultados refinados a la lista
            interpolated_data.append(interpolated_row)
            refined_data.append(refined_row)

        print(imputation_method)
        print(f"Imputed results{imputed_results.shape}")
        print(f"Interpoleted data{get_shape(interpolated_data)}")    
        print(f"Refined data{get_shape(refined_data)}")

    
    elif imputation_method == 'KNN':
        imputed_results = knn_imputation(downsampled_data)
        interpolated_data = interpolation(imputed_results, 15000)
        refined_data = refine_signals(interpolated_data, device)

        print(imputation_method)
        print(f"Imputed results{np.array(imputed_results).shape}")
        print(f"Interpoleted data{np.array(interpolated_data).shape}")    
        print(f"Refined data{np.array(refined_data).shape}")
    else:
        raise ValueError(f"Unrecognized imputation method: {imputation_method}")
    
    # indicating_mask_1 = np.isnan(original_data) & (~np.isnan(oDATA))

    # indicating_mask_2 = np.isnan(downsampled_data) & (~np.isnan(down_oDATA))

    #mae, mse, mre = calculate_metrics(np.array(imputed_results), np.array(np.nan_to_num(down_oDATA)), indicating_mask_2)

    #print(f"Downsampled {imputation_method} metrics: MAE: {mae:.3f}, MSE: {mse:.3f}, MRE: {mre:.3f}")   

    




    # Mask NaN values in the interpolated and refined results
    #masked_interpolated_data = mask_nan_values(original_data, interpolated_data)
    #masked_refined_data = mask_nan_values(original_data, refined_data)
    return interpolated_data, refined_data, imputed_results    
    #return masked_interpolated_data, masked_refined_data


def get_shape(lst):
    if isinstance(lst, list):  # Check if the input is a list
        if len(lst) > 0 :
            return (len(lst),) + get_shape(lst[0])
        else:
            return (len(lst),)
    else:
        return ()  # Return empty tuple if not a list
    

def calculate_metrics(predicted_signals, original_signals, indicating_mask):
    
    # Apply the mask to filter out only the points where we want to calculate the metrics
    masked_predicted = predicted_signals[indicating_mask]
    masked_original = original_signals[indicating_mask]

    # Calculate MAE
    mae = np.mean(np.abs(masked_predicted - masked_original))

    # Calculate MSE
    mse = np.mean((masked_predicted - masked_original) ** 2)

    # Calculate MRE (Mean Relative Error), avoid division by zero
    # nonzero_mask = masked_original != 0
    # mre = np.mean(np.abs((masked_predicted[nonzero_mask] - masked_original[nonzero_mask]) / masked_original[nonzero_mask]))
   # Modify MRE calculation to match brits
    mre = np.sum(np.abs(masked_predicted - masked_original)) / np.sum(np.abs(masked_original) + 1e-12)

    return mae, mse, mre


def compute_all_metrics(imputed_signal, original_signal, indicating_mask):
    
    num_signals, signal_length = original_signal.shape
    high_freq=5
    low_freq= 1
    sampling_rate=1000

    # Initialize lists to store metrics for each signal
    MAE_list = []
    MRE_list = []
    RMSE_list = []
    Sim_list = []
    FSD_list = []
    RMSE_F_list = []
    RMSE_F_low_list = []
    RMSE_F_high_list = []

    # Define frequency bins for low and high frequencies
    freq_bins = np.fft.rfftfreq(signal_length, d=1/sampling_rate)
    low_freq_indices = np.where(freq_bins <= low_freq)[0]
    high_freq_indices = np.where(freq_bins >= high_freq)[0]

    # Loop over each signal
    for i in range(num_signals):
        orig_sig = original_signal[i, :]
        imp_sig = imputed_signal[i, :]
        mask = indicating_mask[i, :]

        # Ensure signals are 1D arrays
        orig_sig = np.ravel(orig_sig)
        imp_sig = np.ravel(imp_sig)
        mask = np.ravel(mask)

        # If there are no imputed positions in this signal, skip evaluation
        if not np.any(mask):
            continue

        # Extract the values at the imputed positions
        orig_values_at_imputed = orig_sig[mask]
        imp_values_at_imputed = imp_sig[mask]

        # Time Domain Metrics
        # MAE
        mae = np.mean(np.abs(orig_values_at_imputed - imp_values_at_imputed))
        MAE_list.append(mae)

        # MRE
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs((orig_values_at_imputed - imp_values_at_imputed) / orig_values_at_imputed)
            relative_errors = np.nan_to_num(relative_errors, nan=0.0, posinf=0.0, neginf=0.0)
        mre = np.mean(relative_errors)
        MRE_list.append(mre)

        # RMSE
        mse = np.mean((orig_values_at_imputed - imp_values_at_imputed) ** 2)
        rmse = np.sqrt(mse)
        RMSE_list.append(rmse)

        # Similarity Metric (Sim)
        orig_mean = np.mean(orig_values_at_imputed)
        imp_mean = np.mean(imp_values_at_imputed)
        numerator = np.sum((orig_values_at_imputed - orig_mean) * (imp_values_at_imputed - imp_mean))
        denominator = np.sqrt(np.sum((orig_values_at_imputed - orig_mean) ** 2) * np.sum((imp_values_at_imputed - imp_mean) ** 2))
        sim = numerator / denominator if denominator != 0 else 0
        Sim_list.append(sim)

        # Fraction of Standard Deviation (FSD)
        std_orig = np.std(orig_values_at_imputed)
        fsd = rmse / std_orig if std_orig != 0 else 0
        FSD_list.append(fsd)

        # Frequency Domain Metrics
        # Compute FFTs of the entire signals
        orig_sig_fft = fft(orig_sig)
        imp_sig_fft = fft(imp_sig)

        # Compute RMSE in frequency domain (entire frequency range)
        mse_f = np.mean(np.abs(orig_sig_fft - imp_sig_fft) ** 2)
        rmse_f = np.sqrt(mse_f)
        RMSE_F_list.append(rmse_f)

        # RMSE in Low Frequencies (RMSE_F_low)
        orig_sig_fft_low = orig_sig_fft[low_freq_indices]
        imp_sig_fft_low = imp_sig_fft[low_freq_indices]
        mse_f_low = np.mean(np.abs(orig_sig_fft_low - imp_sig_fft_low) ** 2)
        rmse_f_low = np.sqrt(mse_f_low)
        RMSE_F_low_list.append(rmse_f_low)

        # RMSE in High Frequencies (RMSE_F_high)
        orig_sig_fft_high = orig_sig_fft[high_freq_indices]
        imp_sig_fft_high = imp_sig_fft[high_freq_indices]
        mse_f_high = np.mean(np.abs(orig_sig_fft_high - imp_sig_fft_high) ** 2)
        rmse_f_high = np.sqrt(mse_f_high)
        RMSE_F_high_list.append(rmse_f_high)

    # Aggregate metrics
    metrics = {
        #OJO: ESTA MÉTRICA ESTÁ MAL: da la misma importancia a todas las señales, pero hay algunas señales que tienen más NaNs que otras
        'MAE_mean': np.mean(MAE_list) if MAE_list else None,
        # 'MAE_std': np.std(MAE_list),
        'MRE_mean': np.mean(MRE_list) if MRE_list else None,
        # 'MRE_std': np.std(MRE_list),
        'RMSE_mean': np.mean(RMSE_list) if RMSE_list else None,
        # 'RMSE_std': np.std(RMSE_list),
        'Sim_mean': np.mean(Sim_list) if Sim_list else None,
        # 'Sim_std': np.std(Sim_list),
        'FSD_mean': np.mean(FSD_list) if FSD_list else None,
        # 'FSD_std': np.std(FSD_list),
        'RMSE_F_mean': np.mean(RMSE_F_list) if RMSE_F_list else None,
         # 'RMSE_F_std': np.std(RMSE_F_list),
        'RMSE_F_Low_mean': np.mean(RMSE_F_low_list) if RMSE_F_low_list else None,
        # 'RMSE_F_Low_std': np.std(RMSE_F_low_list),
        'RMSE_F_High_mean': np.mean(RMSE_F_high_list) if RMSE_F_high_list else None,
        # 'RMSE_F_High_std': np.std(RMSE_F_high_list)
    }

    return metrics

def compute_all_metrics_kfolds(imputed_signal, original_signal, indicating_mask):
    """
    Calcula la media de las métricas para un conjunto de señales (3D array).
    """
    # Convertir a NumPy arrays si no lo son
    if isinstance(imputed_signal, list):
        imputed_signal = np.array(imputed_signal)
    if isinstance(original_signal, list):
        original_signal = np.array(original_signal)
    if isinstance(indicating_mask, list):
        indicating_mask = np.array(indicating_mask)

    # Inicializar diccionarios para almacenar las métricas acumuladas
    aggregated_metrics = {
        'MAE_mean': [],
        'MRE_mean': [],
        'RMSE_mean': [],
        'Sim_mean': [],
        'FSD_mean': [],
        'RMSE_F_mean': [],
        'RMSE_F_Low_mean': [],
        'RMSE_F_High_mean': []
    }

    # Procesar cada fila (primer dimensión)
    num_signals = len(imputed_signal)
    for i in range(num_signals):
        # Extraer la fila actual
        imputed_slice = imputed_signal[i, :, :]


        # Calcular métricas para la fila actual
        metrics = compute_all_metrics(imputed_slice, original_signal, indicating_mask)

        # Agregar métricas al acumulador
        for key in aggregated_metrics:
            if metrics[key] is not None:  # Evitar valores `None`
                aggregated_metrics[key].append(metrics[key])

    # Calcular la media de las métricas
    averaged_metrics = {key: np.mean(values) if values else None for key, values in aggregated_metrics.items()}

    return averaged_metrics


def modified_ssa_imputation(signal_with_blinks, target):
    """
    Impute missing values in the entire signal_with_blinks using Modified SSA.

    Parameters:
    - signal_with_blinks: 1D numpy array with the data to impute. Blinks must be NaNs.
    - target: 1D numpy array representing the stimulus or target signal.

    Returns:
    - imputed_signal: The signal with NaNs imputed (numpy array).
    """
    # Set length_limit (can be adjusted based on your data characteristics)
    length_limit = 200  # or any other appropriate value

    # Copy the signal to avoid modifying the original
    imputed_signal = signal_with_blinks.copy()
    time_series = signal_with_blinks.copy()

    # Find indices where time_series is NaN
    nan_indices = np.isnan(time_series)

    # Find start and end indices of each NaN segment
    gaps = []
    i = 0
    n = len(time_series)
    while i < n:
        if nan_indices[i]:
            inicioCurrentEvent = i
            while i < n and nan_indices[i]:
                i += 1
            finCurrentEvent = i - 1
            gaps.append((inicioCurrentEvent, finCurrentEvent))
        else:
            i += 1

    # For each gap, perform imputation using Modified SSA
    for inicioCurrentEvent, finCurrentEvent in gaps:
        imputed_values = ssa_impute_gap(time_series, inicioCurrentEvent, finCurrentEvent, length_limit, target)
        # Update imputed_signal with imputed_values
        imputed_signal[inicioCurrentEvent:finCurrentEvent+1] = imputed_values

    return imputed_signal

def ssa_impute_gap(time_series, inicioCurrentEvent, finCurrentEvent, length_limit, target):
    """
    Impute a single gap using Modified SSA.

    Parameters:
    - time_series: Original time series with NaNs.
    - inicioCurrentEvent: Start index of the gap.
    - finCurrentEvent: End index of the gap.
    - length_limit: Limit for the length of the front and back segments.
    - target: Target signal for augmentation.

    Returns:
    - imputed_values: Imputed values for the gap.
    """
    Ngap = finCurrentEvent - inicioCurrentEvent + 1  # Size of the gap

    # Find previous non-NaN segment before the gap
    prev_non_nan = inicioCurrentEvent - 1
    while prev_non_nan >= 0 and np.isnan(time_series[prev_non_nan]):
        prev_non_nan -= 1

    # Find next non-NaN segment after the gap
    next_non_nan = finCurrentEvent + 1
    while next_non_nan < len(time_series) and np.isnan(time_series[next_non_nan]):
        next_non_nan += 1

    # Normalization
    meanTimeSeries = np.nanmean(time_series)
    stdTimeSeries = np.nanstd(time_series)
    if stdTimeSeries == 0:
        stdTimeSeries = 1  # Avoid division by zero
    time_series_norm = (time_series - meanTimeSeries) / stdTimeSeries

    stdTarget = np.nanstd(target)
    if stdTarget == 0:
        stdTarget = 1
    target_norm = (target - np.nanmean(target)) / stdTarget

    # Prepare front (data before the gap)
    if prev_non_nan >= 0:
        front = time_series_norm[max(0, prev_non_nan - length_limit + 1):prev_non_nan + 1]
    else:
        front = np.array([])


    ##
    
    # Remove any NaNs in front
    front = front[~np.isnan(front)]

    # Prepare back (data after the gap)
    if next_non_nan < len(time_series):
        back = time_series_norm[next_non_nan:min(len(time_series), next_non_nan + length_limit)]
    else:
        back = np.array([])

    # Remove any NaNs in back
    back = back[~np.isnan(back)]

    # Augment with target if insufficient data
    if len(front) < length_limit:
        n_needed = length_limit - len(front)
        # Get data from target prior to the gap
        target_start_idx = max(0, inicioCurrentEvent - n_needed)
        target_end_idx = inicioCurrentEvent
        target_segment = target_norm[target_start_idx:target_end_idx]
        front = np.concatenate((target_segment, front))

    if len(back) < length_limit:
        n_needed = length_limit - len(back)
        # Get data from target after the gap
        target_start_idx = finCurrentEvent + 1
        target_end_idx = min(len(target_norm), target_start_idx + n_needed)
        target_segment = target_norm[target_start_idx:target_end_idx]
        back = np.concatenate((back, target_segment))

    nf = len(front)
    nb = len(back)

    # Step 2: Utilize SSA Technique
    data_r = None
    if nf > 1:
        try:
            L = min(450, nf - 1)
            N = int(0.25 * L)
            data_r = ssafor(front, L, N, Ngap)  # Forward SSA
        except Exception as e:
            data_r = None

    data_s = None
    if nb > 1:
        try:
            L = min(450, nb - 1)
            N = int(0.25 * L)
            data_s = ssafor(back[::-1], L, N, Ngap)  # Reverse SSA
            data_s = data_s[::-1]  # Revert to original order
        except Exception as e:
            data_s = None

    # Step 3: Weighting
    weights_backcast = np.linspace(0, 1, Ngap)
    weights_forecast = weights_backcast[::-1]

    # Step 4: Calculate interpolation
    # Linear interpolation as fallback
    xIn = []
    dataIn = []
    if nf > 0:
        xIn.extend([inicioCurrentEvent - nf + i for i in range(nf)])
        dataIn.extend(front)
    if nb > 0:
        xIn.extend([finCurrentEvent + 1 + i for i in range(nb)])
        dataIn.extend(back)
    xBuscado = [inicioCurrentEvent + i for i in range(Ngap)]
    if len(xIn) > 1:
        vinterp = np.interp(xBuscado, xIn, dataIn)
    else:
        # If not enough points for interpolation, use mean
        vinterp = np.full(Ngap, np.nanmean(time_series_norm))

    var_ts = np.nanvar(time_series_norm)

    # Conditions to decide which data to use
    if data_s is None or np.isnan(data_s).any():
        data_s = vinterp
    if data_r is None or np.isnan(data_r).any():
        data_r = vinterp

    # Combine SSA forecasts using weights
    if data_s is not None and data_r is not None:
        meanMatrix = data_s * weights_backcast + data_r * weights_forecast
    else:
        meanMatrix = vinterp

    # Unnormalization
    imputed_values = meanMatrix * stdTimeSeries + meanTimeSeries

    return imputed_values

def ssafor(Y, L, N, M):
    """
    SSA Forecasting function.

    Parameters:
    - Y: Input time series (1D numpy array).
    - L: Window length.
    - N: Number of eigenvectors to use.
    - M: Number of points to forecast.

    Returns:
    - F: Forecasted values (numpy array).
    """
    Y = np.array(Y).flatten()
    T = len(Y)
    K = T - L + 1

    # Embedding
    X = np.column_stack([Y[i:i+K] for i in range(L)])

    # Singular Value Decomposition
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Reconstruct time series using N components
    X_reconstructed = np.zeros_like(X)
    for i in range(N):
        X_reconstructed += s[i] * np.outer(U[:, i], Vt[i, :])

    # Diagonal averaging
    reconstructed_series = np.zeros(T)
    weight = np.zeros(T)
    for i in range(-L+1, K):
        diag_elements = np.diag(X_reconstructed, k=i)
        idx = max(0, i + L - 1)
        reconstructed_series[idx:idx + len(diag_elements)] += diag_elements
        weight[idx:idx + len(diag_elements)] += 1
    reconstructed_series /= weight

    # Forecasting
    # Build the forecasting matrix
    F = np.copy(reconstructed_series)
    for i in range(M):
        # Predict next value using last L-1 values
        if len(F) >= L:
            last_values = F[-(L-1):]
        else:
            last_values = np.hstack((np.zeros(L - 1 - len(F)), F))
        # Use coefficients from reconstructed components
        coeffs = np.polyfit(np.arange(len(last_values)), last_values, N - 1)
        next_value = np.polyval(coeffs, len(last_values))
        F = np.append(F, next_value)

    F = F[T:]
    return F

def ssa_imputation_for_signal(signals_with_blinks, targets):
    """
    Impute missing values for all signals using Modified SSA.

    Parameters:
    - signals_with_blinks: List of 1D numpy arrays, each with the data to impute. Blinks must be NaNs.
    - targets: List of 1D numpy arrays representing the stimulus or target signals corresponding to each signal.

    Returns:
    - imputed_signals: List of signals with NaNs imputed (list of numpy arrays).
    """
    imputed_signals = []
    for signal_with_blinks, target in zip(signals_with_blinks, targets):
        imputed_signal = modified_ssa_imputation(signal_with_blinks, target)
        imputed_signals.append(imputed_signal)
    return imputed_signals


def imputation_brits(downsampled_data):
    n_folds = 10
    newL = downsampled_data.shape[1]
    expanded_data = np.expand_dims(downsampled_data, axis=-1)

    reconstructions = []

    custom_optimizer = Adam(lr=0.0004)

    for fold_index in range(1, n_folds + 1):
        brits = BRITS(n_steps=newL, n_features=1, rnn_hidden_size=64, batch_size=128, epochs=500,  
                      optimizer=custom_optimizer, model_saving_strategy='better')

        # This will try to load 1 through 10
        model_path = f"brits_weights/brits_all_artificial_4_kfolds_{fold_index}.pypots"
        brits.load(model_path)

        dataset = {"X": expanded_data}
        imputed_results = brits.impute(dataset)

        reshaped_results = np.squeeze(imputed_results)
        reconstructions.append(reshaped_results)

    return reconstructions

    


def knn_imputation(signals):
    """
    Impute missing values in signals using K-Nearest Neighbors (KNN) imputation.

    Parameters:
    - signals: numpy array of shape (n_signals, n_samples) with NaN values.
    - n_neighbors: number of neighbors to use for KNN imputation (default is 7).

    Returns:
    - imputed_signals: numpy array of same shape as signals with NaNs replaced by imputed values.
    """

    n_neighbors=7
    # Create KNN imputer instance
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Fit the imputer and transform the data
    imputed_signals = imputer.fit_transform(signals)

    return imputed_signals


print("TRAIN")

flat_1_4_train_Ar = flatten_smoothpur_1_8(smoothpur_1_4_train_Ar)
flat_5_8_train_Ar = flatten_smoothpur_1_8(smoothpur_5_8_train_Ar)
flat_x_9_10_train_Ar, flat_y_9_10_train_Ar = flatten_smoothpur_9_12(smoothpur_9_10_train_Ar)
flat_x_11_12_train_Ar, flat_y_11_12_train_Ar = flatten_smoothpur_9_12(smoothpur_11_12_train_Ar)

mean_1_4_train_Ar = np.nanmean(flat_1_4_train_Ar)
std_1_4_train_Ar = np.nanstd(flat_1_4_train_Ar)
flat_1_4_normalized_train_Ar = (flat_1_4_train_Ar - mean_1_4_train_Ar) / std_1_4_train_Ar
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_train_Ar = np.nanmean(flat_5_8_train_Ar)
std_5_8_train_Ar = np.nanstd(flat_5_8_train_Ar)
flat_5_8_normalized_train_Ar = (flat_5_8_train_Ar - mean_5_8_train_Ar) / std_5_8_train_Ar
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_train_Ar = np.nanmean(flat_x_9_10_train_Ar)
std_x_9_10_train_Ar = np.nanstd(flat_x_9_10_train_Ar)
flat_x_9_10_normalized_train_Ar = (flat_x_9_10_train_Ar - mean_x_9_10_train_Ar) / std_x_9_10_train_Ar
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_train_Ar = np.nanmean(flat_y_9_10_train_Ar)
std_y_9_10_train_Ar = np.nanstd(flat_y_9_10_train_Ar)
flat_y_9_10_normalized_train_Ar = (flat_y_9_10_train_Ar - mean_y_9_10_train_Ar) / std_y_9_10_train_Ar
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_train_Ar = np.nanmean(flat_x_11_12_train_Ar)
std_x_11_12_train_Ar = np.nanstd(flat_x_11_12_train_Ar)
flat_x_11_12_normalized_train_Ar = (flat_x_11_12_train_Ar - mean_x_11_12_train_Ar) / std_x_11_12_train_Ar
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_train_Ar = np.nanmean(flat_y_11_12_train_Ar)
std_y_11_12_train_Ar = np.nanstd(flat_y_11_12_train_Ar)
flat_y_11_12_normalized_train_Ar = (flat_y_11_12_train_Ar - mean_y_11_12_train_Ar) / std_y_11_12_train_Ar
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_train_Ar.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_train_Ar.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_train_Ar.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_train_Ar.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_train_Ar.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_train_Ar.shape)

print("TEST")

flat_1_4_test_Ar = flatten_smoothpur_1_8(smoothpur_1_4_test_Ar)
flat_5_8_test_Ar = flatten_smoothpur_1_8(smoothpur_5_8_test_Ar)
flat_x_9_10_test_Ar, flat_y_9_10_test_Ar = flatten_smoothpur_9_12(smoothpur_9_10_test_Ar)
flat_x_11_12_test_Ar, flat_y_11_12_test_Ar = flatten_smoothpur_9_12(smoothpur_11_12_test_Ar)

mean_1_4_test_Ar = np.nanmean(flat_1_4_test_Ar)
std_1_4_test_Ar = np.nanstd(flat_1_4_test_Ar)
flat_1_4_normalized_test_Ar = (flat_1_4_test_Ar - mean_1_4_test_Ar) / std_1_4_test_Ar
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_test_Ar = np.nanmean(flat_5_8_test_Ar)
std_5_8_test_Ar = np.nanstd(flat_5_8_test_Ar)
flat_5_8_normalized_test_Ar = (flat_5_8_test_Ar - mean_5_8_test_Ar) / std_5_8_test_Ar
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_test_Ar = np.nanmean(flat_x_9_10_test_Ar)
std_x_9_10_test_Ar = np.nanstd(flat_x_9_10_test_Ar)
flat_x_9_10_normalized_test_Ar = (flat_x_9_10_test_Ar - mean_x_9_10_test_Ar) / std_x_9_10_test_Ar
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_test_Ar = np.nanmean(flat_y_9_10_test_Ar)
std_y_9_10_test_Ar = np.nanstd(flat_y_9_10_test_Ar)
flat_y_9_10_normalized_test_Ar = (flat_y_9_10_test_Ar - mean_y_9_10_test_Ar) / std_y_9_10_test_Ar
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_test_Ar = np.nanmean(flat_x_11_12_test_Ar)
std_x_11_12_test_Ar = np.nanstd(flat_x_11_12_test_Ar)
flat_x_11_12_normalized_test_Ar = (flat_x_11_12_test_Ar - mean_x_11_12_test_Ar) / std_x_11_12_test_Ar
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_test_Ar = np.nanmean(flat_y_11_12_test_Ar)
std_y_11_12_test_Ar = np.nanstd(flat_y_11_12_test_Ar)
flat_y_11_12_normalized_test_Ar = (flat_y_11_12_test_Ar - mean_y_11_12_test_Ar) / std_y_11_12_test_Ar
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_test_Ar.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_test_Ar.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_test_Ar.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_test_Ar.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_test_Ar.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_test_Ar.shape)

print("TRAIN")

flat_1_4_train_ori = flatten_smoothpur_1_8(smoothpur_1_4_train_ori)
flat_5_8_train_ori = flatten_smoothpur_1_8(smoothpur_5_8_train_ori)
flat_x_9_10_train_ori, flat_y_9_10_train_ori = flatten_smoothpur_9_12(smoothpur_9_10_train_ori)
flat_x_11_12_train_ori, flat_y_11_12_train_ori = flatten_smoothpur_9_12(smoothpur_11_12_train_ori)

mean_1_4_train_ori = np.nanmean(flat_1_4_train_ori)
std_1_4_train_ori = np.nanstd(flat_1_4_train_ori)
flat_1_4_normalized_train_ori = (flat_1_4_train_ori - mean_1_4_train_ori) / std_1_4_train_ori
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_train_ori = np.nanmean(flat_5_8_train_ori)
std_5_8_train_ori = np.nanstd(flat_5_8_train_ori)
flat_5_8_normalized_train_ori = (flat_5_8_train_ori - mean_5_8_train_ori) / std_5_8_train_ori
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_train_ori = np.nanmean(flat_x_9_10_train_ori)
std_x_9_10_train_ori = np.nanstd(flat_x_9_10_train_ori)
flat_x_9_10_normalized_train_ori = (flat_x_9_10_train_ori - mean_x_9_10_train_ori) / std_x_9_10_train_ori
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_train_ori = np.nanmean(flat_y_9_10_train_ori)
std_y_9_10_train_ori = np.nanstd(flat_y_9_10_train_ori)
flat_y_9_10_normalized_train_ori = (flat_y_9_10_train_ori - mean_y_9_10_train_ori) / std_y_9_10_train_ori
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_train_ori = np.nanmean(flat_x_11_12_train_ori)
std_x_11_12_train_ori = np.nanstd(flat_x_11_12_train_ori)
flat_x_11_12_normalized_train_ori = (flat_x_11_12_train_ori - mean_x_11_12_train_ori) / std_x_11_12_train_ori
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_train_ori = np.nanmean(flat_y_11_12_train_ori)
std_y_11_12_train_ori = np.nanstd(flat_y_11_12_train_ori)
flat_y_11_12_normalized_train_ori = (flat_y_11_12_train_ori - mean_y_11_12_train_ori) / std_y_11_12_train_ori
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_train_ori.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_train_ori.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_train_ori.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_train_ori.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_train_ori.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_train_ori.shape)

print("TEST")

flat_1_4_test_ori = flatten_smoothpur_1_8(smoothpur_1_4_test_ori)
flat_5_8_test_ori = flatten_smoothpur_1_8(smoothpur_5_8_test_ori)
flat_x_9_10_test_ori, flat_y_9_10_test_ori = flatten_smoothpur_9_12(smoothpur_9_10_test_ori)
flat_x_11_12_test_ori, flat_y_11_12_test_ori = flatten_smoothpur_9_12(smoothpur_11_12_test_ori)

mean_1_4_test_ori = np.nanmean(flat_1_4_test_ori)
std_1_4_test_ori = np.nanstd(flat_1_4_test_ori)
flat_1_4_normalized_test_ori = (flat_1_4_test_ori - mean_1_4_test_ori) / std_1_4_test_ori
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_test_ori = np.nanmean(flat_5_8_test_ori)
std_5_8_test_ori = np.nanstd(flat_5_8_test_ori)
flat_5_8_normalized_test_ori = (flat_5_8_test_ori - mean_5_8_test_ori) / std_5_8_test_ori
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_test_ori = np.nanmean(flat_x_9_10_test_ori)
std_x_9_10_test_ori = np.nanstd(flat_x_9_10_test_ori)
flat_x_9_10_normalized_test_ori = (flat_x_9_10_test_ori - mean_x_9_10_test_ori) / std_x_9_10_test_ori
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_test_ori = np.nanmean(flat_y_9_10_test_ori)
std_y_9_10_test_ori = np.nanstd(flat_y_9_10_test_ori)
flat_y_9_10_normalized_test_ori = (flat_y_9_10_test_ori - mean_y_9_10_test_ori) / std_y_9_10_test_ori
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_test_ori = np.nanmean(flat_x_11_12_test_ori)
std_x_11_12_test_ori = np.nanstd(flat_x_11_12_test_ori)
flat_x_11_12_normalized_test_ori = (flat_x_11_12_test_ori - mean_x_11_12_test_ori) / std_x_11_12_test_ori
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_test_ori = np.nanmean(flat_y_11_12_test_ori)
std_y_11_12_test_ori = np.nanstd(flat_y_11_12_test_ori)
flat_y_11_12_normalized_test_ori = (flat_y_11_12_test_ori - mean_y_11_12_test_ori) / std_y_11_12_test_ori
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_test_ori.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_test_ori.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_test_ori.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_test_ori.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_test_ori.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_test_ori.shape)

flats_normalized_train_Ar = np.concatenate([
    flat_1_4_normalized_train_Ar,
    flat_5_8_normalized_train_Ar,
    flat_x_9_10_normalized_train_Ar,
    flat_y_9_10_normalized_train_Ar,
    flat_x_11_12_normalized_train_Ar,
    flat_y_11_12_normalized_train_Ar
], axis=0)

flats_normalized_test_Ar = np.concatenate([
    flat_1_4_normalized_test_Ar,
    flat_5_8_normalized_test_Ar,
    flat_x_9_10_normalized_test_Ar,
    flat_y_9_10_normalized_test_Ar,
    flat_x_11_12_normalized_test_Ar,
    flat_y_11_12_normalized_test_Ar
], axis=0)


flats_normalized_train_ori = np.concatenate([
    flat_1_4_normalized_train_ori,
    flat_5_8_normalized_train_ori,
    flat_x_9_10_normalized_train_ori,
    flat_y_9_10_normalized_train_ori,
    flat_x_11_12_normalized_train_ori,
    flat_y_11_12_normalized_train_ori
], axis=0)

flats_normalized_test_ori = np.concatenate([
    flat_1_4_normalized_test_ori,
    flat_5_8_normalized_test_ori,
    flat_x_9_10_normalized_test_ori,
    flat_y_9_10_normalized_test_ori,
    flat_x_11_12_normalized_test_ori,
    flat_y_11_12_normalized_test_ori
], axis=0)


print("TRAIN")
newL=500

down_1_4_train_Ar = downsample(flat_1_4_normalized_train_Ar, newL)
down_5_8_train_Ar = downsample(flat_5_8_normalized_train_Ar, newL)
down_x_9_10_train_Ar, down_y_9_10_train_Ar = downsample(flat_x_9_10_normalized_train_Ar, newL), downsample(flat_y_9_10_normalized_train_Ar, newL)
down_x_11_12_train_Ar, down_y_11_12_train_Ar = downsample(flat_x_11_12_normalized_train_Ar, newL), downsample(flat_y_11_12_normalized_train_Ar, newL)
downs_train_Ar = downsample(flats_normalized_train_Ar, newL)

down_1_4_train_ori = downsample(flat_1_4_normalized_train_ori, newL)
down_5_8_train_ori = downsample(flat_5_8_normalized_train_ori, newL)
down_x_9_10_train_ori, down_y_9_10_train_ori = downsample(flat_x_9_10_normalized_train_ori, newL), downsample(flat_y_9_10_normalized_train_ori, newL)
down_x_11_12_train_ori, down_y_11_12_train_ori = downsample(flat_x_11_12_normalized_train_ori, newL), downsample(flat_y_11_12_normalized_train_ori, newL)
downs_train_ori = downsample(flats_normalized_train_ori, newL)

print('shape of downsampled smoothpur_1_4 :', down_1_4_train_ori.shape)
print('shape of downsampled smoothpur_5_8 :', down_5_8_train_ori.shape)
print('shape of downsampled smoothpur_x_9_10 :', down_x_9_10_train_ori.shape)
print('shape of downsampled smoothpur_y_9_10 :', down_y_9_10_train_ori.shape)
print('shape of downsampled smoothpur_x_11_12 :', down_x_11_12_train_ori.shape)
print('shape of downsampled smoothpur_y_11_12 :', down_y_11_12_train_ori.shape)
print('shape of downsampled All smoothpurs :', downs_train_ori.shape)

print("TEST")
down_1_4_test_Ar = downsample(flat_1_4_normalized_test_Ar, newL)
down_5_8_test_Ar = downsample(flat_5_8_normalized_test_Ar, newL)
down_x_9_10_test_Ar, down_y_9_10_test_Ar = downsample(flat_x_9_10_normalized_test_Ar, newL), downsample(flat_y_9_10_normalized_test_Ar, newL)
down_x_11_12_test_Ar, down_y_11_12_test_Ar = downsample(flat_x_11_12_normalized_test_Ar, newL), downsample(flat_y_11_12_normalized_test_Ar, newL)
downs_test_Ar = downsample(flats_normalized_test_Ar, newL)

down_1_4_test_ori = downsample(flat_1_4_normalized_test_ori, newL)
down_5_8_test_ori = downsample(flat_5_8_normalized_test_ori, newL)
down_x_9_10_test_ori, down_y_9_10_test_ori = downsample(flat_x_9_10_normalized_test_ori, newL), downsample(flat_y_9_10_normalized_test_ori, newL)
down_x_11_12_test_ori, down_y_11_12_test_ori = downsample(flat_x_11_12_normalized_test_ori, newL), downsample(flat_y_11_12_normalized_test_ori, newL)
downs_test_ori = downsample(flats_normalized_test_ori, newL)


print('shape of downsampled smoothpur_1_4 :', down_1_4_test_ori.shape)
print('shape of downsampled smoothpur_5_8 :', down_5_8_test_ori.shape)
print('shape of downsampled smoothpur_x_9_10 :', down_x_9_10_test_ori.shape)
print('shape of downsampled smoothpur_y_9_10 :', down_y_9_10_test_ori.shape)
print('shape of downsampled smoothpur_x_11_12 :', down_x_11_12_test_ori.shape)
print('shape of downsampled smoothpur_y_11_12 :', down_y_11_12_test_ori.shape)
print('shape of downsampled All smoothpurs :', downs_test_ori.shape)


print("TRAIN")
exp_1_4_train_Ar = np.expand_dims(down_1_4_train_Ar, axis=-1)
exp_5_8_train_Ar = np.expand_dims(down_5_8_train_Ar, axis=-1)
exp_x_9_10_train_Ar = np.expand_dims(down_x_9_10_train_Ar, axis=-1)
exp_y_9_10_train_Ar = np.expand_dims(down_y_9_10_train_Ar, axis=-1)
exp_x_11_12_train_Ar = np.expand_dims(down_x_11_12_train_Ar, axis=-1)
exp_y_11_12_train_Ar = np.expand_dims(down_y_11_12_train_Ar, axis=-1)
exps_train_Ar = np.expand_dims(downs_train_Ar, axis=-1)

print("TRAIN")
exp_1_4_train_ori = np.expand_dims(down_1_4_train_ori, axis=-1)
exp_5_8_train_ori = np.expand_dims(down_5_8_train_ori, axis=-1)
exp_x_9_10_train_ori = np.expand_dims(down_x_9_10_train_ori, axis=-1)
exp_y_9_10_train_ori = np.expand_dims(down_y_9_10_train_ori, axis=-1)
exp_x_11_12_train_ori = np.expand_dims(down_x_11_12_train_ori, axis=-1)
exp_y_11_12_train_ori = np.expand_dims(down_y_11_12_train_ori, axis=-1)
exps_train_ori = np.expand_dims(downs_train_ori, axis=-1)


print('shape of expanded smoothpur_1_4 :', exp_1_4_train_ori.shape)
print('shape of expanded smoothpur_5_8 :', exp_5_8_train_ori.shape)
print('shape of expanded smoothpur_x_9_10 :', exp_x_9_10_train_ori.shape)
print('shape of expanded smoothpur_y_9_10 :', exp_y_9_10_train_ori.shape)
print('shape of expanded smoothpur_x_11_12 :', exp_x_11_12_train_ori.shape)
print('shape of expanded smoothpur_y_11_12 :', exp_y_11_12_train_ori.shape)
print('shape of expanded All smoothpur :', exps_train_ori.shape)


print("TEST")
exp_1_4_test_Ar = np.expand_dims(down_1_4_test_Ar, axis=-1)
exp_5_8_test_Ar = np.expand_dims(down_5_8_test_Ar, axis=-1)
exp_x_9_10_test_Ar = np.expand_dims(down_x_9_10_test_Ar, axis=-1)
exp_y_9_10_test_Ar = np.expand_dims(down_y_9_10_test_Ar, axis=-1)
exp_x_11_12_test_Ar = np.expand_dims(down_x_11_12_test_Ar, axis=-1)
exp_y_11_12_test_Ar = np.expand_dims(down_y_11_12_test_Ar, axis=-1)
exps_test_Ar = np.expand_dims(downs_test_Ar, axis=-1)

exp_1_4_test_ori = np.expand_dims(down_1_4_test_ori, axis=-1)
exp_5_8_test_ori = np.expand_dims(down_5_8_test_ori, axis=-1)
exp_x_9_10_test_ori = np.expand_dims(down_x_9_10_test_ori, axis=-1)
exp_y_9_10_test_ori = np.expand_dims(down_y_9_10_test_ori, axis=-1)
exp_x_11_12_test_ori = np.expand_dims(down_x_11_12_test_ori, axis=-1)
exp_y_11_12_test_ori = np.expand_dims(down_y_11_12_test_ori, axis=-1)
exps_test_ori = np.expand_dims(downs_test_ori, axis=-1)


print('shape of expanded smoothpur_1_4 :', exp_1_4_test_ori.shape)
print('shape of expanded smoothpur_5_8 :', exp_5_8_test_ori.shape)
print('shape of expanded smoothpur_x_9_10 :', exp_x_9_10_test_ori.shape)
print('shape of expanded smoothpur_y_9_10 :', exp_y_9_10_test_ori.shape)
print('shape of expanded smoothpur_x_11_12 :', exp_x_11_12_test_ori.shape)
print('shape of expanded smoothpur_y_11_12 :', exp_y_11_12_test_ori.shape)
print('shape of expanded All smoothpur :', exps_test_ori.shape)

Target_1_4 = flatten_Target_1_8(smoothpur_1_4_test_ori)
Target_5_8 = flatten_Target_1_8(smoothpur_5_8_test_ori)
Target_x_9_10, Target_y_9_10= flatten_Target_9_12(smoothpur_9_10_test_ori)
Target_x_11_12, Target_y_11_12= flatten_Target_9_12(smoothpur_11_12_test_ori)


mean_T_1_4 = np.nanmean(Target_1_4)
std_T_1_4 = np.nanstd(Target_1_4)
Target_1_4_n = (Target_1_4 - mean_T_1_4) / std_T_1_4
# Target_1_4_n = min_max_normalize(Target_1_4)


mean_T_5_8= np.nanmean(Target_5_8)
std_T_5_8= np.nanstd(Target_5_8)
Target_5_8_n = (Target_5_8- mean_T_5_8) / std_T_5_8
# Target_5_8_n = min_max_normalize(Target_5_8)

mean_T_x_9_10 = np.nanmean(Target_x_9_10)
std_T_x_9_10 = np.nanstd(Target_x_9_10)
Target_x_9_10_n = (Target_x_9_10 - mean_T_x_9_10) / std_T_x_9_10
# Target_x_9_10_n = min_max_normalize(Target_x_9_10)

mean_T_y_9_10 = np.nanmean(Target_y_9_10)
std_T_y_9_10 = np.nanstd(Target_y_9_10)
Target_y_9_10_n = (Target_y_9_10 - mean_T_y_9_10) / std_T_y_9_10
# Target_y_9_10_n = min_max_normalize(Target_y_9_10)

mean_T_x_11_12 = np.nanmean(Target_x_11_12)
std_T_x_11_12 = np.nanstd(Target_x_11_12)
Target_x_11_12_n = (Target_x_11_12 - mean_T_x_11_12) / std_T_x_11_12
# Target_x_11_12_n = min_max_normalize(Target_x_11_12)

mean_T_y_11_12 = np.nanmean(Target_y_11_12)
std_T_y_11_12 = np.nanstd(Target_y_11_12)
Target_y_11_12_n = (Target_y_11_12 - mean_T_y_11_12) / std_T_y_11_12
# Target_y_11_12_n = min_max_normalize(Target_y_11_12)


Targets_n = np.concatenate([
    Target_1_4_n,
    Target_5_8_n,
    Target_x_9_10_n,
    Target_y_9_10_n,
    Target_x_11_12_n,
    Target_y_11_12_n
], axis=0)


oDATA = flats_normalized_test_ori   ## original_data 
oDATA_AR = flats_normalized_test_Ar  ## original_data_with_Artifical_blink
Target = Targets_n


imputation_method = 'SSA'
SSA_interpolated, SSA_refined, SSA_down = Refinement_nan_pipeline(oDATA_AR, imputation_method, Target, device,oDATA)

imputation_method = 'Cubic'
PCHIP_interpolated, PCHIP_refined, PCHIP_down = Refinement_nan_pipeline(oDATA_AR, imputation_method, Target, device, oDATA)

imputation_method = 'brits'
brits_interpolated, brits_refined, brits_down = Refinement_nan_pipeline(oDATA_AR, imputation_method, Target, device, oDATA)


imputation_method = 'KNN'
KNN_interpolated, KNN_refined, KNN_down = Refinement_nan_pipeline(oDATA_AR, imputation_method, Target, device, oDATA)

indicating_mask_2 = np.isnan(oDATA_AR) & (~np.isnan(oDATA))

down_oDATA = downsample(oDATA, newL)
down_oDATA_AR = downsample(oDATA_AR, newL)
indicating_mask_1 = np.isnan(down_oDATA_AR) & (~np.isnan(down_oDATA))

#brits tiene kfolds del brits y kfolds del Autoencoder
# Procesar cada fila (primer dimensión)
aggregated_metrics = {
        'MAE_mean': [],
        'MRE_mean': [],
        'RMSE_mean': [],
        'Sim_mean': [],
        'FSD_mean': [],
        'RMSE_F_mean': [],
        'RMSE_F_Low_mean': [],
        'RMSE_F_High_mean': []
    }
num_signals = len(brits_refined)
for i in range(num_signals):
    # Extraer la fila actual

    imputed_slice = brits_refined[i]


    # Calcular métricas para la fila actual
    metrics = compute_all_metrics_kfolds(imputed_slice, np.nan_to_num(oDATA), indicating_mask_2)

    # Agregar métricas al acumulador
    for key in aggregated_metrics:
        if metrics[key] is not None:  # Evitar valores `None`
            aggregated_metrics[key].append(metrics[key])

# Calcular la media de las métricas
metrics_brits_refined = {key: np.mean(values) if values else None for key, values in aggregated_metrics.items()}


metrics_SSA_refined = compute_all_metrics_kfolds(SSA_refined, np.nan_to_num(oDATA), indicating_mask_2)
metrics_PCHIP_refined = compute_all_metrics_kfolds(PCHIP_refined, np.nan_to_num(oDATA), indicating_mask_2)
metrics_KNN_refined = compute_all_metrics_kfolds(KNN_refined, np.nan_to_num(oDATA), indicating_mask_2)


metrics_SSA_interpolated = compute_all_metrics(SSA_interpolated, np.nan_to_num(oDATA), indicating_mask_2)
metrics_brits_interpolated = compute_all_metrics_kfolds(brits_interpolated, np.nan_to_num(oDATA), indicating_mask_2)
metrics_PCHIP_interpolated = compute_all_metrics(PCHIP_interpolated, np.nan_to_num(oDATA), indicating_mask_2)
metrics_KNN_interpolated = compute_all_metrics(KNN_interpolated, np.nan_to_num(oDATA), indicating_mask_2)

print(KNN_interpolated.shape)
print(np.array(SSA_down).shape)
print(np.array(down_oDATA).shape)
print(indicating_mask_1.shape)


metrics_SSA_down = compute_all_metrics(np.array(SSA_down), np.nan_to_num(down_oDATA), indicating_mask_1)
metrics_brits_down = compute_all_metrics_kfolds(np.array(brits_down), np.nan_to_num(down_oDATA), indicating_mask_1)
metrics_PCHIP_down = compute_all_metrics(np.array(PCHIP_down), np.nan_to_num(down_oDATA), indicating_mask_1)
metrics_KNN_down = compute_all_metrics(np.array(KNN_down), np.nan_to_num(down_oDATA), indicating_mask_1)

# Print the results
print("All SPT:")
print('-' * 70)
print("Computed Metrics SSA Refined:")
for key, value in metrics_SSA_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics brits Refined:")
for key, value in metrics_brits_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics PCHIP Refined:")
for key, value in metrics_PCHIP_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics KNN Refined:")
for key, value in metrics_KNN_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics SSA Interpolated:")
for key, value in metrics_SSA_interpolated.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics brits Interpolated:")
for key, value in metrics_brits_interpolated.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)


print("Computed Metrics PCHIP Interpolated:")
for key, value in metrics_PCHIP_interpolated.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics KNN Interpolated:")
for key, value in metrics_KNN_interpolated.items():
    print(f"{key}: {value:.4f}")
    
    
print("Computed Metrics SSA Downsampled:")
for key, value in metrics_SSA_down.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics brits Downsampled:")
for key, value in metrics_brits_down.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)


print("Computed Metrics PCHIP Downsampled:")
for key, value in metrics_PCHIP_down.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics KNN Downsampled:")
for key, value in metrics_KNN_down.items():
    print(f"{key}: {value:.4f}")

############################################################################################
# --- inject missing values in the test signals between samples 5000 and 9000 ---
oDATA_AR1 = oDATA_AR.copy()
oDATA_AR1[:, 5000:9000] = np.nan

# --- now run your four imputation pipelines on oDATA_AR1 instead of oDATA_AR ---
imputation_method = 'SSA'
SSA_interpolated, SSA_refined, SSA_down = Refinement_nan_pipeline(oDATA_AR1, imputation_method, Target, device, oDATA)

imputation_method = 'Cubic'
PCHIP_interpolated, PCHIP_refined, PCHIP_down = Refinement_nan_pipeline(oDATA_AR1, imputation_method, Target, device, oDATA)

imputation_method = 'brits'
brits_interpolated, brits_refined, brits_down = Refinement_nan_pipeline(oDATA_AR1, imputation_method, Target, device, oDATA)

imputation_method = 'KNN'
KNN_interpolated, KNN_refined, KNN_down = Refinement_nan_pipeline(oDATA_AR1, imputation_method, Target, device, oDATA)


# --- recompute the mask for exactly that injected region ---
indicating_mask_region = np.isnan(oDATA_AR1) & (~np.isnan(oDATA))


#brits tiene kfolds del brits y kfolds del Autoencoder
# Procesar cada fila (primer dimensión)
aggregated_metrics = {
        'MAE_mean': [],
        'MRE_mean': [],
        'RMSE_mean': [],
        'Sim_mean': [],
        'FSD_mean': [],
        'RMSE_F_mean': [],
        'RMSE_F_Low_mean': [],
        'RMSE_F_High_mean': []
    }
num_signals = len(brits_refined)
for i in range(num_signals):
    # Extraer la fila actual

    imputed_slice = brits_refined[i]


    # Calcular métricas para la fila actual
    metrics = compute_all_metrics_kfolds(imputed_slice, np.nan_to_num(oDATA), indicating_mask_region)

    # Agregar métricas al acumulador
    for key in aggregated_metrics:
        if metrics[key] is not None:  # Evitar valores `None`
            aggregated_metrics[key].append(metrics[key])

# Calcular la media de las métricas
metrics_brits_refined = {key: np.mean(values) if values else None for key, values in aggregated_metrics.items()}


metrics_SSA_refined = compute_all_metrics_kfolds(SSA_refined, np.nan_to_num(oDATA), indicating_mask_region)
metrics_PCHIP_refined = compute_all_metrics_kfolds(PCHIP_refined, np.nan_to_num(oDATA), indicating_mask_region)
metrics_KNN_refined = compute_all_metrics_kfolds(KNN_refined, np.nan_to_num(oDATA), indicating_mask_region)


metrics_SSA_interpolated = compute_all_metrics(SSA_interpolated, np.nan_to_num(oDATA), indicating_mask_region)
metrics_brits_interpolated = compute_all_metrics_kfolds(brits_interpolated, np.nan_to_num(oDATA), indicating_mask_region)
metrics_PCHIP_interpolated = compute_all_metrics(PCHIP_interpolated, np.nan_to_num(oDATA), indicating_mask_region)
metrics_KNN_interpolated = compute_all_metrics(KNN_interpolated, np.nan_to_num(oDATA), indicating_mask_region)

print(KNN_interpolated.shape)
print(np.array(SSA_down).shape)
print(np.array(down_oDATA).shape)
print(indicating_mask_region.shape)


# Print the results
print("All SPT:")
print('-' * 70)
print("Computed Metrics SSA Refined:")
for key, value in metrics_SSA_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics brits Refined:")
for key, value in metrics_brits_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics PCHIP Refined:")
for key, value in metrics_PCHIP_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics KNN Refined:")
for key, value in metrics_KNN_refined.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics SSA Interpolated:")
for key, value in metrics_SSA_interpolated.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics brits Interpolated:")
for key, value in metrics_brits_interpolated.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)


print("Computed Metrics PCHIP Interpolated:")
for key, value in metrics_PCHIP_interpolated.items():
    print(f"{key}: {value:.4f}")
print('-' * 50)

print("Computed Metrics KNN Interpolated:")
for key, value in metrics_KNN_interpolated.items():
    print(f"{key}: {value:.4f}")
    
