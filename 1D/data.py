import h5py
import numpy    as np
import scipy.io as sio
import tensorflow as tf
import os


def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)


def gene_full_data(x):
    real_data = x[:,:,:, 0]
    imag_data = x[:,:,:, 1]
    full_data = real_data + 1j * imag_data
    return full_data


def load_batch(nb_train, path, y_axis=1, x_axis=128):
    input_data = sio.loadmat(path + 'input_data.mat')
    mask_data = sio.loadmat(path + 'mask.mat')
    input_data = input_data['input_data']
    mask_data = mask_data['mask_data']
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft2(input_f_data)
    input_data = np.reshape(input_t, [nb_train, y_axis, x_axis, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_axis, 1])
    input_data = complex2real(input_data)
    return input_data, mask

