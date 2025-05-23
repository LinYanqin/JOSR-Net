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
    full_data = sio.loadmat(path + 'label_data.mat')
    input_data = sio.loadmat(path + 'input_data.mat')
    mask_data = sio.loadmat(path + 'mask.mat')
    input_data = input_data['input_data']
    full_data = full_data['label_data']
    mask_data = mask_data['mask_data']
    full_f_data = gene_full_data(full_data)
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft2(input_f_data)
    label_t = np.fft.ifft2(full_f_data)
    x_dc = np.fft.ifft2(input_f_data)
    label_f_all = np.reshape(full_f_data, [nb_train, y_axis, x_axis, 1])
    label_t_data = np.reshape(label_t, [nb_train, y_axis, x_axis, 1])
    input_data = np.reshape(input_t, [nb_train, y_axis, x_axis, 1])
    x_dc = np.reshape(x_dc, [nb_train, y_axis, x_axis, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_axis, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_all)
    return label_f_data, label_t_data, input_data, x_dc, mask

