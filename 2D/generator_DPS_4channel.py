import numpy as np
import os
import tensorflow as tf
import time
import scipy.io as sio


def gene_full_data(x):
    real_data1 = x[:,:,:, 0]
    imag_data1 = x[:,:,:, 1]
    real_data2 = x[:,:,:, 2]
    imag_data2 = x[:,:,:, 3]
    full_data1 = real_data1 + 1j * imag_data1
    full_data2 = real_data2 + 1j * imag_data2
    return np.stack([full_data1, full_data2], axis=-1)


def complex2real(x):
    data1_real = np.real(x[:, :, :, 0])
    data1_imag = np.imag(x[:, :, :, 0])
    data2_real = np.real(x[:, :, :, 1])
    data2_imag = np.imag(x[:, :, :, 1])
    return np.stack([data1_real, data1_imag, data2_real, data2_imag], axis=-1)


def ifftshift(x):
    s0 = (x.shape[-2] // 2)  # 32
    s1 = (x.shape[-1] // 2)  # 32
    x = np.concatenate([x[:, :, s0:, :], x[:, :, :s0, :]], axis=-2)  # exchange down and up
    x = np.concatenate([x[:, :, :, s1:], x[:, :, :, :s1]], axis=-1)  # exchange right and left
    return x


def data_generator(f_ytrain, batch_size, y_axis=64, x_axis=64):
    idx = np.arange(len(f_ytrain))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(f_ytrain), batch_size*(i+1)))] for i in range(len(f_ytrain)//batch_size)]
    while True:
        for batch in batches:
            Ytrain = []
            Y = np.zeros((len(batch), y_axis, x_axis, 4))
            for i in batch:
                Ytrain.append(f_ytrain[i])
            for i in range(len(batch)):
                y_train_temp = []
                file_path2 = Ytrain[i]
                fr2 = open(file_path2)
                for line in fr2.readlines():
                    lineArr = line.strip().split()
                    y_train_temp.append(lineArr[:])
                y_train_temp = np.asarray(y_train_temp, dtype=np.float32)
                y_train_temp = np.reshape(y_train_temp, (y_axis, x_axis, 4), 'F')
                Y[i,:,:,:]=y_train_temp
            full_f_data = gene_full_data(Y)     # [?, y_axis, x_aixs, 2]
            full_f_data = np.transpose(full_f_data, [0, 3, 1, 2])
            full_f_ifftshift = ifftshift(full_f_data)
            label_t = np.fft.ifft2(full_f_ifftshift)  # 标签时域数据谱
            label_f_all = np.transpose(full_f_data, [0, 2, 3, 1])
            label_t_data = np.transpose(label_t, [0, 2, 3, 1])
            label_t_data = complex2real(label_t_data)     # [?, y_axis, x_aixs, 4]
            label_f_data = complex2real(label_f_all)      # [?, y_axis, x_aixs, 4]

            yield {'Input': label_t_data}, {'dc_SDN3': label_f_data, 'dc_SDN4': label_f_data, 'dc_SDN5': label_f_data,
                                            'dc_SDN6': label_f_data}
