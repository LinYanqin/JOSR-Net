import numpy as np
import os
import tensorflow as tf
import scipy.io as sio


def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)


def gene_full_data(x):
    real_data = x[:,:,:, 0]
    imag_data = x[:,:,:, 1]
    full_data = real_data + 1j * imag_data
    return full_data


def data_generator(f_ytrain, batch_size, y_axis=1, x_axis=128):
    idx = np.arange(len(f_ytrain))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(f_ytrain), batch_size*(i+1)))] for i in range(len(f_ytrain)//batch_size)]
    while True:
        for batch in batches:
            Ytrain = []
            Y = np.zeros((len(batch), y_axis, x_axis, 2))
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
                y_train_temp = np.reshape(y_train_temp, (y_axis, x_axis, 2), 'F')
                Y[i,:,:,:]=y_train_temp
            full_f_data = gene_full_data(Y)
            label_t = np.fft.ifft(full_f_data)  # 标签时域数据谱
            label_f_all = np.reshape(full_f_data, [len(batch), y_axis, x_axis, 1])
            label_t_data = np.reshape(label_t, [len(batch), y_axis, x_axis, 1])
            label_t_data = complex2real(label_t_data)
            label_f_data = complex2real(label_f_all)
            yield {'Input': label_t_data}, {'dc_SDN3': label_f_data, 'dc_SDN4': label_f_data, 'dc_SDN5': label_f_data,
                                            'dc_SDN6': label_f_data}
