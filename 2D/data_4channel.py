import h5py
import numpy    as np
import scipy.io as sio
import os

def gene_full_data(x):
    real_data1 = x[:, :, :, 0]
    imag_data1 = x[:, :, :, 1]
    real_data2 = x[:, :, :, 2]
    imag_data2 = x[:, :, :, 3]
    full_data1 = real_data1 + 1j * imag_data1
    full_data2 = real_data2 + 1j * imag_data2
    return np.stack([full_data1, full_data2], axis=-1)

def ifftshift(x):
    s0 = (x.shape[-2] // 2)  # 32
    s1 = (x.shape[-1] // 2)  # 32
    x = np.concatenate([x[:, :, s0:, :], x[:, :, :s0, :]], axis=-2)  # exchange down and up
    x = np.concatenate([x[:, :, :, s1:], x[:, :, :, :s1]], axis=-1)  # exchange right and left
    return x

def complex2real(x):
    data1_real = np.real(x[:, :, :, 0])
    data1_imag = np.imag(x[:, :, :, 0])
    data2_real = np.real(x[:, :, :, 1])
    data2_imag = np.imag(x[:, :, :, 1])
    return np.stack([data1_real, data1_imag, data2_real, data2_imag], axis=-1)


def load_data(data_path_1, direct_num, y_axis=64, x_axis=64):
    mask = []
    mask_data = sio.loadmat(data_path_1 + 'mask3D.mat')
    mask_data = mask_data['mask']
    for i in range(direct_num):
        mask.append(mask_data)

    factor_data1 = sio.loadmat(data_path_1 + 'factor1.mat')
    factor1 = factor_data1['factor1']
    factor_data2 = sio.loadmat(data_path_1 + 'factor2.mat')
    factor2 = factor_data2['factor2']
    factor = np.concatenate([factor1, factor2], axis=0)

    under_data1 = sio.loadmat(data_path_1 + 'inputreal1.mat')
    under_data1 = under_data1['input_real1']     # [?, y, x, 2]
    under_data2 = sio.loadmat(data_path_1 + 'inputreal2.mat')
    under_data2 = under_data2['input_real2']     # [?, y, x, 2]
    under_data = np.concatenate([under_data1, under_data2], axis=-1)    # [?, y, x, 4]

    under_f_data = gene_full_data(under_data)
    under_f_data = np.transpose(under_f_data, [0, 3, 1, 2])
    under_f_data = ifftshift(under_f_data)
    input_t = np.fft.ifft2(under_f_data)

    print('over loading the data')

    input_data = np.transpose(input_t, [0, 2, 3, 1])
    mask1 = np.reshape(mask, [direct_num, y_axis, x_axis, 1])
    input_data = complex2real(input_data)

    return input_data, mask1, factor



