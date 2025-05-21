import nmrglue as ng
#import mat4py as mt
import pylab
import numpy as np
import scipy.io as sio
import os
import subprocess


def postprocess_data(test_path, y_axis=64, x_axis=64, z_axis=280):
    name_res = test_path + 'res.mat'
    res_data = sio.loadmat(name_res)['res']
    res_data = np.transpose(res_data, (1, 2, 0, 3))
    res_3D = np.zeros((2 * y_axis, 2 * x_axis, z_axis))
    for k in range(z_axis):
        temp = np.fft.ifft2(np.fft.ifftshift(res_data[:, :, k, 0]))
        res_3D[::2, ::2, k] = np.real(temp)
        res_3D[::2, 1::2, k] = np.imag(temp)
        temp = np.fft.ifft2(np.fft.ifftshift(res_data[:, :, k, 1]))
        res_3D[1::2, ::2, k] = np.real(temp)
        res_3D[1::2, 1::2, k] = np.imag(temp)
    name_res = test_path + 'res_full.mat'
    res_3D = np.transpose(res_3D, (1, 0, 2))  # is used for BMRB, if data is simulation data, this operation needs move
    sio.savemat(name_res, {'res_3D': res_3D})


def fid_with_nmrpipe(data_path_idx, nmrpipe_dat_path, com_path):

    dic, data = ng.pipe.read(f'{nmrpipe_dat_path}')

    print(data.ndim)
    print(data.shape)
    print(data.dtype)

    M = sio.loadmat(os.path.join('./Res_data/', data_path_idx, 'res_full.mat'))
    Data = M['res_3D']
    Data = np.array(Data, dtype='float32')
    ng.pipe.write('./nmrpipe_data/res_full.dat', dic, Data, overwrite=True)

    M = sio.loadmat(os.path.join('./Processed_data/', data_path_idx, 'label_3D.mat'))
    Data = M['fid']
    Data = np.array(Data, dtype='float32')
    ng.pipe.write('./nmrpipe_data/label_3D.dat', dic, Data, overwrite=True)

    print("[*] FID to nmrpipe Job finished!")

    subprocess.run(["csh",f'{com_path}'])
    print("[*] nmrpipe shell Job finished!")

    dic, data = ng.pipe.read('./nmrpipe_data/resCN.dat')
    dic1, data1 = ng.pipe.read('./nmrpipe_data/resCN3D.dat')
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'resCN3D.mat'), {'resCN3D': data1})
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'resCN.mat'), {'resCN': data})
    dic, data = ng.pipe.read('./nmrpipe_data/label3D.dat')
    dic1, data1 = ng.pipe.read('./nmrpipe_data/labelCN.dat')
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'label_3D.mat'), {'label_3D': data})
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'labelCN.mat'), {'labelCN': data1})

    print("[*] nmrpipe to FID Job finished!")


