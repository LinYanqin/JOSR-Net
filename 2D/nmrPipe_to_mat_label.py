import nmrglue as ng
import scipy.io as sio
import os
#

dic, data = ng.pipe.read('./nmrpipe_data/label3D.dat')
dic1, data1 = ng.pipe.read('./nmrpipe_data/labelCN.dat')
sio.savemat(os.path.join('./Res_data/BMRB_data/label_3D.mat'), {'label_3D': data})
sio.savemat(os.path.join('./Res_data/BMRB_data/labelCN.mat'), {'labelCN': data1})



