import nmrglue as ng
import scipy.io as sio
import os
#

filepath = "./Processed_data/BMRB_data/"

dic, data = ng.pipe.read(os.path.join(filepath,"fid_temp_ZF.dat"))

sio.savemat(os.path.join(filepath,"test.mat"),{'data':data})



