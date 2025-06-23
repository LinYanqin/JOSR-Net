import nmrglue as ng
import scipy.io as sio
import os
#

filepath = "./test/BMRB_data/"

dic, data = ng.pipe.read(os.path.join(filepath,"test.ft2"))

sio.savemat(os.path.join(filepath,"test.mat"),{'data':data})



