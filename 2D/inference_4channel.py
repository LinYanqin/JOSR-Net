import numpy as np
from keras import backend as K
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import Model

import myModel_4channel
import scipy.io as sio
import os
from data_4channel import load_data
from postprocess import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# =============================================================================
versionName = "JOSR_model/32-64-123/"
weightFile = "_weights"
savedir = os.path.join(os.path.dirname(__file__), versionName)

rec = True
if rec:
    nmrpipe_dat_path = './Processed_data/BMRB_data/fid_temp_ZF.dat'
    com_path = './recFT.com'
else:
    nmrpipe_dat_path = None
    com_path = None

# =============================================================================

# %%
"""
=============================================================================
    Load testset
=============================================================================
"""
# define the data setting
train_M = 280         # direct dimension
fid_rows = 64         # direct dimension 1
fid_cols = 64         # direct dimension 2
mux_out = 123         # sampling points
DPS_rows = 32         # sampling matrix rows
DPS_cols = 64         # sampling matrix cols


# load data
print('[INFO] load data path ... ')
data_index = 'BMRB_data/'
data_path = os.path.join('./Processed_data/', data_index)

input_size = (fid_rows, fid_cols)               # sizes of the inputs to the network
DPS_size = (DPS_rows, DPS_cols)                 # sizes of the sampling matrix to the network
input_dim = [fid_rows, fid_cols, 4]             # Dimensions of the inputs to the network
DPS_dim = [DPS_rows, DPS_cols, 4]               # Dimensions of the samping matrix to the network


# %%
"""
=============================================================================
    Parameter definitions
=============================================================================
"""
# Subsampling parameters
tempIncr = 5.0  # Multiplier for temperature  parameter of softmax function. The temperature drops from (tempIncr*TempUpdateBasisTemp) to (tempIncr*TempUpdateFinalTemp) defined in temperatureUpdate.py file
learningrate = 1e-4
subSampLrMult = 1000  # Multiplier of learning rate for trainable unnormalized logits in A (with respect to the learning rate of the reconstruction part of the network)

# Parameters for entropy penalty multiplier (EM)
startEM = 0.0005  # Start value of the multiplier
EM = startEM
entropyMult = K.variable(value=EM)

# Training parameters
n_epochs = 300  # Number of epochs during training
batch_size = 32  # Batch size for data generators
batchPerEpoch = 16000 // batch_size  # Number of batches used per epoch
"""
=============================================================================
    Model definition
=============================================================================
"""
import AdamWithLearningRateMultiplier

lr_mult = {}
lr_mult['CreateSampleMatrix'] = subSampLrMult

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


loss = [mae, mae, mae, mae]
loss_weight = [1.0, 1.0, 1.0, 1.0]



model = myModel_4channel.full_model(
    input_dim,
    DPS_dim,
    mux_out,
    tempIncr,
    entropyMult,
    n_epochs,
    batchPerEpoch)

optimizer = AdamWithLearningRateMultiplier.Adam_lr_mult(lr=learningrate, multipliers=lr_mult)
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weight)
model.load_weights(os.path.join(savedir, weightFile + ".hdf5"))

model_recon = myModel_4channel.recModel(fid_rows, fid_cols)
optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model_recon.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weight)
model_recon.load_weights(os.path.join(savedir, weightFile + ".hdf5"), by_name=True)


#
"""
=============================================================================
    Inference
=============================================================================
"""

print('[*] start draw sampling mask ...')
model_sampling = Model(inputs=model.input, outputs=model.get_layer("AtranA_0").output)
pattern = model_sampling.predict_on_batch(tf.zeros((1, fid_rows, fid_cols, 4)))[0]
sio.savemat(os.path.join(data_path, "DPSmask.mat"), {'DPSmask': pattern})

if rec:
    print('[*] load data ... ')
    pred = np.zeros([train_M, fid_rows, fid_cols, 2], dtype=np.complex)
    input_data, mask_batch, factor = load_data(data_path, train_M, fid_rows, fid_cols)

    print('[*] start testing ...')

    y_SDN1, y_SDN2, y_SDN3, y_SDN4 = model_recon.predict([input_data, mask_batch], batch_size=32, verbose=1)
    x1_real = y_SDN4[:, :, :, 0]
    x1_imag = y_SDN4[:, :, :, 1]
    x2_real = y_SDN4[:, :, :, 2]
    x2_imag = y_SDN4[:, :, :, 3]
    pred1_c = x1_real + 1j * x1_imag
    pred2_c = x2_real + 1j * x2_imag
    pred[:, :, :, 0] = pred1_c
    pred[:, :, :, 1] = pred2_c
    for q in range(train_M):
        pred[q, :, :, 0] = pred[q, :, :, 0] * factor[0, q]
        pred[q, :, :, 1] = pred[q, :, :, 1] * factor[1, q]

    test_path = "./Res_data/" + str(data_index)
    sio.savemat(os.path.join(test_path, "res.mat"), {'res': pred})
    postprocess_data(test_path, z_axis=train_M)
    fid_with_nmrpipe(data_path_idx=data_index, nmrpipe_dat_path=nmrpipe_dat_path, com_path=com_path)

print("[*] job finished!")

