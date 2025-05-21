import numpy as np
from keras import backend as K
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import Model

import myModel
import scipy.io as sio
import os
from data import load_batch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# =============================================================================
versionName = "JOSR_model/128_32/"
weightFile = "_weights"
savedir = os.path.join(os.path.dirname(__file__), versionName)


data_index = 'BMRB_data/'
rec = True

# =============================================================================

# %%
"""
=============================================================================
    Load testset
=============================================================================
"""
# define all setting
train_M = 129       # direct dimension
fid_rows = 1
fid_cols = 128      # indirect dimension
mux_out = 32        # sampling points
DPS_rows = 1
DPS_cols = 128      # sampling matrix dimension

# load test data
print('[INFO] load data ... ')
data_path = os.path.join('./test/', data_index)

input_size = (fid_rows, fid_cols)           # sizes of the inputs to the network
DPS_size = (DPS_rows, DPS_cols)             # sizes of the sampling matrix to the network
input_dim = [fid_rows, fid_cols, 2]         # Dimensions of the inputs to the network
DPS_dim = [DPS_rows, DPS_cols, 2]           # Dimensions of the sampling matrix to the network

"""
=============================================================================
    Parameter definitions
=============================================================================
"""
# Subsampling parameters
learningrate = 1e-4
subSampLrMult = 1000
tempIncr = 3.0

# Parameters for entropy penalty multiplier (EM)
startEM = 0.005  # Start value of the multiplier
EM = startEM
entropyMult = K.variable(value=EM)

# Training parameters
n_epochs = 300  # Number of epochs during training
batch_size = 32
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

model = myModel.full_model(
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

model_recon = myModel.recModel(fid_rows, fid_cols)
optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model_recon.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weight)
model_recon.load_weights(os.path.join(savedir, weightFile + ".hdf5"), by_name=True)


# %%
"""
=============================================================================
    Inference
=============================================================================
"""

print('[*] start draw sampling mask ...')
model_sampling = Model(inputs=model.input, outputs=model.get_layer("AtranA_0").output)
pattern = model_sampling.predict_on_batch(tf.zeros((1, fid_rows, fid_cols, 2)))[0]
sio.savemat(os.path.join(data_path, "DPSmask.mat"), {'DPSmask': pattern})


if rec:
    _, _, input_data, _, mask = load_batch(train_M, data_path)
    y_SDN1, y_SDN2, y_SDN3, y_SDN4 = model_recon.predict([input_data, mask], batch_size=train_M, verbose=1)
    sio.savemat(os.path.join(data_path, "rec_1FID.mat"), {'rec1': y_SDN4})

print("[*] Job finished!")

