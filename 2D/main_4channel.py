import scipy.io as sio
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
import tensorflow as tf
import myModel_4channel
from keras import backend as K
import tensorlayer as tl
from generator_DPS_4channel import *
from keras.utils.vis_utils import plot_model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

"""
=============================================================================
    Load and prepare the datasets
=============================================================================
"""
# define the data setting
fid_rows = 64           # indirect dimension 1
fid_cols = 64           # indirect dimension 2
mux_out = 123           # sampling point
DPS_rows = 32           # sampling matrix rows
DPS_cols = 64           # sampling matrix cols

# load data
print('[INFO] load data ... ')
data_path = './JOSRdata/'
train_M = 20000
training_gooddata_path = str(data_path) + "Label"
validation_gooddata_path = str(data_path) + "ValidationGoodData"

y_train = tl.files.load_file_list(path=training_gooddata_path,
                                  regx='.*.txt',
                                  printable=False)
y_train.sort(key=tl.files.natural_keys)

y_val = tl.files.load_file_list(path=validation_gooddata_path,
                                regx='.*.txt',
                                printable=False)
y_val.sort(key=tl.files.natural_keys)

train_all_num = len(y_train)
val_all_num = len(y_val)
f_ytrain = []
f_yval = []

for i in range(train_all_num):
    f_ytrain.append(os.path.join(training_gooddata_path, y_train[i]))

for i in range(val_all_num):
    f_yval.append(os.path.join(validation_gooddata_path, y_val[i]))

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

versionName = "JOSR_model/{}-{}-{}".format(DPS_rows, DPS_cols, mux_out)

savedir = os.path.join(os.path.dirname(__file__), versionName)
if savedir:
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

# Training parameters
n_epochs = 300  # Number of epochs during training
batch_size = 32  # Batch size for data generators
batchPerEpoch = 16000 // batch_size  # Number of batches used per epoch

"""
=============================================================================
    Model definition
=============================================================================
"""

model = myModel_4channel.full_model(
    input_dim,
    DPS_dim,
    mux_out,
    tempIncr,
    entropyMult,
    n_epochs,
    batchPerEpoch)

## Print model summary:
if os.path.exists(os.path.join(savedir, "model.txt")):
    os.remove(os.path.join(savedir, "model.txt"))
with open(os.path.join(savedir, "model.txt"), 'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))
plot_model(model, to_file=os.path.join(savedir, "model.png"), show_shapes=True)


"""
=============================================================================
    Initialize and compile model
=============================================================================
"""

## Define Optimizer:
import AdamWithLearningRateMultiplier

lr_mult = {}
lr_mult['CreateSampleMatrix'] = subSampLrMult
optimizer = AdamWithLearningRateMultiplier.Adam_lr_mult(lr=learningrate, multipliers=lr_mult)

# Define loss and metric
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

loss = [mae, mae, mae, mae]
loss_weight = [1.0, 1.0, 1.0, 1.0]

# Compile model
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weight)

print("Learning rate: ", learningrate)
print("Temperature increase : ", tempIncr)
print("subSampLrMult: ", subSampLrMult)
print('startEM: ', startEM)
print('mux_out: ', mux_out)

# %%
"""
=============================================================================
    Training
=============================================================================
"""
from keras.callbacks import LambdaCallback


# save directory to save distribution at end of training. Make an empty list to not save anything


def IncrEntropyMult(epoch):
    value = startEM
    print("EntropyPen mult:", value)
    K.set_value(entropyMult, value)


EntropyMult_cb = LambdaCallback(on_epoch_end=lambda epoch, log: IncrEntropyMult(epoch))

callbacks = [
    ModelCheckpoint(os.path.join(savedir, 'weights-{epoch:03d}-{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=1,
                    save_best_only=True, save_weights_only=True, mode='auto', period=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto'),
    EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min', restore_best_weights=True),
    EntropyMult_cb
]
"""
outputPerNEpochs:      How often do you want the callback to provide output
outputLastNepochs:     First argument indicates the X last epochs you want the callback to prpvide output
"""

H = model.fit_generator(data_generator(f_ytrain, batch_size, y_axis=fid_rows, x_axis=fid_cols),
                        steps_per_epoch=len(f_ytrain) // batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        validation_data=data_generator(f_yval, batch_size, y_axis=fid_rows, x_axis=fid_cols),
                        validation_steps=len(f_yval) // batch_size,
                        callbacks=callbacks)


"""
=============================================================================
    Save model weights
=============================================================================
"""

if savedir:
    model.save_weights(os.path.join(savedir, '_weights.hdf5'))
    print('Saved: ', versionName, ' in ', savedir)
else:
    print("Not saved anything yet")
#