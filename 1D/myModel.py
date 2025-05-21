import sys, os

import scipy.io as sio
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import initializers
from keras.layers import concatenate, Conv2DTranspose, add
from keras.layers import LeakyReLU, Input, Conv2D, Lambda
import temperatureUpdate
import tensorflow as tf
from keras.engine.topology import Layer
  

class entropy_reg(tf.keras.regularizers.Regularizer):

    def __init__(self, entropyMult):
        self.entropyMult = entropyMult

    def __call__(self, logits):
        normDist = tf.nn.softmax(logits,1)
        logNormDist = tf.log(normDist+1e-20)
        
        rowEntropies = -tf.reduce_sum(tf.multiply(normDist, logNormDist),1)
        sumRowEntropies = tf.reduce_sum(rowEntropies)
        
        multiplier = self.entropyMult
        return multiplier*sumRowEntropies

    def get_config(self):
        return {'entropyMult': float(self.entropyMult)}

#######################################################################

# Create the trainable logits matrix
class CreateSampleMatrix(Layer):
    def __init__(self,mux_in,mux_out,entropyMult,name=None,**kwargs):
        self.mux_in = mux_in
        self.mux_out = mux_out
        self.entropyMult = entropyMult
        super(CreateSampleMatrix, self).__init__(name=name,**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(name='TrainableLogits',
                                      shape=(self.mux_out, self.mux_in),
                                      initializer=initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                      regularizer=entropy_reg(self.entropyMult),
                                      trainable=True)
        super(CreateSampleMatrix, self).build(input_shape)  

    def call(self, x):    
        return self.kernel
    
    def compute_output_shape(self, input_shape):
        return (self.mux_out, self.mux_in)
    
    def get_config(self):
        base_config = super(CreateSampleMatrix, self).get_config()
        return base_config    
    
#######################################################################

# Create a mask for the logits dependent on the already sampled classes.
# This mask ensures that one class is only sampled once over the M distributions in the next layer
def MaskingLogits(inp):
    logits = inp[0]
    inpData = inp[1]
    mux_in = logits.shape[1]
    mux_out = logits.shape[0]
    mux_out_scalar = logits.shape.as_list()[0]
     
    # Create gumbel noise, which is different for every patch in the batch. So size of GN: [Batch size, mux_out, mux_in]
    # Where mux_in is the original amount of Fourier bins of the signal and mux_out is the amount to be selected
    seed_num = 256
    np.random.seed(seed_num)
    GN = -tf.log(-tf.log(tf.random_uniform(tf.stack([tf.shape(inpData)[0], mux_out, mux_in], 0), 0, 1, seed=seed_num) + 1e-20) + 1e-20)

    #Repeat the logits over (dynamic) batch size by adding zeros of this shape
    dummyForRepOverBS = tf.zeros_like(GN)
    logitsRep = logits + dummyForRepOverBS
        
    mask = tf.ones(tf.stack([tf.shape(inpData)[0],mux_in],0))
    maskedLogits = [None]*mux_out_scalar
      
    #Shuffle rows in order to apply sampling without replacement in random row order
    shuffledRows = np.arange(mux_out_scalar)
    np.random.shuffle(shuffledRows)
    
    unnormProbs = tf.exp(logitsRep)     # constructing the logistic normal distribution, by the way transform all number to positive
    for i in range(mux_out_scalar):      
        row = shuffledRows[i]
        unnormProbRow = unnormProbs[:,row,:]
  
        maskedLogitRow = tf.log(tf.multiply(unnormProbRow,mask) + 1e-20)
        maskedLogits[row] = maskedLogitRow
    
        #Find next mask: change a one to a zero where the hard sample will be taken
        hardSampleForMasking = tf.one_hot(tf.argmax(maskedLogitRow+GN[:,row,:],axis=1),depth=mux_in)
        mask = mask - hardSampleForMasking
    maskedLogits = tf.stack(maskedLogits,axis=1)
    
    # Return GN as well to make sure the same GN is used in the softSampling layer
    return [maskedLogits, GN]

def MaskingLogits_output_shape(input_shape):
    outputShapeTup = (input_shape[1][0],)+input_shape[0]             
    return [outputShapeTup,outputShapeTup]

######################################################################    

# Apply row-based sampling from the Gumbel-softmax distribution, with a variable temperature parameter, depending on the epoch
class SoftSampling(Layer):
    def __init__(self, tempIncr=1, n_epochs=1, batchPerEpoch=32, name=None, **kwargs):
        self.tempIncr = tempIncr
        self.n_epochs = n_epochs
        self.batchPerEpoch = batchPerEpoch
        super(SoftSampling, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.step = K.variable(0)
        super(SoftSampling, self).build(input_shape)

    def call(self, inp):
        maskedLogits = inp[0]
        GN = inp[1]

        # Find temperature for gumbel softmax based on epoch and update epoch
        epoch = self.step / self.batchPerEpoch
        Temp = temperatureUpdate.temperature_update_tf(self.tempIncr, epoch, self.n_epochs)

        updateSteps = []
        updateSteps.append((self.step, self.step + 1))
        self.add_update(updateSteps, inp)

        return tf.nn.softmax((maskedLogits + GN) / Temp, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0])

######################################################################  
        
# Apply hard sampling of the soft-samples (only in the forward pass)
# This hard sampling does happen unordered: non-sequential
def hardSampling(maskedSoftSamples):
    hardSamples = tf.one_hot(tf.argmax(maskedSoftSamples,axis=-1),depth=maskedSoftSamples.shape[-1])
    return tf.stop_gradient(hardSamples - maskedSoftSamples) + maskedSoftSamples

def identity_output_shape(input_shape):
    return input_shape


def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])
    else:
        return None


def dc(inp):
    generated = inp[0]
    mask = inp[1]
    x_dc = inp[2]
    X_k = real2complex(x_dc)
    gene_complex = real2complex(generated)
    gene_complex = tf.transpose(gene_complex, [0, 3, 1, 2])
    gene_ifft = tf.ifft(gene_complex)
    gene_ifft = tf.transpose(gene_ifft, [0, 2, 3, 1])
    gene_channel = complex_real(gene_ifft)
    gene_sample_channel = tf.multiply(gene_channel, (1.0 - mask))
    gene_sample_complex = real2complex(gene_sample_channel)
    gene_sample_complex = tf.transpose(gene_sample_complex, [0, 3, 1, 2])
    X_k = tf.transpose(X_k, [0, 3, 1, 2])
    out_fft = X_k + gene_sample_complex
    output_complex = tf.fft(out_fft)
    output_complex = tf.transpose(output_complex, [0, 2, 3, 1])
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real, output_imag], axis=-1)
    return output

def dc_tdomain(inp):
    generated = inp[0]
    mask = inp[1]
    x_dc = inp[2]
    X_k = real2complex(x_dc)
    gene = tf.multiply(generated, (1.0 - mask))
    gene_t = real2complex(gene)
    out_fft = X_k + gene_t
    output_complex = out_fft
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real,output_imag], axis=-1)
    return output

def complex_real(x):
    output_real = tf.cast(tf.real(x), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(x), dtype=tf.float32)
    output = tf.concat([output_real, output_imag], axis=-1)
    return output


######################################################################

    
def full_model(input_dim, DPS_dim, mux_out, tempIncr, entropyMult, n_epochs, batchPerEpoch):

    DPS_in = DPS_dim[0] * DPS_dim[1]
 
    input_layer = Input(shape=input_dim, name="Input")
    shape_input = input_layer.shape.as_list() #[BS, 32,32]
    print('shape input: ', shape_input)

    """
    =============================================================================
        SUB-SAMPLING NETWORK
    =============================================================================
    """
    def Amat(samples):
        Amatrix = tf.reshape(tf.reduce_sum(samples, axis=1), (-1, DPS_dim[0], DPS_dim[1], 1))
        A = tf.pad(Amatrix, [[0, 0], [0, input_dim[0] - DPS_dim[0]], [0, input_dim[1] - DPS_dim[1]], [0, 0]])
        return A

    def Ax(inp):
        x = inp[0]
        Amatrix = inp[1]
        y = tf.multiply(x,Amatrix)
        return y


    logits = CreateSampleMatrix(mux_out=mux_out, mux_in=DPS_in, entropyMult=entropyMult, name="CreateSampleMatrix")(
        input_layer)
    print('logits shape: ', logits.shape)

    maskedLogits = Lambda(MaskingLogits, name="MaskingLogits", output_shape=MaskingLogits_output_shape)(
        [logits, input_layer])
    print('masked logits shape: ', maskedLogits[0].shape)

    samples = SoftSampling(tempIncr=tempIncr, n_epochs=n_epochs, batchPerEpoch=batchPerEpoch, name="SoftSampling")(
        maskedLogits)
    print('soft samples shape: ', samples.shape)

    samples = Lambda(hardSampling, name="OneHotArgmax", output_shape=identity_output_shape)(samples)
    print('hard samples shape:', samples.shape)

    Amatrix = Lambda(Amat, name="AtranA_0")(samples)
    upSampledInp = Lambda(Ax, name="HardSampling")([input_layer, Amatrix])  # Amatrix is the samplign matrix
    print('Shape after inverse A: ', upSampledInp.shape)   # upSampleInp is the undersampled data

    def ft_layer(k_temp_real):
        k_temp1 = real2complex(k_temp_real)
        k_temp = tf.transpose(k_temp1, [0, 3, 1, 2])
        temp = tf.fft(k_temp)
        temp = tf.transpose(temp, [0, 2, 3, 1])
        temp_1 = complex_real(temp)
        return temp_1

    def HRU0_layer(x, upSampledInp, Amatrix, concat_tensor, stage, block):
        bn_name_base = 'bn_' + block + '_'
        conv_name_base = 'complex_conv_' + block + '_'
        relu_name_base = 'leaky_relu_' + block + '_'
        n1 = 1
        n2 = 3
        n3 = 1
        n4 = 2
        conv1 = Conv2D(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'a')(concat_tensor)
        conv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'a')(conv1)
        conv1 = LeakyReLU(alpha=0.2, name=relu_name_base + 'a')(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'b')(conv1)
        conv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'b')(conv2)
        conv2 = LeakyReLU(alpha=0.2, name=relu_name_base + 'b')(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'c')(conv2)
        conv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'c')(conv3)
        conv3 = LeakyReLU(alpha=0.2, name=relu_name_base + 'c')(conv3)

        deconv3 = Conv2DTranspose(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'c')(conv3)
        deconv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'c')(deconv3)
        deconv3 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'c')(deconv3)

        up3 = concatenate([deconv3, conv2], name=block + '_concat3')

        deconv2 = Conv2DTranspose(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'b')(up3)
        deconv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'b')(deconv2)
        deconv2 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'b')(deconv2)

        up2 = concatenate([deconv2, conv1], name=block + '_concat2')

        deconv1 = Conv2DTranspose(filters=16, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'a')(up2)
        deconv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'a')(deconv1)
        deconv1 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'a')(deconv1)

        up1 = concatenate([deconv1, concat_tensor], name=block + '_concat1')

        conv0 = Conv2DTranspose(filters=2, kernel_size=(n1, n2), strides=(1, 1), padding="same",
                                name='out' + conv_name_base)(up1)
        conv0 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='out' + bn_name_base)(conv0)
        conv0 = LeakyReLU(alpha=0.2, name='out' + relu_name_base)(conv0)

        up0 = add([conv0, x], name='add_' + block)

        if stage:
            temp = Lambda(dc_tdomain, name='dc_tdomain_' + block)([up0, Amatrix, upSampledInp])
        else:
            temp = Lambda(dc, name='dc_' + block)([up0, Amatrix, upSampledInp])
        return temp, deconv1, deconv2, deconv3

    def HRU_layer(x, upSampledInp, Amatrix, concat_tensor, de1, de2, de3, stage, block):
        bn_name_base = 'bn_' + block + '_'
        conv_name_base = 'complex_conv_' + block + '_'
        relu_name_base = 'leaky_relu_' + block + '_'
        n1 = 1
        n2 = 3
        n3 = 1
        n4 = 2
        conv1 = Conv2D(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'a')(concat_tensor)
        conv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'a')(conv1)
        conv1 = LeakyReLU(alpha=0.2, name=relu_name_base + 'a')(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'b')(conv1)
        conv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'b')(conv2)
        conv2 = LeakyReLU(alpha=0.2, name=relu_name_base + 'b')(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'c')(conv2)
        conv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'c')(conv3)
        conv3 = LeakyReLU(alpha=0.2, name=relu_name_base + 'c')(conv3)

        deconv3 = Conv2DTranspose(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'c')(conv3)
        deconv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'c')(deconv3)
        deconv3 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'c')(deconv3)

        up3 = concatenate([deconv3, conv2, de3], name=block + '_concat3')

        deconv2 = Conv2DTranspose(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'b')(up3)
        deconv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'b')(deconv2)
        deconv2 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'b')(deconv2)

        up2 = concatenate([deconv2, conv1, de2], name=block + '_concat2')

        deconv1 = Conv2DTranspose(filters=16, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'a')(up2)
        deconv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'a')(deconv1)
        deconv1 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'a')(deconv1)

        up1 = concatenate([deconv1, concat_tensor, de1], name=block + '_concat1')

        conv0 = Conv2D(filters=2, kernel_size=(n1, n2), strides=(1, 1), padding="same",
                       name='out' + conv_name_base)(up1)
        conv0 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='out' + bn_name_base)(conv0)
        conv0 = LeakyReLU(alpha=0.2, name='out' + relu_name_base)(conv0)

        up0 = add([conv0, x], name='add_' + block)

        if stage:
            temp = Lambda(dc_tdomain, name='dc_tdomain_' + block)([up0, Amatrix, upSampledInp])
        else:
            temp = Lambda(dc, name='dc_' + block)([up0, Amatrix, upSampledInp])
        return temp, deconv1, deconv2, deconv3


    temp_1 = Lambda(ft_layer, name='ft_layer')(upSampledInp)

    temp_2, de1_s1, de2_s1, de3_s1 = HRU0_layer(temp_1, upSampledInp, Amatrix, temp_1, stage=False, block='SDN1')
    temp_3, de1_s2, de2_s2, de3_s2 = HRU_layer(temp_2, upSampledInp, Amatrix, temp_2, de1_s1, de2_s1, de3_s1, stage=False, block='SDN2')
    temp_31 = concatenate([temp_2, temp_3], name='concat_2')
    temp_4, de1_s3, de2_s3, de3_s3 = HRU_layer(temp_3, upSampledInp, Amatrix, temp_31, de1_s2, de2_s2, de3_s2, stage=False, block='SDN3')
    temp_41 = concatenate([temp_2, temp_3, temp_4], name='concat_3')
    temp_5, de1_s4, de2_s4, de3_s4 = HRU_layer(temp_4, upSampledInp, Amatrix, temp_41, de1_s3, de2_s3, de3_s3, stage=False, block='SDN4')
    temp_51 = concatenate([temp_2, temp_3, temp_4, temp_5], name='concat_4')
    temp_6, de1_s5, de2_s5, de3_s5 = HRU_layer(temp_5, upSampledInp, Amatrix, temp_51, de1_s4, de2_s4, de3_s4, stage=False, block='SDN5')
    temp_61 = concatenate([temp_2, temp_3, temp_4, temp_5, temp_6], name='concat_5')
    temp_7, _, _, _ = HRU_layer(temp_6, upSampledInp, Amatrix, temp_61, de1_s5, de2_s5, de3_s5, stage=False, block='SDN6')

    model = Model(inputs=input_layer, outputs=[temp_4, temp_5, temp_6, temp_7])

    return model

    
def recModel(fid_rows, fid_cols):

    upSampledInp = Input(shape=[fid_rows, fid_cols, 2], name="HardSampling")
    Amatrix = Input(shape=[fid_rows, fid_cols, 1], name='AtranA_0')

    def ft_layer(k_temp_real):
        k_temp1 = real2complex(k_temp_real)
        k_temp = tf.transpose(k_temp1, [0, 3, 1, 2])
        temp = tf.fft(k_temp)
        temp = tf.transpose(temp, [0, 2, 3, 1])
        temp_1 = complex_real(temp)
        return temp_1

    def HRU0_layer(x, upSampledInp, Amatrix, concat_tensor, stage, block):
        bn_name_base = 'bn_' + block + '_'
        conv_name_base = 'complex_conv_' + block + '_'
        relu_name_base = 'leaky_relu_' + block + '_'
        n1 = 1
        n2 = 3
        n3 = 1
        n4 = 2
        conv1 = Conv2D(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'a')(concat_tensor)
        conv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'a')(conv1)
        conv1 = LeakyReLU(alpha=0.2, name=relu_name_base + 'a')(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'b')(conv1)
        conv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'b')(conv2)
        conv2 = LeakyReLU(alpha=0.2, name=relu_name_base + 'b')(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'c')(conv2)
        conv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'c')(conv3)
        conv3 = LeakyReLU(alpha=0.2, name=relu_name_base + 'c')(conv3)

        deconv3 = Conv2DTranspose(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'c')(conv3)
        deconv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'c')(deconv3)
        deconv3 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'c')(deconv3)

        up3 = concatenate([deconv3, conv2], name=block + '_concat3')

        deconv2 = Conv2DTranspose(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'b')(up3)
        deconv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'b')(deconv2)
        deconv2 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'b')(deconv2)

        up2 = concatenate([deconv2, conv1], name=block + '_concat2')

        deconv1 = Conv2DTranspose(filters=16, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'a')(up2)
        deconv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'a')(deconv1)
        deconv1 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'a')(deconv1)

        up1 = concatenate([deconv1, concat_tensor], name=block + '_concat1')

        conv0 = Conv2DTranspose(filters=2, kernel_size=(n1, n2), strides=(1, 1), padding="same",
                                name='out' + conv_name_base)(up1)
        conv0 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='out' + bn_name_base)(conv0)
        conv0 = LeakyReLU(alpha=0.2, name='out' + relu_name_base)(conv0)

        up0 = add([conv0, x], name='add_' + block)

        if stage:
            temp = Lambda(dc_tdomain, name='dc_tdomain_' + block)([up0, Amatrix, upSampledInp])
        else:
            temp = Lambda(dc, name='dc_' + block)([up0, Amatrix, upSampledInp])
        return temp, deconv1, deconv2, deconv3

    def HRU_layer(x, upSampledInp, Amatrix, concat_tensor, de1, de2, de3, stage, block):
        bn_name_base = 'bn_' + block + '_'
        conv_name_base = 'complex_conv_' + block + '_'
        relu_name_base = 'leaky_relu_' + block + '_'
        n1 = 1
        n2 = 3
        n3 = 1
        n4 = 2
        conv1 = Conv2D(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'a')(concat_tensor)
        conv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'a')(conv1)
        conv1 = LeakyReLU(alpha=0.2, name=relu_name_base + 'a')(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'b')(conv1)
        conv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'b')(conv2)
        conv2 = LeakyReLU(alpha=0.2, name=relu_name_base + 'b')(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                       name=conv_name_base + 'c')(conv2)
        conv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name=bn_name_base + 'c')(conv3)
        conv3 = LeakyReLU(alpha=0.2, name=relu_name_base + 'c')(conv3)

        deconv3 = Conv2DTranspose(filters=64, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'c')(conv3)
        deconv3 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'c')(deconv3)
        deconv3 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'c')(deconv3)

        up3 = concatenate([deconv3, conv2, de3], name=block + '_concat3')

        deconv2 = Conv2DTranspose(filters=32, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'b')(up3)
        deconv2 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'b')(deconv2)
        deconv2 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'b')(deconv2)

        up2 = concatenate([deconv2, conv1, de2], name=block + '_concat2')

        deconv1 = Conv2DTranspose(filters=16, kernel_size=(n1, n2), strides=(n3, n4), padding="same",
                                  name='de' + conv_name_base + 'a')(up2)
        deconv1 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='de' + bn_name_base + 'a')(deconv1)
        deconv1 = LeakyReLU(alpha=0.2, name='de' + relu_name_base + 'a')(deconv1)

        up1 = concatenate([deconv1, concat_tensor, de1], name=block + '_concat1')

        conv0 = Conv2D(filters=2, kernel_size=(n1, n2), strides=(1, 1), padding="same",
                       name='out' + conv_name_base)(up1)
        conv0 = BatchNormalization(gamma_initializer='glorot_normal', axis=3, name='out' + bn_name_base)(conv0)
        conv0 = LeakyReLU(alpha=0.2, name='out' + relu_name_base)(conv0)

        up0 = add([conv0, x], name='add_' + block)

        if stage:
            temp = Lambda(dc_tdomain, name='dc_tdomain_' + block)([up0, Amatrix, upSampledInp])
        else:
            temp = Lambda(dc, name='dc_' + block)([up0, Amatrix, upSampledInp])
        return temp, deconv1, deconv2, deconv3


    temp_1 = Lambda(ft_layer, name='ft_layer')(upSampledInp)

    temp_2, de1_s1, de2_s1, de3_s1 = HRU0_layer(temp_1, upSampledInp, Amatrix, temp_1, stage=False, block='SDN1')
    temp_3, de1_s2, de2_s2, de3_s2 = HRU_layer(temp_2, upSampledInp, Amatrix, temp_2, de1_s1, de2_s1, de3_s1, stage=False, block='SDN2')
    temp_31 = concatenate([temp_2, temp_3], name='concat_2')
    temp_4, de1_s3, de2_s3, de3_s3 = HRU_layer(temp_3, upSampledInp, Amatrix, temp_31, de1_s2, de2_s2, de3_s2, stage=False, block='SDN3')
    temp_41 = concatenate([temp_2, temp_3, temp_4], name='concat_3')
    temp_5, de1_s4, de2_s4, de3_s4 = HRU_layer(temp_4, upSampledInp, Amatrix, temp_41, de1_s3, de2_s3, de3_s3, stage=False, block='SDN4')
    temp_51 = concatenate([temp_2, temp_3, temp_4, temp_5], name='concat_4')
    temp_6, de1_s5, de2_s5, de3_s5 = HRU_layer(temp_5, upSampledInp, Amatrix, temp_51, de1_s4, de2_s4, de3_s4, stage=False, block='SDN5')
    temp_61 = concatenate([temp_2, temp_3, temp_4, temp_5, temp_6], name='concat_5')
    temp_7, _, _, _ = HRU_layer(temp_6, upSampledInp, Amatrix, temp_61, de1_s5, de2_s5, de3_s5, stage=False, block='SDN6')

    model = Model(inputs=[upSampledInp, Amatrix], outputs=[temp_4, temp_5, temp_6, temp_7])

    return model
