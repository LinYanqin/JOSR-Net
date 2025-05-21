import tensorflow as tf 

TempUpdateBasisTemp = 1.0
TempUpdateFinalTemp = 0.1

def temperature_update_tf(tempIncr, epoch, n_epochs):
    TempUpdate = (TempUpdateBasisTemp-TempUpdateFinalTemp)/(n_epochs-1-100)
    new_t = tf.cond(epoch < 200,
                    lambda: (tf.subtract(tf.to_float(TempUpdateBasisTemp),tf.multiply(tf.to_float(epoch),TempUpdate)))*tempIncr,
                    lambda: 0.1 * tempIncr)
    return new_t

    # return (tf.subtract(tf.to_float(TempUpdateBasisTemp),tf.multiply(tf.to_float(epoch),TempUpdate)))*tempIncr


def temperature_update_numeric(tempIncr, epoch, n_epochs): 
    TempUpdateTempUpdate = (TempUpdateBasisTemp-TempUpdateFinalTemp)/(n_epochs-1)    
    return (TempUpdateBasisTemp - (epoch) * TempUpdateTempUpdate)*tempIncr
