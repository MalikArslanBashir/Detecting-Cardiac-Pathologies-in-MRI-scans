try:
    import tensorflow.keras.backend          as K
    from tensorflow.keras                    import losses
    from tensorflow.keras.utils              import to_categorical
except:
    import keras.backend                     as K
    from keras                               import losses
    from keras.utils                         import to_categorical


_REDUCTION_DIMS = (0, 1, 2)
_EPSILON        = 1e-7

def get_loss_function(loss_choice):
    """Interface to return loss function in main file"""
    if loss_choice==1:
        return xentropy_tf
    elif loss_choice==2:
        return dice_loss_tf
    elif loss_choice==3:
        return dice_plus_xentropy_tf
    else:
        print('[ERROR] Could not recognize loss function selection:', loss_choice)

#---------------------------- TensorFlow FUNCTIONS ----------------------------#
###------ Cost Functions

def xentropy_tf(y_true, y_pred):
    # returns mean of pixel-wise cross-entropy
    return K.mean(losses.categorical_crossentropy(y_true, y_pred))

def dice_loss_tf(y_true, y_pred):
    # similar to soft_dice_score_tf()
    # returns mean of class-wise dice
    numerator       = 2.0 * K.sum(K.cast(y_true, dtype='float32') * y_pred, axis=_REDUCTION_DIMS)
    denominator     = K.sum(K.square(K.cast(y_true, dtype='float32')) + K.square(y_pred), axis=_REDUCTION_DIMS)
    return 1.0 - K.mean((numerator+_EPSILON)/(denominator+_EPSILON))

def dice_plus_xentropy_tf(y_true, y_pred):
    # non-weighted sum
    return dice_loss_tf(y_true, y_pred) + xentropy_tf(y_true, y_pred)

###--------------------------------------------------------------------------###

###--------------------------------------------------------------------------###

 
###------ Metrics
def soft_dice_score_tf(y_true, y_pred):
    # similar to soft_dice_score_tf(), not in use
    numerator       = 2.0 * K.sum(K.cast(y_true, dtype='float32') * y_pred, axis=_REDUCTION_DIMS)
    denominator     = K.sum(K.square(K.cast(y_true, dtype='float32')) + K.square(y_pred), axis=_REDUCTION_DIMS)
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))

def dice_score_tf(y_true, y_pred):
    """Binary Dice Metric"""
    y_pred          = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    numerator       = 2.0 * K.sum(y_true * y_pred, axis=_REDUCTION_DIMS)
    denominator     = K.sum(y_true + y_pred, axis=_REDUCTION_DIMS)
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))

def iou_score_tf(y_true, y_pred):
    """Binary IoU Metric"""
    y_pred          = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    numerator       = K.sum(y_true * y_pred, axis=_REDUCTION_DIMS)
    denominator     = K.sum(y_true + y_pred, axis=_REDUCTION_DIMS) - numerator
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))
'''
def dice_LV_Myo_tf(y_true, y_pred):
    """MultiClass Dice Metric for LV"""
    y_true_binary   = K.cast(K.equal(K.argmax(y_true, axis=-1), 1), dtype='float32')
    y_pred_binary   = K.cast(K.equal(K.argmax(y_pred, axis=-1), 1), dtype='float32')
    numerator       = 2.0 * K.sum(y_true_binary * y_pred_binary, axis=_REDUCTION_DIMS)
    denominator     = K.sum(y_true_binary + y_pred_binary, axis=_REDUCTION_DIMS)
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))
'''
def dice_LV_tf(y_true, y_pred):
    """MultiClass Dice Metric for LV"""
    y_true_binary   = K.cast(K.equal(K.argmax(y_true, axis=-1), 1), dtype='float32')
    y_pred_binary   = K.cast(K.equal(K.argmax(y_pred, axis=-1), 1), dtype='float32')
    numerator       = 2.0 * K.sum(y_true_binary * y_pred_binary, axis=_REDUCTION_DIMS)
    denominator     = K.sum(y_true_binary + y_pred_binary, axis=_REDUCTION_DIMS)
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))

def dice_RV_tf(y_true, y_pred):
    """MultiClass Dice Metric for RV"""
    y_true_binary   = K.cast(K.equal(K.argmax(y_true, axis=-1), 3), dtype='float32')
    y_pred_binary   = K.cast(K.equal(K.argmax(y_pred, axis=-1), 3), dtype='float32')
    numerator       = 2.0 * K.sum(y_true_binary * y_pred_binary, axis=_REDUCTION_DIMS)
    denominator     = K.sum(y_true_binary + y_pred_binary, axis=_REDUCTION_DIMS)
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))

def dice_Myo_tf(y_true, y_pred):
    """MultiClass Dice Metric for Myo"""
    y_true_binary   = K.cast(K.equal(K.argmax(y_true, axis=-1), 2), dtype='float32')
    y_pred_binary   = K.cast(K.equal(K.argmax(y_pred, axis=-1), 2), dtype='float32')
    numerator       = 2.0 * K.sum(y_true_binary * y_pred_binary, axis=_REDUCTION_DIMS)
    denominator     = K.sum(y_true_binary + y_pred_binary, axis=_REDUCTION_DIMS)
    return K.mean((numerator+_EPSILON)/(denominator+_EPSILON))

###############################################################################
