import os, sys
from parse_config                          import parse_config
from net_utils                             import get_activation, get_init, get_reg
try:
    from tensorflow.python.keras.models    import Model
    from tensorflow.python.keras.layers    import Dropout, Input
    from tensorflow.python.keras.layers    import BatchNormalization, Conv2D, MaxPooling2D
    from tensorflow.python.keras.layers    import Conv2DTranspose, Concatenate
    from tensorflow.python.keras.layers    import Activation, Add, SpatialDropout2D
    from tensorflow.python.keras.activations import linear 

except:
    from keras.models 				       import Model
    from keras.layers 				       import Dropout, Input
    from keras.layers 				       import BatchNormalization,Conv2D, MaxPooling2D
    from keras.layers 				       import Conv2DTranspose, Concatenate
    from keras.layers 				       import Activation, Add, SpatialDropout2D
    from tensorflow.kera.activations               import linear



def get_model(config, model_choice, weights=None):

    if model_choice == 1:
        print('Model: Chen_Unet')
        return Chen_Unet_model(config, weights=None)
    
    elif model_choice == 2:
        print('Model: AdaEn_2D')
        return AdaEn_2D_model(config, weights=None)

    elif model_choice == 3:
        print('Model: Isensee_2D')
        return Isensee_2D_model(config, weights=None)
    
    else:
        print('[WARNING]: Invalid Model choice')
        return None

def Chen_Unet_model(config, weights=None):

    config_train    = config['Train']
    crop_size       = config_train.get('crop_size')
    num_classes     = config_train.get('num_classes')
    input_shape     = (crop_size[0], crop_size[1], 1)
    
    config_network  = config['Network']
    bn_flag         = config_network.get('bn_flag')    
    acti_choice     = config_network.get('acti_choice')    
    use_bais        = config_network.get('use_bias')
    init_seed       = config_network.get('init_seed')
    kernel_init     = config_network.get('kernel_init')
    bias_init       = config_network.get('bias_init')
    kernel_reg      = config_network.get('kernel_reg')
    bias_reg        = config_network.get('bias_reg')    

    acti_func       = get_activation(config, acti_choice)
    k_init          = get_init(kernel_init, init_seed)
    b_init          = get_init(bias_init, init_seed)
    k_reg           = get_reg(config, kernel_reg)
    b_reg           = get_reg(config, bias_reg)

    # Defining convolution parameters
    k_size          = 3
    strd            = 1
    act             = None
    pad             = 'same'
    prob_dropOut    = 0.2

    # Binary or multiclass segmentation
    if num_classes == 2:
        acti 		  = 'sigmoid'
        n_class     = num_classes - 1
    elif num_classes > 2:
        acti        = 'softmax'
        n_class     = num_classes

    data            = Input(shape=input_shape, name='data')

    # Layer 1a
    level1          = Conv2D(filters=64,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(data)
    if bn_flag:
        level1      = BatchNormalization()(level1)

    level1          = Activation(acti_func)(level1)

    # Layer 1b
    level1          = Conv2D(filters=64,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level1)
    if bn_flag:
        level1      = BatchNormalization()(level1)

    level1          = Activation(acti_func)(level1)        

    # Layer 2a
    level2          = Conv2D(filters=128,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level1)
    if bn_flag:
        level2      = BatchNormalization()(level2)

    level2          = Activation(acti_func)(level2)

    # Layer 2b
    level2          = Conv2D(filters=128,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level2)
    if bn_flag:
        level2      = BatchNormalization()(level2)

    level2          = Activation(acti_func)(level2)    
    max_level2      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level2)

    # Layer 3a
    level3          = Conv2D(filters=256,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(max_level2)
    if bn_flag:
        level3      = BatchNormalization()(level3)

    level3          = Activation(acti_func)(level3)

    # Layer 3b
    level3          = Conv2D(filters=256,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level3)
    if bn_flag:
        level3      = BatchNormalization()(level3)

    level3          = Activation(acti_func)(level3)    
    max_level3      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level3)

    # Layer 4a
    level4          = Conv2D(filters=512,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(max_level3)
    if bn_flag:
        level4      = BatchNormalization()(level4)

    level4          = Activation(acti_func)(level4)

    # Layer 4b
    level4          = Conv2D(filters=512,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level4)
    if bn_flag:
        level4      = BatchNormalization()(level4)

    level4          = Activation(acti_func)(level4)    
    max_level4      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level4)

    # Layer 5a
    level5          = Conv2D(filters=512,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(max_level4)
    if bn_flag:
        level5      = BatchNormalization()(level5)

    level5          = Activation(acti_func)(level5)

    # Layer 5b
    level5          = Conv2D(filters=512,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level5)
    if bn_flag:
        level5      = BatchNormalization()(level5)

    level5          = Activation(acti_func)(level5)
    max_level5      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level5)
    

############  Upsample Side  ############    
    # Layer 6
    # 512 + 512
    level6          = Conv2DTranspose(filters=512, 
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(max_level5)

    level6          = Concatenate(axis=-1)([level6, max_level4])
    level6          = Dropout(prob_dropOut)(level6)

    level6          = Conv2D(filters=512,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level6)
    if bn_flag:
        level6      = BatchNormalization()(level6)
	
    level6          = Activation(acti_func)(level6)
    
    level6          = Conv2D(filters=512,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init, 
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level6)
    if bn_flag:
        level6      = BatchNormalization()(level6)

    level6          = Activation(acti_func)(level6)
    
    # Layer 7
	# 256 + 256
    level7          = Conv2DTranspose(filters=256, 
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level6)

    level7          = Concatenate(axis=-1)([level7, max_level3])
    level7          = Dropout(prob_dropOut)(level7)
    
    level7          = Conv2D(filters=256,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level7)
    if bn_flag:
        level7      = BatchNormalization()(level7)

    level7          = Activation(acti_func)(level7)
    
    level7          = Conv2D(filters=256,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level7)
    if bn_flag:
        level7      = BatchNormalization()(level7)

    level7          = Activation(acti_func)(level7)
        
    # Layer 8
    # 128 + 128
    level8          = Conv2DTranspose(filters=128, 
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level7)
    
    level8          = Concatenate(axis=-1)([level8, max_level2])
    level8          = Dropout(prob_dropOut)(level8)
    
    level8          = Conv2D(filters=128,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level8)
    if bn_flag:
        level8      = BatchNormalization()(level8)

    level8          = Activation(acti_func)(level8)
    
    level8          = Conv2D(filters=128,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level8)
    if bn_flag:
        level8      = BatchNormalization()(level8)

    level8          = Activation(acti_func)(level8)
    
    # Layer 9
    # 64 + 64
    level9          = Conv2DTranspose(filters=64,
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level8)

    level9          = Concatenate(axis=-1)([level9, level1])
    level9          = Dropout(prob_dropOut)(level9)
    
    level9          = Conv2D(filters=64,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level9)
    if bn_flag:
        level9      = BatchNormalization()(level9)

    level9          = Activation(acti_func)(level9)
    
    level9          = Conv2D(filters=64,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level9)
    if bn_flag:
        level9      = BatchNormalization()(level9)

    level9          = Activation(acti_func)(level9)
    
    # Layer 10
    # 64 + 64    
    level10         = Conv2D(filters=64,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level9)
    if bn_flag:
        level10     = BatchNormalization()(level10)

    level10         = Activation(acti_func)(level10)
    
    level10         = Conv2D(filters=64,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level10)
    if bn_flag:
        level10     = BatchNormalization()(level10)

    level10         = Activation(acti_func)(level10)    
    
    # Layer 11
    predictions     = Conv2D(filters=n_class,
                             kernel_size=1,
                             activation=acti)(level10)

    model           = Model(inputs=data, outputs=predictions)

    return model


def AdaEn_2D_model(config, weights=None):
	
    config_train    = config['Train']
    crop_size       = config_train.get('crop_size')
    num_classes     = config_train.get('num_classes')
    input_shape     = (crop_size[0], crop_size[1], 1)


    config_network  = config['Network']
    bn_flag         = config_network.get('bn_flag')
    acti_choice     = config_network.get('acti_choice')
    use_bais        = config_network.get('use_bias')
    kernel_init     = config_network.get('kernel_init')
    init_seed       = config_network.get('init_seed')
    bias_init       = config_network.get('bias_init')
    bias_reg        = config_network.get('bias_reg')
    kernel_reg      = config_network.get('kernel_reg')


    acti_func       = get_activation(config, acti_choice)
    k_init          = get_init(kernel_init,init_seed)
    b_init          = get_init(bias_init,init_seed)
    k_reg           = get_reg(config, kernel_reg)
    b_reg           = get_reg(config, bias_reg)

    # Defining convolution parameters
    k_size          = [3,1,7]			
    strides         = [1]
    filt            = [16,32,64,128,64,32,16]
    prob_dropOut    = 0
    pad             = 'same'
    act             = None

    # Binary or multiclass segmentation
    if num_classes == 2:
        acti        = 'sigmoid'
        n_class     = num_classes - 1
    else:
        acti        = 'softmax'
        n_class     = num_classes


    data                = Input(shape=input_shape, dtype='float32', name='data')
    
    # Residual Block 1 
    residual1           = Conv2D(filters=filt[0], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(data)
    if bn_flag:
        residual1       = BatchNormalization()(residual1)

    residual1           = Activation(acti_func)(residual1)
    residual1_relu1     = residual1
    
    residual1           = Conv2D(filters=filt[0], 
                                 kernel_size=k_size[1],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual1)
    if bn_flag:
        residual1       = BatchNormalization()(residual1)

    residual1           = Activation(acti_func)(residual1)
    residual1           = Conv2D(filters=filt[0], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual1)
    if bn_flag:
        residual1       = BatchNormalization()(residual1)

    residual1           = Activation(acti_func)(residual1)
    residual1           = Add()([residual1,residual1_relu1])
    max_pool1           = MaxPooling2D(pool_size=2, strides=2, padding='valid')(residual1)
    sp_drop1            = SpatialDropout2D(prob_dropOut)(max_pool1)

    # Residual Block 2
    residual2           = Conv2D(filters=filt[1], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(sp_drop1)
    if bn_flag:
        residual2       = BatchNormalization()(residual2)

    residual2           = Activation(acti_func)(residual2)
    residual2_relu1     = residual2
    residual2           = Conv2D(filters=filt[1], 
                                 kernel_size=k_size[1],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual2)
    if bn_flag:
        residual2       = BatchNormalization()(residual2)

    residual2           = Activation(acti_func)(residual2)
    residual2           = Conv2D(filters=filt[1], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual2)
    if bn_flag:
        residual2       = BatchNormalization()(residual2)

    residual2           = Activation(acti_func)(residual2)
    residual2           = Add()([residual2,residual2_relu1])
    max_pool2           = MaxPooling2D(pool_size=2, strides=2, padding='valid')(residual2)
    sp_drop2            = SpatialDropout2D(prob_dropOut)(max_pool2)

    # Residual Block 3 
    residual3           = Conv2D(filters=filt[2], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(sp_drop2)
    if bn_flag:
        residual3       = BatchNormalization()(residual3)

    residual3           =  Activation(acti_func)(residual3)
    residual3_relu1     =  residual3
    residual3           =  Conv2D(filters=filt[2], 
                                  kernel_size=k_size[1],
                                  strides=strides[0],
                                  padding=pad,
                                  activation=act,
                                  use_bias=use_bais,
                                  kernel_initializer=k_init,
                                  bias_initializer=b_init,
                                  kernel_regularizer=k_reg,
                                  bias_regularizer=b_reg)(residual3)
    if bn_flag:
        residual3       = BatchNormalization()(residual3)

    residual3           = Activation(acti_func)(residual3)
    residual3           = Conv2D(filters=filt[2], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual3)
	
    if bn_flag:
        residual3       = BatchNormalization()(residual3)

    residual3           = Activation(acti_func)(residual3)
    residual3           = Add()([residual3,residual3_relu1])
    max_pool3           = MaxPooling2D(pool_size=2, strides=2, padding='valid')(residual3)
    sp_drop3            = SpatialDropout2D(prob_dropOut)(max_pool3)
    
    # Residual Block 4
    residual4           = Conv2D(filters=filt[3], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(sp_drop3)
    if bn_flag:
        residual4       = BatchNormalization()(residual4)

    residual4           = Activation(acti_func)(residual4)
    residual4_relu1     = residual4
    residual4           = Conv2D(filters=filt[3], 
                                 kernel_size=k_size[1],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual4)
    if bn_flag:
        residual4       = BatchNormalization()(residual4)

    residual4           = Activation(acti_func)(residual4)
    residual4           = Conv2D(filters=filt[3], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual4)
    if bn_flag:
        residual4       = BatchNormalization()(residual4)

    residual4           = Activation(acti_func)(residual4)
    residual4           = Add()([residual4,residual4_relu1])

    # Transpose 1
    residual4           = Conv2DTranspose(filters=filt[4], kernel_size=2, strides=2, padding='Same')(residual4)
    
    # Concatenate 1    
    residual4           = Add()([residual4,residual3])
    
    # Drop Out
    sp_drop4            = SpatialDropout2D(prob_dropOut)(residual4)	
	
    # Residual Block 5
    residual5           = Conv2D(filters=filt[4], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(sp_drop4)
    if bn_flag:
        residual5       = BatchNormalization()(residual5)
	
    residual5           = Activation(acti_func)(residual5)
    residual5_relu1     = residual5
    residual5           = Conv2D(filters=filt[4], 
                                 kernel_size=k_size[1],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual5)
    if bn_flag:
        residual5       = BatchNormalization()(residual5)

    residual5           = Activation(acti_func)(residual5)
    residual5           = Conv2D(filters=filt[4], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual5)
    if bn_flag:
        residual5       = BatchNormalization()(residual5)

    residual5           = Activation(acti_func)(residual5)
    residual5           = Add()([residual5,residual5_relu1])
    
    # Transpose 2
    residual5           = Conv2DTranspose(filters=filt[5], kernel_size=2,strides=2,padding='Same')(residual5)

    # Concatenate 2   
    residual5           = Add()([residual5,residual2])

    # Drop Out
    sp_drop5            = SpatialDropout2D(prob_dropOut)(residual5)	
    
    # Residual Block 6
    residual6           = Conv2D(filters=filt[5], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(sp_drop5)
    if bn_flag:
        residual6       = BatchNormalization()(residual6)

    residual6           = Activation(acti_func)(residual6)
    residual6_relu1     = residual6
    residual6           = Conv2D(filters=filt[5], 
                                 kernel_size=k_size[1],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual6)
    if bn_flag:
        residual6       = BatchNormalization()(residual6)

    residual6           = Activation(acti_func)(residual6)
    residual6           = Conv2D(filters=filt[5], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual6)
    if bn_flag:
        residual6       = BatchNormalization()(residual6)

    residual6           = Activation(acti_func)(residual6)
    residual6           = Add()([residual6,residual6_relu1])

    # Transpose 3
    residual6           = Conv2DTranspose(filters=filt[6], kernel_size=2,strides=2,padding='Same')(residual6)

    # Concatenate 3
    residual6           = Add()([residual6,residual1])
	
    # Drop Out
    sp_drop6           = SpatialDropout2D(prob_dropOut)(residual6)	

    # Residual Block 7
    residual7           = Conv2D(filters=filt[6], 
                                 kernel_size=k_size[0],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(sp_drop6)
    if bn_flag:
        residual7       = BatchNormalization()(residual7)

    residual7           = Activation(acti_func)(residual7)
    residual7_relu1     = residual7
    residual7           = Conv2D(filters=filt[6], 
                                 kernel_size=k_size[1],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual7)
    if bn_flag:
        residual7       = BatchNormalization()(residual7)

    residual7           = Activation(acti_func)(residual7)
    residual7           = Conv2D(filters=filt[6], 
                                 kernel_size=k_size[2],
                                 strides=strides[0],
                                 padding=pad,
                                 activation=act,
                                 use_bias=use_bais,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)(residual7)
    if bn_flag:
        residual7       = BatchNormalization()(residual7)

    residual7           = Activation(acti_func)(residual7)
    residual7           = Add()([residual7,residual7_relu1])
	# Convolution + Softmax
    predictions         = Conv2D(filters=n_class, 
								 kernel_size=1, 
								 activation=acti)(residual7)

    model               = Model(inputs=data, outputs=predictions)

    return model
    

def Isensee_2D_model(config, weights=None):

    config_train    = config['Train']
    crop_size       = config_train.get('crop_size')
    num_classes     = config_train.get('num_classes')
    input_shape     = (crop_size[0], crop_size[1], 1)
    
    config_network  = config['Network']
    bn_flag         = config_network.get('bn_flag')    
    acti_choice     = config_network.get('acti_choice')    
    use_bais        = config_network.get('use_bias')
    init_seed       = config_network.get('init_seed')
    kernel_init     = config_network.get('kernel_init')
    bias_init       = config_network.get('bias_init')
    kernel_reg      = config_network.get('kernel_reg')
    bias_reg        = config_network.get('bias_reg')    

    acti_func       = get_activation(config, acti_choice)
    k_init          = get_init(kernel_init, init_seed)
    b_init          = get_init(bias_init, init_seed)
    k_reg           = get_reg(config, kernel_reg)
    b_reg           = get_reg(config, bias_reg)

    # Defining convolution parameters
    k_size          = 3
    strd            = 1
    act             = None
    pad             = 'same'
    prob_dropOut    = 0.0

    # Binary or multiclass segmentation
    if num_classes == 2:
        acti        = 'sigmoid'
        n_class     = num_classes - 1
    elif num_classes > 2:
        acti        = 'softmax'
        n_class     = num_classes

    data            = Input(shape=input_shape, name='data')

    # Layer 1a
    level1          = Conv2D(filters=48,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(data)
    if bn_flag:
        level1      = BatchNormalization()(level1)

    level1          = Activation(acti_func)(level1)

    # Layer 1b
    level1          = Conv2D(filters=48,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level1)
    if bn_flag:
        level1      = BatchNormalization()(level1)

    level1          = Activation(acti_func)(level1)  
    max_level1      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level1)      

    # Layer 2a
    level2          = Conv2D(filters=96,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(max_level1)
    if bn_flag:
        level2      = BatchNormalization()(level2)

    level2          = Activation(acti_func)(level2)

    # Layer 2b
    level2          = Conv2D(filters=96,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level2)
    if bn_flag:
        level2      = BatchNormalization()(level2)

    level2          = Activation(acti_func)(level2)    
    max_level2      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level2)
    
    dp_level2          = Dropout(prob_dropOut)(max_level2)


    # Layer 3a
    level3          = Conv2D(filters=192,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(dp_level2)
    if bn_flag:
        level3      = BatchNormalization()(level3)

    level3          = Activation(acti_func)(level3)

    # Layer 3b
    level3          = Conv2D(filters=192,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level3)
    if bn_flag:
        level3      = BatchNormalization()(level3)

    level3          = Activation(acti_func)(level3)    
    max_level3      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level3)
    dp_level3          = Dropout(prob_dropOut)(max_level3)


    # Layer 4a
    level4          = Conv2D(filters=384,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(dp_level3)
    if bn_flag:
        level4      = BatchNormalization()(level4)

    level4          = Activation(acti_func)(level4)

    # Layer 4b
    level4          = Conv2D(filters=384,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level4)
    if bn_flag:
        level4      = BatchNormalization()(level4)

    level4          = Activation(acti_func)(level4)    
    max_level4      = MaxPooling2D(pool_size=2, strides=2, padding='valid')(level4)
    
    dp_level4          = Dropout(prob_dropOut)(max_level4)


    # Layer 5a
    level5          = Conv2D(filters=768,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(dp_level4)
    if bn_flag:
        level5      = BatchNormalization()(level5)

    level5          = Activation(acti_func)(level5)

    # Layer 5b
    level5          = Conv2D(filters=768,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level5)
    if bn_flag:
        level5      = BatchNormalization()(level5)

    level5          = Activation(acti_func)(level5)
    

############  Upsample Side  ############    
    # Layer 6

    level6          = Conv2DTranspose(filters=384, 
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level5)

    level6          = Concatenate(axis=-1)([level6, level4])
    level6          = Dropout(prob_dropOut)(level6)

    level6          = Conv2D(filters=384,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level6)
    if bn_flag:
        level6      = BatchNormalization()(level6)
	
    level6          = Activation(acti_func)(level6)
    
    level6          = Conv2D(filters=384,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init, 
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level6)
    if bn_flag:
        level6      = BatchNormalization()(level6)

    level6          = Activation(acti_func)(level6)
    
    # Layer 7

    level7          = Conv2DTranspose(filters=192, 
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level6)

    level7          = Concatenate(axis=-1)([level7, level3])
    level7          = Dropout(prob_dropOut)(level7)
    
    level7          = Conv2D(filters=192,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level7)
    if bn_flag:
        level7      = BatchNormalization()(level7)

    level7          = Activation(acti_func)(level7)
    
    level7          = Conv2D(filters=192,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level7)
    if bn_flag:
        level7      = BatchNormalization()(level7)

    level7          = Activation(acti_func)(level7)
        
    # Layer 8
    # 128 + 128
    level8          = Conv2DTranspose(filters=96, 
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level7)
    
    level8          = Concatenate(axis=-1)([level8, level2])
    level8          = Dropout(prob_dropOut)(level8)
    
    level8          = Conv2D(filters=96,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level8)
    if bn_flag:
        level8      = BatchNormalization()(level8)

    level8          = Activation(acti_func)(level8)
    
    level8          = Conv2D(filters=96,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level8)
    if bn_flag:
        level8      = BatchNormalization()(level8)

    level8          = Activation(acti_func)(level8)
    
    # Layer 9
    # 64 + 64
    level9          = Conv2DTranspose(filters=48,
                                      kernel_size=3,
                                      strides=2,
                                      padding='Same')(level8)

    level9          = Concatenate(axis=-1)([level9, level1])
    
    level9          = Conv2D(filters=48,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,
                             use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level9)
    if bn_flag:
        level9      = BatchNormalization()(level9)

    level9          = Activation(acti_func)(level9)
    
    level9          = Conv2D(filters=48,
                             kernel_size=k_size,
                             strides=strd,
                             padding=pad,
                             activation=act,use_bias=use_bais,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=k_reg,
                             bias_regularizer=b_reg)(level9)
    if bn_flag:
        level9      = BatchNormalization()(level9)

    level9          = Activation(acti_func)(level9)
    
    #Addition Element-wise
    
    add_level7          = Conv2D(filters=n_class,
                             kernel_size=1,
                             strides=1,
                             padding=pad,
                             activation=None,
                             use_bias=False,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=None,
                             bias_regularizer=None)(level7)
    add_level7 = linear(add_level7)
    add_level7         = Conv2DTranspose(filters=n_class,
                                      kernel_size=1,
                                      strides=2,
                                      padding='Same')(add_level7)
    add_level8          = Conv2D(filters=n_class,
                             kernel_size=1,
                             strides=1,
                             padding=pad,
                             activation=None,
                             use_bias=False,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=None,
                             bias_regularizer=None)(level8)
    add_level8 = linear(add_level8)
    
    add1            = Add()([add_level7,add_level8])
    add1            = Conv2DTranspose(filters=n_class,
                                      kernel_size=1,
                                      strides=2,
                                      padding='Same')(add1)
    add_level9          = Conv2D(filters=n_class,
                             kernel_size=1,
                             strides=1,
                             padding=pad,
                             activation=None,
                             use_bias=False,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             kernel_regularizer=None,
                             bias_regularizer=None)(level9)
    add_level9 = linear(add_level9)

    addition        = Add()([add1,add_level9])

    
    # Final Prediction
    predictions     = Conv2D(filters=n_class,
                             kernel_size=1,
                             activation=acti)(addition)

    model           = Model(inputs=data, outputs=predictions)

    return model
    
    
if __name__=='__main__':
    
    config_file         = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    
    config              = parse_config(config_file)

    Isensee_2D_model(config, None)
    
###############################################################################
