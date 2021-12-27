import numpy as np
import tensorflow_addons as tfa
from layers.stn import BilinearInterpolation
from tensorflow import keras
from tensorflow.keras import layers

relu6 = layers.ReLU(6.)

def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    x = layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = layers.BatchNormalization()(x)
    return relu6(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    tchannel = inputs.shape[-1] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = relu6(x)

    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if r:
        x = layers.add([x, inputs])
    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x
##################################################################################################################################

def separable_conv(x, p_filters, d_kernel_size=(3,3), d_strides=(1,1), d_padding='valid'):
    x = layers.DepthwiseConv2D(kernel_size=d_kernel_size, strides=d_strides, padding=d_padding, use_bias=True)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU(6)(x)
    x = layers.Conv2D(p_filters, kernel_size=(1,1), strides=(1,1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    return x

def get_initial_weights(output_size):
    b = np.random.normal(0.0, 0.001, (2, 2))            # init weight zero won't trigger backpropagation
    b[0, 0] = 0.25
    b[1, 1] = 0.5
    W = np.random.normal(0.0, 0.01, (output_size, 4))  # init weight zero won't trigger backpropagation
    weights = [W, b.flatten()]
    return weights

def vgg_style(x):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = separable_conv(x, p_filters=512, d_kernel_size=(3,3), d_strides=(1,1), d_padding='same')
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)
    x = separable_conv(x, p_filters=512, d_kernel_size=(3,3), d_strides=(1,1), d_padding='valid')

    x = layers.Reshape((-1, 512))(x)
    return x

def build_stn(img, interpolation_size):
    # x = layers.Conv2D(32, (5, 5), padding='SAME')(img) # 20
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU(6)(x)
    # x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(64, (5, 5), padding='SAME')(x)    #20
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU(6)(x)
    # x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=2)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU(6)(x)
    # # TODO change to global max pooling
    # # TODO increasing channel number
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(32)(x)
    # x = layers.ReLU(6)(x)
    # transform_mat = layers.Dense(6, weights=get_initial_weights(32), name="stn")(x)
    # interpolated_image = BilinearInterpolation(interpolation_size, name='bilinear_interpolation')([img, transform_mat])
    # return interpolated_image, transform_mat

    #x = layers.DepthwiseConv2D(kernel_size=d_kernel_size, strides=d_strides, padding=d_padding, use_bias=True)(x) 
    #x = layers.Conv2D(p_filters, kernel_size=(1,1), strides=(1,1), use_bias=False)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.ReLU(6)(x)    

    x = _conv_block(img ,32 , (5,5) , strides=(1,1))
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    #
    #x = layers.Conv2D(32, (5, 5), padding='SAME', use_bias=False)(img) # 20
    #x = layers.BatchNormalization()(x)
    #x = layers.ReLU(6)(x)
    #x = layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = _inverted_residual_block(x, 64 , (5,5) , t = 1 , strides=1 , n=1) 
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    #//DepthwiseConv2D(kernel_size=d_kernel_size, strides=d_strides, padding=d_padding, use_bias=True)(x)
    #x = layers.DepthwiseConv2D( (5,5), (1,1), padding='SAME', use_bias=False) (x)
    #x = layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), use_bias=False)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.ReLU(6)(x)
    #x = layers.MaxPool2D(pool_size=(2, 2))(x)
    #
    
    #x = layers.Conv2D(64, (5, 5), padding='SAME', use_bias=False)(x)    #20
    #x = layers.BatchNormalization()(x)
    #x = layers.ReLU(6)(x)
    #x = layers.MaxPool2D(pool_size=(2, 2))(x)
    
    #

    x = _inverted_residual_block(x, 128 , (3,3) , t = 6 , strides=2 , n=2) 
    #x = layers.DepthwiseConv2D( (3,3),(1,1), padding='SAME', use_bias=False) (x)
    #x = layers.Conv2D(128, kernel_size=(1,1), strides=(1,1), use_bias=False)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.ReLU(6)(x)

    #x = layers.Conv2D(128, (3, 3), padding='SAME', use_bias=False)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.ReLU(6)(x)

    x1 = _inverted_residual_block(x, 128 , (3,3) , t = 6 , strides=1 , n=1) 
    #x1 = layers.DepthwiseConv2D( (3,3), (1,1), padding='SAME', use_bias=False, dilation_rate=1) (x)
    #x1 = layers.Conv2D(128, kernel_size=(1,1), strides=(1,1), use_bias=False, dilation_rate=1)(x1)
    #x1 = layers.BatchNormalization()(x1)
    #x1 = layers.ReLU(6)(x1)

    #x1 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=1, use_bias=False)(x)
    #x1 = layers.BatchNormalization()(x1)
    #x1 = layers.ReLU(6)(x1)
    
    #
    x2 = _inverted_residual_block(x, 128 , (3,3) , t = 6 , strides=1 , n=1) 
    #x2 = layers.DepthwiseConv2D( (3,3), (1,1), padding='SAME', use_bias=False, dilation_rate=2) (x)
    #x2 = layers.Conv2D(128, kernel_size=(1,1), strides=(1,1), use_bias=False, dilation_rate=2)(x2)
    #x2 = layers.BatchNormalization()(x2)
    #x2 = layers.ReLU(6)(x2)

    #x2 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=2, use_bias=False)(x)
    #x2 = layers.BatchNormalization()(x2)
    #x2 = layers.ReLU(6)(x2)

    #
    x3 = _inverted_residual_block(x, 128 , (3,3) , t = 6 , strides=1 , n=1) 
    #x3 = layers.DepthwiseConv2D( (3,3), (1,1), padding='SAME', use_bias=False, dilation_rate=3) (x)
    #x3 = layers.Conv2D(128, kernel_size=(1,1), strides=(1,1), use_bias=False)(x3)
    #x3 = layers.BatchNormalization()(x3)
    #x3 = layers.ReLU(6)(x3)

    
    


    #x3 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=3, use_bias=False)(x)
    #x3 = layers.BatchNormalization()(x3)
    #x3 = layers.ReLU(6)(x3)

    #

    x = layers.Concatenate()([x1,x2,x3])
    x = layers.Conv2D(256, (1, 1), padding='SAME', use_bias=False)(x)
    x = layers.BatchNormalization()(x) #10x50


    # x = layers.ReLU(6)(x)
    # TODO change to global max pooling
    # TODO increasing channel number
    x = tfa.layers.SpatialPyramidPooling2D([[6,9],[4,6],[2,3]])(x) # 17408
    # x = tfa.layers.SpatialPyramidPooling2D([[6,24],[4,16],[2,8]])(x) # 17408

    x = layers.Flatten()(x)
    x = layers.Dense(32, use_bias=False)(x) # 32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    transform_mat = layers.Dense(4, weights=get_initial_weights(32), name="stn")(x)
    interpolated_image = BilinearInterpolation(interpolation_size, name='bilinear_interpolation')([img, transform_mat])
    return interpolated_image, transform_mat


def build_model(num_classes,
                require_coords=False,
                weight=None,
                preprocess=None,
                postprocess=None,
                img_shape=(32, None, 3),
                model_name='crnn'):

    x = img_input = keras.Input(shape=img_shape)
    if preprocess is not None:
        x = preprocess(x)
    
    interpolate_img, transform_mat = build_stn(x, (48, 48))
    
    x = vgg_style(interpolate_img)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True), name='bi_lstm1')(x)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True), name='bi_lstm2')(x)
    x = layers.Dense(units=num_classes, name='ctc_logits')(x)
    
    if postprocess is not None:
        x = postprocess(x)

    if require_coords:
        model = keras.Model(inputs=img_input, outputs=[x, transform_mat], name=model_name)
    else:
        model = keras.Model(inputs=img_input, outputs=x, name=model_name , name= "stn_model")

    stn_model = keras.Model(inputs=img_input, outputs=[interpolate_img, transform_mat])
    stn_model.summary()
    
    if weight:
        model.load_weights(weight, by_name=True, skip_mismatch=True)
        trainable=False
        for layer in model.layers:
            if layer.name in ['concatenate','conv2d_7','re_lu_11']:
                trainable = not trainable
            layer.trainable = trainable
            print(f'{layer.name} {layer.trainable}')
        
    return model, stn_model