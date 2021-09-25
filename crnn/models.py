import numpy as np
from layers.stn import BilinearInterpolation
from tensorflow import keras
from tensorflow.keras import layers

def get_initial_weights(output_size):
    b = np.random.normal(0.0, 0.001, (2, 3))            # init weight zero won't trigger backpropagation
    b[0, 0] = 0.25
    b[1, 1] = 0.5
    W = np.random.normal(0.0, 0.01, (output_size, 6))  # init weight zero won't trigger backpropagation
    weights = [W, b.flatten()]
    return weights

def vgg_style(x):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    x = layers.Conv2D(
        64, 3, padding='same', activation='relu', name='conv1')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same', name='pool1')(x)

    x = layers.Conv2D(
        128, 3, padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same', name='pool2')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.Conv2D(
        256, 3, padding='same', activation='relu', name='conv4')(x)
    x = layers.MaxPool2D(
        pool_size=2, strides=(2, 1), padding='same', name='pool4')(x)

    x = layers.Conv2D(512, 3, padding='same', use_bias=False, name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('relu', name='relu5')(x)
    x = layers.Conv2D(
        512, 3, padding='same', activation='relu', name='conv6')(x)
    x = layers.MaxPool2D(
        pool_size=2, strides=(2, 1), padding='same', name='pool6')(x)

    x = layers.Conv2D(512, 2, use_bias=False, name='conv7')(x)
    x = layers.BatchNormalization(name='bn7')(x)
    x = layers.Activation('relu', name='relu7')(x)

    x = layers.Reshape((-1, 512), name='reshape7')(x)
    return x

def build_stn(img, interpolation_size):
    x = layers.Conv2D(32, (5, 5), padding='SAME')(img) # 20
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (5, 5), padding='SAME')(x)    #20
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # TODO change to global max pooling
    # TODO increasing channel number
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    transform_mat = layers.Dense(6, weights=get_initial_weights(32), name="stn")(x)
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
        model = keras.Model(inputs=img_input, outputs=x, name=model_name)

    stn_model = keras.Model(inputs=img_input, outputs=[interpolate_img, transform_mat])
    if weight:
        model.load_weights(weight, by_name=True, skip_mismatch=True)
    return model, stn_model