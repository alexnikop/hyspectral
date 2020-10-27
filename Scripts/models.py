from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Input, Reshape, Conv2D, Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPool2D
import tensorflow as tf

# core 3d2d architecture
def model_3d2d(spatial_size, spectral_components, output_units=14):

    if spectral_components == -1:
        spectral_components = 176

    kernel_initializer = 'he_normal'

    ## input layer
    input_layer = Input((spatial_size, spatial_size, spectral_components, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8,
                         kernel_size=(3, 3, 7),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(input_layer)

    conv_layer2 = Conv3D(filters=16,
                         kernel_size=(3, 3, 5),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_layer1)

    conv_layer3 = Conv3D(filters=32,
                         kernel_size=(3, 3, 3),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_layer2)

    conv3d_shape = conv_layer3.shape

    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2],
                           conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)

    conv_layer4 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256,
                         kernel_initializer=kernel_initializer,
                         activation='relu')(flatten_layer)

    dense_layer1 = Dropout(0.4)(dense_layer1)

    dense_layer2 = Dense(units=128,
                         kernel_initializer=kernel_initializer,
                         activation='relu')(dense_layer1)

    dense_layer2 = Dropout(0.4)(dense_layer2)

    output_layer = Dense(units=output_units,
                         activation='softmax')(dense_layer2)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    return model

# a variation of the core 3d2d architecture
def model_3d2d_light(spatial_size, spectral_components, output_units=14):

    if spectral_components == -1:
        spectral_components = 176

    kernel_initializer = 'he_normal'

    ## input layer
    input_layer = Input((spatial_size, spatial_size, spectral_components, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8,
                         kernel_size=(3, 3, 7),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(input_layer)

    conv_layer2 = Conv3D(filters=16,
                         kernel_size=(3, 3, 5),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_layer1)

    conv_layer3 = Conv3D(filters=32,
                         kernel_size=(3, 3, 3),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_layer2)

    conv3d_shape = conv_layer3.shape

    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2],
                           conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)

    conv_layer4 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    dense_layer2 = Dense(units=64,
                         kernel_initializer=kernel_initializer,
                         activation='relu')(flatten_layer)

    dense_layer2 = Dropout(0.4)(dense_layer2)

    output_layer = Dense(units=output_units,
                         activation='softmax')(dense_layer2)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    return model

# residual architecture
def model_3d2d_residual(spatial_size=15, spectral_components=16, output_units=14):
    if spectral_components == -1:
        spectral_components = 176

    kernel_initializer = 'he_normal'

    ## input layer
    input_layer = Input((spatial_size, spatial_size, spectral_components, 1))

    conv1_1 = Conv3D(filters=4,
                     kernel_size=(1, 1, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     activation='relu')(input_layer)

    conv1_2 = Conv3D(filters=4,
                     kernel_size=(3, 3, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     activation='relu')(input_layer)

    conv1_3 = Conv3D(filters=4,
                     kernel_size=(3, 5, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     activation='relu')(input_layer)

    concat_1 = tf.keras.layers.concatenate([conv1_1, conv1_2, conv1_3],
                                           axis=-1)


    conv_2_1_1 = Conv3D(filters=16,
                      kernel_size=(3, 3, 3),
                      kernel_initializer=kernel_initializer,
                      activation='relu')(concat_1)

    conv_2_1_2 = Conv3D(filters=32,
                      kernel_size=(3, 3, 3),
                      kernel_initializer=kernel_initializer,
                      activation='relu')(conv_2_1_1)
    
    conv_2_2 = Conv3D(filters=32,
                      kernel_size=(5, 5, 5),
                      kernel_initializer=kernel_initializer,
                      activation='relu')(concat_1)
    
    conv_2 = conv_2_1_2 + conv_2_2


    conv_3 = Conv3D(filters=40,
                      kernel_size=(1, 1, 3),
                      padding='same',
                      kernel_initializer=kernel_initializer,
                      activation='relu')(conv_2)


    conv_4_1_1 = Conv3D(filters=48,
                      kernel_size=(3, 3, 3),
                      kernel_initializer=kernel_initializer,
                      activation='relu')(conv_3)

    conv_4_1_2 = Conv3D(filters=64,
                      kernel_size=(3, 3, 3),
                      kernel_initializer=kernel_initializer,
                      activation='relu')(conv_4_1_1)
    
    conv_4_2 = Conv3D(filters=64,
                      kernel_size=(5, 5, 5),
                      kernel_initializer=kernel_initializer,
                      activation='relu')(conv_3)
    
    conv_4 = conv_4_1_2 + conv_4_2

    conv3d_shape = conv_4.shape

    conv_4_res = Reshape((conv3d_shape[1], conv3d_shape[2],
                           conv3d_shape[3] * conv3d_shape[4]))(conv_4)
    

    conv_5_1 = Conv2D(filters=128,
                         kernel_size=(4, 4),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_4_res)

    conv_5_2 = Conv2D(filters=512,
                         kernel_size=(4, 4),
                         kernel_initializer=kernel_initializer,
                         activation='relu')(conv_5_1)

    max_pool_layer = MaxPool2D((7, 7))(conv_4_res)
    
    flatten_layer = Flatten()(conv_5_2 + max_pool_layer)

    output_layer = Dense(units=output_units,
                         activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    return model