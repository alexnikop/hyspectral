from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Input, Reshape, Conv2D, Dense, Dropout, Flatten

def model_3d2d(spatial_size, spectral_components, output_units = 14):

    if spectral_components == -1:
        spectral_components = 176
    
    ## input layer
    input_layer = Input((spatial_size, spatial_size, spectral_components, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)

    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)

    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    return model

def model_3d2d_lighter(spatial_size, spectral_components, output_units = 14):

    if spectral_components == -1:
        spectral_components = 176
    
    ## input layer
    input_layer = Input((spatial_size, spatial_size, spectral_components, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)

    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    '''
    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    '''

    output_layer = Dense(units=output_units, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()
    
    return model

