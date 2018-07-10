'''
networks.py
Updated: 12/29/17

README:

'''

# For Neural Network
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Add, Dense, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense, Concatenate
from keras.layers import Conv3D, AveragePooling2D, Activation, MaxPooling3D
from keras.layers import Reshape
from keras.optimizers import SGD, Adam, Adamax, Adadelta, RMSprop
from keras.constraints import maxnorm

import sys
sys.path.insert(0, "capsnet")

import capsulelayers
import numpy as np

################################################################################

def D1NET_v1(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 242,114
    Parameters w/ 1 Chans: 228,810

    '''
    # Input Layer
    x = Input(shape=(262144, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=64, strides=9, padding='valid', activation='relu')(x)
    l = MaxPooling1D(9)(l)
    l = Conv1D(filters=32, kernel_size=64, strides=9, padding='valid', activation='relu')(l)
    l = MaxPooling1D(9)(l)
    l = Flatten()(l)

    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics


def D1NET_v2(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 310,978
    Parameters w/ 1 Chans: 284,906

    '''
    # Input Layer
    x = Input(shape=(262144, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=121, strides=9, padding='valid', activation='relu')(x)
    l = MaxPooling1D(9)(l)
    l = Conv1D(filters=32, kernel_size=121, strides=9, padding='valid', activation='relu')(l)
    l = MaxPooling1D(9)(l)
    l = Flatten()(l)

    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D1NET_v3(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 440,002
    Parameters w/ 1 Chans: 390,634

    '''
    # Input Layer
    x = Input(shape=(262144, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=225, strides=9, padding='valid', activation='relu')(x)
    l = MaxPooling1D(9)(l)
    l = Conv1D(filters=32, kernel_size=225, strides=9, padding='valid', activation='relu')(l)
    l = MaxPooling1D(9)(l)
    l = Flatten()(l)

    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET_v1(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 184,770
    Parameters w/ 1 Chans: 171,466

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (8, 8), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (8, 8), strides = (3,3), padding='valid', activation='relu')(l)
    l = MaxPooling2D((3,3))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET_v2(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 257,730
    Parameters w/ 1 Chans: 231,658

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (11, 11), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (11, 11), strides = (3,3), padding='valid', activation='relu')(l)
    l = MaxPooling2D((3,3))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET_v3(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 353,986
    Parameters w/ 1 Chans: 304,618

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (15, 15), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (15, 15), strides = (3,3), padding='valid', activation='relu')(l)
    l = MaxPooling2D((3,3))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET_v4(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:
    Parameters w/ 1 Chans:

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (8, 8), strides = (2,2), padding='valid', activation='relu')(x)
    l = MaxPooling2D((2,2))(l)
    l = Conv2D(32, (8, 8), strides = (2,2), padding='valid', activation='relu')(l)
    l = MaxPooling2D((2,2))(l)
    l = Conv2D(32, (8, 8), strides = (2,2), padding='valid', activation='relu')(l)
    l = MaxPooling2D((2,2))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET_v5(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:
    Parameters w/ 1 Chans:

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (8, 8), strides = (2,2), padding='valid', activation='relu')(x)
    l = MaxPooling2D((2,2))(l)
    l = Conv2D(32, (8, 8), strides = (1,1), padding='valid', activation='relu')(l)
    l = MaxPooling2D((2,2))(l)
    l = Conv2D(32, (8, 8), strides = (1,1), padding='valid', activation='relu')(l)
    l = MaxPooling2D((2,2))(l)
    l = Conv2D(32, (8, 8), strides = (1,1), padding='valid', activation='relu')(l)
    l = MaxPooling2D((2,2))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def RESNET_v1(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans::

    '''
    from keras.layers import Add, Activation
    x = Input(shape=(512,  512, nb_chans))
    ll = Conv2D(32, (1, 1), strides = (1,1), padding='same')(x)
    l0 = Conv2D(32, (8, 8), strides = (1,1), padding='same', activation='relu')(ll)
    l1 = Conv2D(32, (8, 8), strides = (1,1), padding='same')(l0)
    l2 = Add()([ll, l1])
    l3 = Activation('relu')(l2)
    l4 = MaxPooling2D((9,9))(l3)
    l5 = Conv2D(32, (8 ,8), strides = (1,1), padding='same', activation='relu')(l4)
    l6 = Conv2D(32, (8, 8), strides = (1,1), padding='same')(l5)
    l7 = Add()([l4,l6])
    l8 = Activation('relu')(l7)
    l9 = MaxPooling2D((9,9))(l8)

    # Fully Connected Layer
    l = Flatten()(l9)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D3NET_v1(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 192,962
    Parameters w/ 1 Chans: 179,658

    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D3NET_v2(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 271,042
    Parameters w/ 1 Chans: 244,074

    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D3NET_v3(nb_chans, nb_class):
    '''
    Parameters w/ 8 Chans:: 309,698
    Parameters w/ 1 Chans: 262,346

    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (6, 6, 6), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (6, 6, 6), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NETREG_v1(nb_chans):
    '''
    '''
    from keras.losses import mean_squared_error

    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (8, 8), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (8, 8), strides = (3,3), padding='valid', activation='relu')(l)
    l = MaxPooling2D((3,3))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(1, activation='sigmoid')(l)

    model = Model(inputs=x, outputs=y)
    loss = mean_squared_error
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = ['accuracy',]

    return model, loss, optimizer, metrics

def CAPSNET(nb_chans, nb_class):
    ######## network setup start ########

    from keras.losses import mean_squared_error

    input_shape = (512,  512, nb_chans)

    x = Input(shape = input_shape)

    # Layer 1: Conv2D layer
    conv1 = Conv2D(filters = 256, kernel_size = 9, strides = 1, padding = 'valid', activation = 'relu', name = 'conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = capsulelayers.PrimaryCap(conv1, dim_capsule = 8, n_channels = nb_chans, kernel_size = 9, strides = 2, padding = 'valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    voxelscap = capsulelayers.CapsuleLayer(num_capsule = nb_class, dim_capsule = 16, routings = 3, name = 'voxelscap')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_caps = capsulelayers.Length(name='capsnet')(voxelscap)

    # Decoder network.
    y = Input(shape = (nb_class, ))
    masked_by_y = capsulelayers.Mask()([voxelscap, y])
    masked = capsulelayers.Mask()(voxelscap)

    # Shared Decoder model in training and prediction
    decoder = Sequential(name = 'decoder')
    decoder.add(Dense(512, activation = 'relu', input_dim = 16 * nb_class))
    decoder.add(Dense(1024, activation = 'relu'))
    decoder.add(Dense(np.prod(input_shape), activation = 'sigmoid'))
    decoder.add(Reshape(target_shape = input_shape, name = 'out_recon'))

    ########  network setup end  ########

    # Models for training and evaluation (prediction)
    model = Model(x, out_caps)
    #eval_model = Model(x, [out_caps, decoder(masked)])

    loss = mean_squared_error

    optimizer = Adam(lr = 1e-3, decay = 9e-1)

    metrics = ['accuracy',]

    return model, loss, optimizer, metrics