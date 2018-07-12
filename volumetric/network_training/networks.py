from keras.models import Model
from keras.layers import Conv2D, Input
from keras.optimizers import Adam

import sys
sys.path.insert(0, "capsnet")
import capsulelayers
import numpy as np

def CAPSNET(args, nb_class):
    ######## network setup start ########
    from keras.losses import mean_squared_error

    input_shape = (args.voxel_size,  args.voxel_size, args.nb_chans)

    x = Input(shape = input_shape)

    # Layer 1: Conv2D layer
    conv1 = Conv2D(filters = args.filters, kernel_size = args.kernel_size, strides = 1, padding = 'valid', activation = 'relu', name = 'conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = capsulelayers.PrimaryCap(conv1, dim_capsule = 8, n_channels = args.nb_chans, kernel_size = args.kernel_size, strides = 2, padding = 'valid')

    # Layer 3: Capsule layer. Routing algorithm runs here.
    voxelscap = capsulelayers.CapsuleLayer(num_capsule = nb_class, dim_capsule = 16, routings = args.routings, name = 'voxelscap')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_caps = capsulelayers.Length(name = 'capsnet')(voxelscap)

    # Decoder network.
    # y = Input(shape = (nb_class, ))
    # masked_by_y = capsulelayers.Mask()([voxelscap, y])
    # masked = capsulelayers.Mask()(voxelscap)

    # Shared Decoder model in training and prediction
    # decoder = Sequential(name = 'decoder')
    # decoder.add(Dense(512, activation = 'relu', input_dim = 16 * nb_class))
    # decoder.add(Dense(1024, activation = 'relu'))
    # decoder.add(Dense(np.prod(input_shape), activation = 'sigmoid'))
    # decoder.add(Reshape(target_shape = input_shape, name = 'out_recon'))
    ########  network setup end  ########

    # Models for training and evaluation (prediction)
    model = Model(x, out_caps)
    #eval_model = Model(x, [out_caps, decoder(masked)])

    loss = mean_squared_error

    optimizer = Adam(lr = args.lr, decay = args.lr_decay)

    metrics = ['accuracy',]

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model