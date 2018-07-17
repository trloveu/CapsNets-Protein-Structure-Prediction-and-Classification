import os
import numpy as np
from h5py import File
from tqdm import tqdm
from networks import CAPSNET
from keras.utils import to_categorical as one_hot
from keras.optimizers import Adam, Nadam, RMSprop
from keras.losses import logcosh, binary_crossentropy
from keras.activations import relu, elu, sigmoid

from argparse import ArgumentParser

import talos as ta

seed = 1234

if __name__ == '__main__':
    parser = ArgumentParser(description = "Capsule network on protein volumetric data.")
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--voxel_size', default = 512, type = int)
    parser.add_argument('--filters', default = 256, type = int)
    parser.add_argument('--kernel_size', default = 9, type = int)
    parser.add_argument('--lr', default = 0.001, type = float, help = "Initial learning rate.")
    parser.add_argument('--lr_decay', default = 0.9, type = float, help = "The value multiplied by lr at each epoch. Set a larger value for larger epochs.")
    parser.add_argument('--routings', default = 3, type = int, help = "Number of iterations used in routing algorithm. should > 0.")
    parser.add_argument('--nb_chans', default = 8, type = int, help = "Number of channels.")
    parser.add_argument('--result_dir', default = 'capsnet_results/', help = "Path where the results will be saved.")
    parser.add_argument('--data_folder', default = '../../data/KrasHras/', help = "Path where the data resides.")
    parser.add_argument('--dim_type', default = '-2d', help = "Data dimensionality.")
    parser.add_argument('--debug', default = 0, type = int)
    parser.add_argument('--train_acc_epsilon', default = 1e-6, type = float)
    args = parser.parse_args()

    if bool(args.debug) != 1:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load HDF5 dataset
    f = File(args.data_folder + "dataset.hdf5", "r")

    # Shuffle train data
    train_set = f['train']
    classes = list(train_set.keys())

    x_train = []
    y_train = []
    for i in range(len(classes)):
        x = [name for name in train_set[classes[i]] if name.endswith(args.dim_type)]
        y = [i for j in range(len(x))]
        x_train += x
        y_train += y
    x_train = np.expand_dims(x_train, axis = -1)
    y_train = np.expand_dims(y_train, axis = -1)
    train = np.concatenate([x_train,y_train], axis = -1)
    np.random.seed(seed)
    np.random.shuffle(train)

    # Load Model

    def init_model(model, x_train, y_train, x_val, y_val):
        print('Generating Model')
        model = CAPSNET(args, len(classes))
    
        model.summary()
        history = model.fit([x_train, y_train], [y_train, x_train], 
                        # validation_data=[x_val, y_val],
                        batch_size = 100,
                        epochs=50,
                        verbose=0)

        return history, model

    # Training Loop
    history = []
    best_val_loss = 0.0
    for epoch in range(args.epochs):
        print("Epoch %d" % (epoch + 1))

        # Fit training data
        print('Fit training data')
        train_status = []
        train_time = 0.0
        
        for i in tqdm(range(len(train))):
            x = np.array(train_set[classes[int(train[i, 1])] + '/' + train[i, 0]])
            x = np.expand_dims(x, axis = 0)
            y = one_hot(train[i, 1], num_classes = len(classes))
            y = np.expand_dims(y, axis = 0)

            t = ta.Scan(x=np.array([x, y]),
                        y=np.array([y, x]),
                        model=init_model,
                        grid_downsample=0.01, 
                        params= {'lr': (0.5, 5, 10),
                                 'first_neuron':[4, 8, 16, 32, 64],
                                 'hidden_layers':[0, 1, 2],
                                 'batch_size': (2, 30, 10),
                                 'epochs': [50],
                                 'dropout': (0, 0.5, 5),
                                 'weight_regulizer':[None],
                                 'emb_output_dims': [None],
                                 'shape':['brick','long_funnel'],
                                 'optimizer': [Adam, Nadam, RMSprop],
                                 'losses': [logcosh, binary_crossentropy],
                                 'activation':[relu, elu],
                                 'last_activation': [sigmoid]},
                        dataset_name='voxels',
                        experiment_no='1'
            )