'''
train_network.py
Updated: 06/29/18

This script is used to train keras nueral networks using a defined HDF5 file.
Best training validation accuracy will be saved.

'''
import os
import numpy as np
import h5py as hp
from tqdm import tqdm
from networks import *
from keras.utils import to_categorical as one_hot
import argparse
import sys

sys.path.insert(0, "capsnet")
import capsulelayers, capsulenet, utils

# Network Training Parameters
epochs = 20
model_def = CAPSNET
dims = 2
nb_chans = 8
# model_folder = ''

# Data Parameters
model_folder = data_folder = '../../data/KrasHras/'
dim_type = '-%dd' % dims # -1d, -2d, or -3d

################################################################################

seed = 1234

# Capsule Network Implementation
if __name__ == '__main__':

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # x and y training data

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load HDF5 dataset
    f = hp.File(data_folder+"dataset.hdf5", "r")


    '''TRAINING DATA'''

    # Shuffle train data
    train_set = f['train']
    classes = list(train_set.keys())

    x_train = []
    y_train = []
    for i in range(len(classes)):
        x = [name for name in train_set[classes[i]] if name.endswith(dim_type)]
        y = [i for j in range(len(x))]
        x_train += x
        y_train += y
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    train = np.concatenate([x_train,y_train],axis=-1)
    np.random.seed(seed)
    np.random.shuffle(train)

    # Load voxels data
    for i in range(len(train)):
        x_train = np.array(train_set[classes[int(train[i,1])]+'/'+train[i,0]])
        x_train = np.expand_dims(x_train, axis=0)
        y_train = one_hot(train[i,1], num_classes=len(classes))
        y_train = np.expand_dims(y_train, axis=0)

    '''VALIDATION DATA'''

    # Shuffle validation data
    val_set = f['val']
    classes = list(val_set.keys())

    x_val = []
    y_val = []
    for i in range(len(classes)):
        x = [name for name in val_set[classes[i]] if name.endswith(dim_type)]
        y = [i for j in range(len(x))]
        x_val += x
        y_val += y
    x_val = np.expand_dims(x_val, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    val = np.concatenate([x_val,y_val],axis=-1)
    np.random.seed(seed)
    np.random.shuffle(val)

    '''TESTING DATA'''

    # Parse test data
    test_set = f['test']
    classes = list(test_set.keys())

    x_test = []
    y_test = []
    for i in range(len(classes)):
        x = [name for name in test_set[classes[i]] if name.endswith(dim_type)]
        y = [i for j in range(len(x))]
        x_test += x
        y_test += y
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    test = np.concatenate([x_test,y_test],axis=-1)

    # Evaluate test data
    for i in range(len(test)):
        x_test = np.array(test_set[classes[int(test[i,1])]+'/'+test[i,0]])
        x_test = np.expand_dims(x_test, axis=0)
        y_test = one_hot(test[i,1], num_classes=len(classes))
        y_test = np.expand_dims(y_test, axis=0)

    model, eval_model, manipulate_model = model_def(input_shape = (512,  512, nb_chans), n_class = len(classes), routings = args.routings)


    print(x_train.shape, 'y train', y_train, x_test, y_test, x_val, y_val)

    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        capsulenet.train(model=model, data=((x_train, y_train), (x_test, y_test)), args = args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_val, y_val), args=args)