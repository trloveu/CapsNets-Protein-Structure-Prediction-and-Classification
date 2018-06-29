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
sys.path.insert(0, "capsnet")

import capsulenet


# Network Training Parameters
epochs = 10
model_def = CAPSNET
dims = 3
nb_chans = 2 ** dims
# model_folder = ''

# Data Parameters
model_folder = data_folder = '../../data/KrasHras/'
dim_type = '-%dd' % dims # -1d, -2d, or -3d

################################################################################

seed = 1234

# if __name__ == '__main__':

#     # Set paths relative to this file
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))

#     # Load HDF5 dataset
#     f = hp.File(data_folder+"dataset.hdf5", "r")

#     # Shuffle train data
#     train_set = f['train']
#     classes = list(train_set.keys())

#     x_train = []
#     y_train = []
#     for i in range(len(classes)):
#         x = [name for name in train_set[classes[i]] if name.endswith(dim_type)]
#         y = [i for j in range(len(x))]
#         x_train += x
#         y_train += y
#     x_train = np.expand_dims(x_train, axis=-1)
#     y_train = np.expand_dims(y_train, axis=-1)
#     train = np.concatenate([x_train,y_train],axis=-1)
#     np.random.seed(seed)
#     np.random.shuffle(train)

#     # Shuffle validation data
#     val_set = f['val']
#     classes = list(val_set.keys())

#     x_val = []
#     y_val = []
#     for i in range(len(classes)):
#         x = [name for name in val_set[classes[i]] if name.endswith(dim_type)]
#         y = [i for j in range(len(x))]
#         x_val += x
#         y_val += y
#     x_val = np.expand_dims(x_val, axis=-1)
#     y_val = np.expand_dims(y_val, axis=-1)
#     val = np.concatenate([x_val,y_val],axis=-1)
#     np.random.seed(seed)
#     np.random.shuffle(val)

#     # Load Model
#     model, loss, optimizer, metrics = model_def(len(classes), x_train, y_train, 3)
#     model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#     model.summary()

#     # Training Loop
#     history = []
#     best_val_loss = 0.0
#     for epoch in range(epochs):
#         print("Epoch", epoch, ':')

#         # Fit training data
#         print('Fitting:')
#         train_status = []
#         for i in tqdm(range(len(train))):
#             x = np.array(train_set[classes[int(train[i,1])]+'/'+train[i,0]])
#             x = np.expand_dims(x, axis=0)
#             y = one_hot(train[i,1], num_classes=len(classes))
#             y = np.expand_dims(y, axis=0)
#             output = model.train_on_batch(x, y)
#             train_status.append(output)

#         # Calculate training loss and accuracy
#         train_status = np.array(train_status)
#         train_loss = np.average(train_status[:,0])
#         train_acc = np.average(train_status[:,1])
#         print('Train Loss ->',train_loss)
#         print('Train Accuracy ->',train_acc,'\n')

#         # Test on validation data
#         print('Evaluating:')
#         val_status = []
#         for i in tqdm(range(len(val))):
#             x = np.array(val_set[classes[int(val[i,1])]+'/'+val[i,0]])
#             x = np.expand_dims(x, axis=0)
#             y = one_hot(val[i,1], num_classes=len(classes))
#             y = np.expand_dims(y, axis=0)
#             output = model.test_on_batch(x, y)
#             val_status.append(output)

#         # Calculate validation loss and accuracy
#         val_status = np.array(val_status)
#         val_loss = np.average(val_status[:,0])
#         val_acc = np.average(val_status[:,1])
#         print('Val Loss ->',val_loss)
#         print('Val Accuracy ->',val_acc,'\n')

#         if val_loss > best_val_loss:
#             best_val_loss = val_loss

#             # Save weights of model
#             model.save_weights(data_folder+model_def.__name__+'.hdf5')

#         history.append([epoch, train_loss, train_acc, val_loss, val_acc])

#     # Parse test data
#     test_set = f['test']
#     classes = list(test_set.keys())

#     x_test = []
#     y_test = []
#     for i in range(len(classes)):
#         x = [name for name in test_set[classes[i]] if name.endswith(dim_type)]
#         y = [i for j in range(len(x))]
#         x_test += x
#         y_test += y
#     x_test = np.expand_dims(x_test, axis=-1)
#     y_test = np.expand_dims(y_test, axis=-1)
#     test = np.concatenate([x_test,y_test],axis=-1)

#     # Load weights of best model
#     model.load_weights(model_folder+model_def.__name__+'.hdf5')

#     # Evaluate test data
#     print('Evaluating Test:')
#     test_status = []
#     for i in tqdm(range(len(test))):
#         x = np.array(test_set[classes[int(test[i,1])]+'/'+test[i,0]])
#         x = np.expand_dims(x, axis=0)
#         yy = int(test[i,1])
#         y = one_hot(test[i,1], num_classes=len(classes))
#         y = np.expand_dims(y, axis=0)
#         output = model.test_on_batch(x, y)
#         test_status.append(output)

#     # Calculate test loss and accuracy
#     test_status = np.array(test_status)
#     test_loss = np.average(test_status[:,0])
#     test_acc = np.average(test_status[:,1])
#     print('Test Loss ->',test_loss)
#     print('Test Accuracy ->',test_acc,'\n')

#     # Save training history to csv file
#     history = np.array(history)
#     test_footer = 'Test [loss, acc]: ' + str(test_loss) + ', ' + str(test_acc)
#     np.savetxt(model_folder+model_def.__name__+'.csv', history, fmt= '%1.3f',
#                 delimiter=', ', header='LABELS: epoch, loss, acc, val_loss, val_acc',
#                 footer=test_footer)

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

    # Shuffle train data
    train_set = f['train']
    val_set = f['val']
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

    model, eval_model, manipulate_model = model_def(input_shape = x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))), routings = args.routings)

    model.summary()

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

    # print("x and y training", x_train.shape, y_train.shape, val.shape)

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