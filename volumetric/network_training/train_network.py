'''
train_network.py
Updated: 12/27/17

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

# Network Training Parameters
model_def = CAPSNET

seed = 1234

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Capsule network on protein volumetric data.")
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--voxel_size', default = 512, type = int)
    parser.add_argument('--filters', default = 256, type = int)
    parser.add_argument('--kernel_size', default = 9, type = int)
    parser.add_argument('--lr', default = 0.001, type = float, help = "Initial learning rate.")
    parser.add_argument('--lr_decay', default = 0.9, type = float, help = "The value multiplied by lr at each epoch. Set a larger value for larger epochs.")
    parser.add_argument('--routings', default = 3, type = int, help = "Number of iterations used in routing algorithm. should > 0.")
    parser.add_argument('--nb_chans', default = 8, type = int, help = "Number of channels.")
    parser.add_argument('--result_dir', default = 'capsnet_results/')
    parser.add_argument('--data_folder', default = '../../data/KrasHras/')
    parser.add_argument('--dim_type', default = '-2d')
    parser.add_argument('--debug', default = False, type = bool)
    args = parser.parse_args()

    epochs = args.epochs
    result_folder = args.result_dir
    nb_chans = args.nb_chans

    if not args.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load HDF5 dataset
    f = hp.File(args.data_folder + "dataset.hdf5", "r")

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
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    train = np.concatenate([x_train,y_train],axis=-1)
    np.random.seed(seed)
    np.random.shuffle(train)

    # Shuffle validation data
    val_set = f['val']
    classes = list(val_set.keys())

    x_val = []
    y_val = []
    for i in range(len(classes)):
        x = [name for name in val_set[classes[i]] if name.endswith(args.dim_type)]
        y = [i for j in range(len(x))]
        x_val += x
        y_val += y
    x_val = np.expand_dims(x_val, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    val = np.concatenate([x_val,y_val],axis=-1)
    np.random.seed(seed)
    np.random.shuffle(val)

    # Load Model
    model = model_def(args, len(classes))
    
    model.summary()

    # Training Loop
    history = []
    best_val_loss = 0.0
    for epoch in range(epochs):
        print("Epoch %d:" % (epoch + 1))

        # Fit training data
        print('Fitting:')
        train_status = []
        for i in tqdm(range(len(train))):
            x = np.array(train_set[classes[int(train[i,1])]+'/'+train[i,0]])
            x = np.expand_dims(x, axis=0)
            y = one_hot(train[i,1], num_classes=len(classes))
            y = np.expand_dims(y, axis=0)
            output = model.train_on_batch(x, y)
            train_status.append(output)

        # Calculate training loss and accuracy
        train_status = np.array(train_status)
        if not args.debug:
            print(train_status)
        train_loss = np.average(train_status[:,0])
        train_acc = np.average(train_status[:,1])
        print('Train Loss ->',train_loss)
        print('Train Accuracy ->',train_acc,'\n')

        # Test on validation data
        print('Evaluating:')
        val_status = []
        for i in tqdm(range(len(val))):
            x = np.array(val_set[classes[int(val[i,1])]+'/'+val[i,0]])
            x = np.expand_dims(x, axis=0)
            y = one_hot(val[i,1], num_classes=len(classes))
            y = np.expand_dims(y, axis=0)
            output = model.test_on_batch(x, y)
            val_status.append(output)

        # Calculate validation loss and accuracy
        val_status = np.array(val_status)
        if not args.debug:
            print(val_status)
        val_loss = np.average(val_status[:,0])
        val_acc = np.average(val_status[:,1])
        print('Val Loss ->',val_loss)
        print('Val Accuracy ->',val_acc,'\n')

        if val_loss > best_val_loss:
            best_val_loss = val_loss

            # Save weights of model
            model.save_weights(result_folder + model_def.__name__+'.hdf5')

        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

    # Parse test data
    test_set = f['test']
    classes = list(test_set.keys())

    x_test = []
    y_test = []
    for i in range(len(classes)):
        x = [name for name in test_set[classes[i]] if name.endswith(args.dim_type)]
        y = [i for j in range(len(x))]
        x_test += x
        y_test += y
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    test = np.concatenate([x_test,y_test],axis=-1)

    # Load weights of best model
    model.load_weights(result_folder + model_def.__name__ + '.hdf5')

    # Evaluate test data
    print('Evaluating Test:')
    test_status = []
    for i in tqdm(range(len(test))):
        x = np.array(test_set[classes[int(test[i,1])]+'/'+test[i,0]])
        x = np.expand_dims(x, axis=0)
        yy = int(test[i,1])
        y = one_hot(test[i,1], num_classes=len(classes))
        y = np.expand_dims(y, axis=0)
        output = model.test_on_batch(x, y)
        test_status.append(output)

    # Calculate test loss and accuracy
    test_status = np.array(test_status)
    if not args.debug:
        print(test_status)
    test_loss = np.average(test_status[:,0])
    test_acc = np.average(test_status[:,1])
    print('Test Loss ->', test_loss)
    print('Test Accuracy ->', test_acc, '\n')

    # Save training history to csv file
    history = np.array(history)
    test_footer = 'Test [loss, acc]: ' + str(test_loss) + ', ' + str(test_acc)
    np.savetxt(result_folder+model_def.__name__+'.csv', history, fmt= '%1.3f',
                delimiter=', ', header='LABELS: epoch, loss, acc, val_loss, val_acc',
                footer=test_footer)