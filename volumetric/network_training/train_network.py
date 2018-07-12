import os
import numpy as np
from h5py import File
from tqdm import tqdm
from networks import CAPSNET
from keras.utils import to_categorical as one_hot
from argparse import ArgumentParser
from time import clock

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
    x_val = np.expand_dims(x_val, axis = -1)
    y_val = np.expand_dims(y_val, axis = -1)
    val = np.concatenate([x_val, y_val], axis = -1)
    np.random.seed(seed)
    np.random.shuffle(val)

    # Load Model
    print('Generating Model')
    model, eval_model = CAPSNET(args, len(classes))
    
    model.summary()

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
            
            train_start_time = clock()
            output = model.train_on_batch([x, y], [y, x])
            train_time += clock() - train_start_time
            
            train_status.append(output)

        # Calculate training loss and accuracy
        train_status = np.array(train_status)
        
        if args.debug == True:
            print(train_status)
        
        train_loss = np.average(train_status[:, 0])
        train_acc = np.average(train_status[:, 1])
        print('Train Loss ->', train_loss)
        print('Train Accu ->', train_acc)
        print('Train Time ->', train_time, '\n')

        # Test on validation data
        print('Testing on Validation Data')
        val_status = []
        val_time = 0.0
        for i in tqdm(range(len(val))):
            x = np.array(val_set[classes[int(val[i, 1])] + '/' + val[i, 0]])
            x = np.expand_dims(x, axis = 0)
            y = one_hot(val[i, 1], num_classes = len(classes))
            y = np.expand_dims(y, axis = 0)
            
            val_start_time = clock()
            output = model.test_on_batch([x, y], [y, x])
            val_time += clock() - val_start_time
            
            val_status.append(output)

        # Calculate validation loss and accuracy
        val_status = np.array(val_status)
        
        if args.debug == True:
            print(val_status)
        
        val_loss = np.average(val_status[:, 0])
        val_acc = np.average(val_status[:, 1])
        print('Val Loss ->', val_loss)
        print('Val Accu ->', val_acc)
        print('Val Time ->', val_time, '\n')

        if val_loss > best_val_loss:
            best_val_loss = val_loss

            # Save weights of model

            print('Saving Model Weights')
            model.save_weights(args.result_dir + CAPSNET.__name__ + '.hdf5')

        history.append([epoch, train_loss, train_acc, train_time, val_loss, val_acc, val_time])

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
    x_test = np.expand_dims(x_test, axis = -1)
    y_test = np.expand_dims(y_test, axis = -1)
    test = np.concatenate([x_test,y_test], axis = -1)

    # Load weights of best model
    model.load_weights(args.result_dir + CAPSNET.__name__ + '.hdf5')

    # Evaluate test data
    print('Evaluating Test Data')
    test_status = []
    test_time = 0.0

    for i in tqdm(range(len(test))):
        x = np.array(test_set[classes[int(test[i, 1])] + '/' + test[i, 0]])
        x = np.expand_dims(x, axis = 0)
        yy = int(test[i, 1])
        y = one_hot(test[i, 1], num_classes = len(classes))
        y = np.expand_dims(y, axis = 0)
        
        test_start_time = clock()
        output = eval_model.test_on_batch([x], [y, x])
        test_time += clock() - test_start_time
        
        test_status.append(output)

    # Calculate test loss and accuracy
    test_status = np.array(test_status)

    if args.debug:
        print(test_status)

    test_loss = np.average(test_status[:, 0])
    test_acc = np.average(test_status[:, 1])
    print('Test Loss ->', test_loss)
    print('Test Accu ->', test_acc)
    print('Test Time ->', test_time, '\n')

    # Save training history to csv file
    history = np.array(history)
    test_footer = 'Test [loss accu time], %f, %f, %f' % (test_loss, test_acc, test_time)
    
    np.savetxt(
        args.result_dir + CAPSNET.__name__ + '.csv',
        history,
        fmt = '%1.3f',
        delimiter = ', ',
        header = 'epoch, train_loss, train_acc, train_time, val_loss, val_acc, val_time',
        footer = test_footer
    )