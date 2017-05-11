import numpy as np
import pandas as pd
import math
import theano
from keras.datasets import mnist


#Miscellaneous
def make_csv(dictionary, name_order, path):
    if len(dictionary.keys()) != len(name_order):
        print('# of keys in dict != # names')
    else:
        df = pd.DataFrame(dictionary)
        df = df.ix[:,name_order]
        df.to_csv(path, index=False)

def load_and_condition_MNIST_data():
    ''' loads and shapes MNIST image data '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print "\nLoaded MNIST images"
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX) #before conversion were uint8
    X_test = X_test.astype(theano.config.floatX)
    X_train.resize(len(y_train), 784) # 28 pix x 28 pix = 784 pixels
    X_test.resize(len(y_test), 784)
    print '\nFirst 5 labels of MNIST y_train: ', y_train[:5]
    y_train_ohe = np_utils.to_categorical(y_train)
    print '\nFirst 5 labels of MNIST y_train (one-hot):\n', y_train_ohe[:5]
    print ''
    return X_train, y_train, X_test, y_test, y_train_ohe

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    print '\nRandom number generator seed: ', rng_seed
    print '\nFirst 30 labels:      ', y_train[:30]
    print 'First 30 predictions: ', y_train_pred[:30]
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print '\nTraining accuracy: %.2f%%' % (train_acc * 100)
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print 'Test accuracy: %.2f%%' % (test_acc * 100)
    if test_acc < 0.95:
        print '\nMan, your test accuracy is bad! '
        print "Can't you get it up to 95%?"
    else:
        print "\nYou've made some improvements, I see..."
