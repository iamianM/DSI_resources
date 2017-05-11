



def run_cnn():
    from __future__ import print_function
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K

    batch_size = 128
    nb_classes = 10
    nb_epoch = 12

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def run_rnn():
    import numpy as np

    #def nonlin(x,deriv=False):
    #	if(deriv==True):
    #	    return x*(1-x)
    #	return 1/(1+np.exp(-x))

    def nonlin(x,deriv=False):
    	if(deriv==True):
    	    return nonlin(x)*(1-nonlin(x))
    	return 1/(1+np.exp(-x))

    X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])

    y = np.array([[0],
    			[1],
    			[1],
    			[0]])

    np.random.seed(1)

    # add learning rate
    alpha = 1

    # randomly initialize our weights with mean 0
    syn0 = 2*np.random.random((3,4)) - 1
    syn1 = 2*np.random.random((4,1)) - 1

    for j in xrange(60000):

    	# Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))

        # how much did we miss the target value?
        l2_error = y - l2

        if (j% 10000) == 0:
            print "Error:" + str(np.mean(np.abs(l2_error)))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)

        syn1 += alpha*l1.T.dot(l2_delta)
        syn0 += alpha*l0.T.dot(l1_delta)

    if __name__ == '__main__':
        print "y_t\ty_p"
        for yt, yp in zip(y,l2):
            print yt, yp
