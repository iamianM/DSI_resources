from future.utils import iteritems
import math

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from string import printable
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.decomposition import PCA

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano

# Machine Learning
def run_OLS(X, y, X_test=None):
    ols = sm.OLS(y_train, X_train)
    ols.fit()
    if X_test != None:
        return ols, ols.predict(X_test)
    else:
        return ols

def run_LinearRegression(X, y, X_test=None):
    lr = LinearRegression()
    lr.fit(X,y)
    if X_test != None:
        return lr, lr.predict(X_test)
    else:
        return lr

def run_LogisticRegression(X, y, X_test=None):
    lr = LogisticRegression()
    lr.fit(X, y)
    if X_test != None:
        return lr, lr.predict(X_test)
    else:
        return lr

def run_RandomForestClassifier(X, y, X_test=None, **kwargs):
    rfc = RandomForestClassifier(**kwargs)
    rfc.fit(X, y)
    if X_test is not None:
        return rfc, rfc.predict(X_test)
    else:
        return rfc

def run_lSVC(X, y, X_test=None, **kwargs):
    lSVC = LinearSVC(**kwargs)
    lSVC.fit(X, y)
    if X_test != None:
        return lSVC, lSVC.predict(X_test)
    else:
        return lSVC

def run_SVC(X, y, X_test=None, scale=True, **kwargs):
    # All kwargs are all passed along to SVC object
    model_args = {'model__{}'.format(k): v for k, v in iteritems(kwargs)}

    if scale:
        model = Pipeline([
            ('scaler', StandardScaler),
            ('model', SVC)
        ])
    else:
        model = Pipeline([('model', SVC)])

    model.set_params(**model_args)

    model.fit(X, y)
    if X_test is not None:
        return model, model.predict(X_test)
    else:
        return model

def run_KNeighborsClassifier(X, y, X_test=None, n_neighbors=5, weights='uniform', metric='minkowski', n_jobs=-1):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    if X_test != None:
        return knn, knn.predict(X_test)
    else:
        return knn

def run_KMeans(X, X_test=None, n_clusters=8, max_iter=300, n_jobs=-1, random_state=None):
    km = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_jobs=n_jobs, random_state=random_state)
    km.fit(X)
    if X_test != None:
        return km, km.predict(X_test)

def create_dentrogram(tf):
    '''
    Assumes its being given a tfidf matrix
    '''
    d = pdist(tf.todense())
    sf = squareform(d)
    Z = linkage(sf)
    return dendrogram(Z)

def run_GaussianNB(X, y, priors=None):
    gnb = GaussianNB(priors=priors)
    gnb.fit(X, y)
    if X_test != None:
        return gnb, gnb.predict(X_test)
    else:
        return gnb

# A bunch of these functions are super repeditive
# This is much more abstract and replaces a lot of these
def run_model(cls, X, y, X_test=None, scale=False, **kwargs):
    # All kwargs are all passed along to SVC object
    model_args = {'model__{}'.format(k): v for k, v in iteritems(kwargs)}

    if scale:
        model = Pipeline([
            ('scaler', StandardScaler),
            ('model', cls)
        ])
    else:
        model = Pipeline([('model', cls)])

    # Pass all given args to the specified model
    model.set_params(**model_args)
    # Fit the model to the given training data
    model.fit(X, y)

    if X_test is not None:
        return model, model.predict(X_test)
    else:
        return model

# example usage
# run_model(RandomForestRegressor, X, y, X_test, scale=False, n_estimators=10, n_jobs=-1)


# Boosting
def run_RandomForestRegressor(n_estimators=10,  criterion='mse', max_depth=None, oob_score=False, n_jobs=-1, random_state=None):
    rfr = RandomForestRegressor(n_estimators=n_estimators,  criterion=criterion, max_depth=max_depth, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state)
    # X and y weren't passed in
    rfr.fit(X, y)
    if X_test != None:
        return rfr, rfr.predict(X_test)
    else:
        return rfr

def run_GradientBoostingRegressor(learning_rate=0.1, loss='ls', n_estimators=100, random_state=None):
    gbr = GradientBoostingRegressor(learning_rate=learning_rate, loss=loss, n_estimators=n_estimators, random_state=random_state)
    gbr.fit(X, y)
    if X_test != None:
        return gbr, gbr.predict(X_test)
    else:
        return gbr

def run_AdaBoostRegressor(estimator, learning_rate=1.0, loss='linear', n_estimators=50, random_state=None):
    abr = AdaBoostRegressor(estimator, learning_rate=learning_rate, loss=loss, n_estimators=n_estimators, random_state=random_state)
    gnb.fit(X, y)
    if X_test != None:
        return gnb, gnb.predict(X_test)
    else:
        return gnb

def run_cross_val_score(estimator, X, y=None, scoring='neg_mean_squared_error', n_jobs=-1):
    return cross_val_score(estimator, X, y=y, scoring=scoring, n_jobs=n_jobs)

def run_GridSearchCV(estimator, param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True):
    rf_gridsearch = GridSearchCV(estimator, param_grid, n_jobs=n_jobs, verbose=verbose, scoring=scoring)

def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array
    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    estimator.fit(X_train,y_train)
    y_tp = np.array(list(estimator.staged_predict(X_train)))
    y_predict = np.array(list(estimator.staged_predict(X_test)))
    mse_train = []
    mse_test = []
    for i in range(y_predict.shape[0]):
        mse_train.append(mean_squared_error(y_train,list(y_tp[i,:])))
        mse_test.append(mean_squared_error(y_test,list(y_predict[i,:])))
    plt.plot(mse_train,'--',label=estimator.__class__.__name__+' Train - learning rate '+str(estimator.learning_rate))
    plt.plot(mse_test,label=estimator.__class__.__name__+' Test - learning rate '+str(estimator.learning_rate))

# Text preprocessing
def run_tokenizer(content):
    '''
    Takes nx1 array where column is strings.
    Outputs list of strings
    '''
    stopwords = set(stopwords.words('english'))
    wordnet = WordNetLemmatizer()
    toks = []
    for i, c in enumerate(content):
        if len(c) != 0:
            c = ''.join([l for l in c if l in printable])
            wt = word_tokenize(c)
            c = [w for w in wt if w.lower() not in stopwords]
            lemmatized = [wordnet.lemmatize(i) for i in c]
            toks.append(' '.join(lemmatized))
    return toks

def run_TfidfVectorizer(toks):
    tfidfV = TfidfVectorizer()
    tf = tfidfV.fit_transform(toks)
    return tf, tfidfV

def run_BeautifulSoup(html_doc):
    '''
    html_doc should be from html_str.text (where html_str may be from requests.get(html))
    '''
    return BeautifulSoup(html_doc, 'html.parser')

def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print ('WARNING'), response.status_code
        return 0
    else:
        return response.json()

#PCA
def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])

    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

def plot_embedding(X, y, title=None):
    '''
    INPUT:
    X - decomposed feature matrix
    y - target labels (digits)
    Creates a pyplot object showing digits projected onto 2-dimensional
    feature space. PCA should be performed on the feature matrix before
    passing it to plot_embedding.
    '''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,1.1])

    if title is not None:
        plt.title(title, fontsize=16)

def run_PCA(df, n_components=None):
    pca = PCA(n_components=components).fit(df)
    return pca, pca.transform(df)


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
    print("\nLoaded MNIST images")
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX) #before conversion were uint8
    X_test = X_test.astype(theano.config.floatX)
    X_train.resize(len(y_train), 784) # 28 pix x 28 pix = 784 pixels
    X_test.resize(len(y_test), 784)
    print('\nFirst 5 labels of MNIST y_train: ', y_train[:5])
    y_train_ohe = np_utils.to_categorical(y_train)
    print('\nFirst 5 labels of MNIST y_train (one-hot):\n', y_train_ohe[:5])
    print('')
    return X_train, y_train, X_test, y_test, y_train_ohe

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    print('\nRandom number generator seed: ', rng_seed)
    print('\nFirst 30 labels:      ', y_train[:30])
    print('First 30 predictions: ', y_train_pred[:30])
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('\nTraining accuracy: %.2f%%' % (train_acc * 100))
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    if test_acc < 0.95:
        print('\nMan, your test accuracy is bad! ')
        print("Can't you get it up to 95%?")
    else:
        print("\nYou've made some improvements, I see...")
