import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from future.utils import iteritems
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.feature_extraction.text import TfidfVectorizer


# Machine Learning

# A bunch of these functions are super repeditive
# This is much more abstract and replaces a lot of these
def run_model(cls, X, y, X_test=None, scale=False, **kwargs):
    # All kwargs are all passed along to SVC object
    model_args = {'model__{}'.format(k): v for k, v in iteritems(kwargs)}

    if scale:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', cls())
        ])
    else:
        model = Pipeline([('model', cls())])

    # Pass all given args to the specified model
    model.set_params(**model_args)
    # Fit the model to the given training data
    model.fit(X, y)

    if X_test is not None:
        return model, model.predict(X_test)
    else:
        return model

def run_OLS(X, y):
    ols = sm.OLS(y, X)
    ols.fit()
    return ols

def run_KMeans(X, X_test=None, n_clusters=8, max_iter=300, n_jobs=-1, random_state=None):
    km = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_jobs=n_jobs, random_state=random_state)
    km.fit(X)
    if X_test is not None:
        return km, km.predict(X_test)

def create_dentrogram(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['content'])
    features = vectorizer.get_feature_names()

    # now get distances
    distxy = squareform(pdist(X.todense(), metric='cosine'))

    # 4. Pass this matrix into scipy's linkage function to compute our
    # hierarchical clusters.
    link = linkage(distxy, method='complete')

    # 5. Using scipy's dendrogram function plot the linkages as
    # a hierachical tree.

    labels = (df['headline'] + ' :: ' + df['section_name']).values
    dendro = dendrogram(link, color_threshold=1.5, leaf_font_size=9,
                        labels=labels)
    # fix spacing to better view dendrogram and the labels
    plt.subplots_adjust(top=.99, bottom=0.5, left=0.05, right=0.99)
    plt.show()


if __name__=='__main__':
    # get data
    data = np.genfromtxt('data/spam.csv', delimiter=',')
    y = data[:, -1]
    X = data[:, 0:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # example usage
    ols = run_OLS(X_train, y_train)
    gnb, y_predict_gnb = run_model(GaussianNB, X_train, y_train, X_test=X_test, priors=None)
    linr, y_predict_linr = run_model(LinearRegression, X_train, y_train, X_test=X_test)
    logr, y_predict_logr = run_model(LogisticRegression, X_train, y_train, X_test=X_test)
    rfc, y_predict_rfc = run_model(RandomForestClassifier, X_train, y_train, X_test=X_test)
    lSVCy_predict_lSVC = run_model(LinearSVC, X_train, y_train, X_test=X_test)
    svc, y_predict_svc = run_model(SVC, X_train, y_train, X_test=X_test, scale=True)
    knn, y_predict_knn = run_model(KNeighborsClassifier, X_train, y_train, X_test=X_test, n_neighbors=5, weights='uniform', metric='minkowski', n_jobs=-1)

    # now try KMeans and dentrogram
    iris = datasets.load_iris()
    X = iris.data
    X_train, X_test = train_test_split(X)
    km, y_predict_km = run_KMeans(X_train, X_test=X_test)

    articles_df = pd.read_pickle("data/articles.pkl")
    small_mask = np.zeros(len(articles_df)).astype(bool)
    indices = np.arange(len(articles_df))
    for category in articles_df['section_name'].unique():
        category_mask = (articles_df['section_name']==category).values
        new_index = np.random.choice(indices[category_mask])
        small_mask[new_index] = True
    additional_indices = np.random.choice(indices[np.logical_not(small_mask)],
                                          100 - sum(small_mask),
                                          replace=False)
    small_mask[additional_indices] = True
    small_df = articles_df.ix[small_mask]
    create_dentrogram(small_df)
