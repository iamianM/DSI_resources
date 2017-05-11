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

def run_RandomForestClassifier(X, y, X_test=None, n_estimators=10, oob_score=False, n_jobs=-1, max_depth=None):
    rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, oob_score=oob_score, n_jobs=n_jobs)
    rfc.fit(X, y)
    if X_test != None:
        return rfc, rfc.predict(X_test)
    else:
        return rfc

def run_lSVC(X, y, X_test=None):
    lSVC = LinearSVC()
    lSVC.fit(X, y)
    if X_test != None:
        return lSVC, lSVC.predict(X_test)
    else:
        return lSVC

def run_SVC(X, y, X_test=None, C=1.0, kernel='rbf', degree=3, scale=True):
    if scale:
        X_new = StandardScaler().fit_transform(X)
    svc = SVC(C=1.0, kernel=kernel, degree=degree)
    svc.fit(X_new)
    if X_test != None:
        return svc, svc.predict(X_test)
    else:
        return svc

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
