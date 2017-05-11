from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV



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
