import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from future.utils import iteritems
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston



# Boosting

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

def run_cross_val_score(estimator, X, y=None, scoring='neg_mean_squared_error', n_jobs=-1):
    return cross_val_score(estimator, X, y=y, scoring=scoring, n_jobs=n_jobs)

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in iteritems(parameter_grid):
        print("{0:<20s} | {1:<8s} | {2}".format(str(param),
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

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

def load_and_split_data():
    ''' Loads sklearn's boston dataset and splits it into train:test datasets
        in a ratio of 80:20. Also sets the random_state for reproducible
        results each time model is run.

        Parameters: None

        Returns:  (X_train, X_test, y_train, y_test):  tuple of numpy arrays
                  column_names: numpy array containing the feature names
    '''
    boston = load_boston() #load sklearn's dataset
    X, y = boston.data, boston.target
    column_names = boston.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                       test_size = 0.2,
                                       random_state = 1)
    return (X_train, X_test, y_train, y_test), column_names

if __name__=='__main__':
    # get data
    data = np.genfromtxt('data/spam.csv', delimiter=',')
    y = data[:, -1]
    X = data[:, 0:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    (X_train, X_test, y_train, y_test), column_names = load_and_split_data()

    # example usage
    rfr, y_predict_rfr = run_model(RandomForestRegressor, X_train, y_train, X_test=X_test, scale=False, n_estimators=10,  criterion='mse', max_depth=None, oob_score=False, n_jobs=-1, random_state=None)
    gbr, y_predict_gbr = run_model(GradientBoostingRegressor, X_train, y_train, X_test=X_test, scale=False, learning_rate=0.1, loss='ls', n_estimators=100, random_state=None)
    adr, y_predict_adr = run_model(AdaBoostRegressor, X_train, y_train, X_test=X_test, scale=False, base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=None)

    score_rfr = run_cross_val_score(rfr, X_train, y=y_train, scoring='neg_mean_squared_error', n_jobs=-1)
    score_gbr = run_cross_val_score(gbr, X_train, y=y_train, scoring='neg_mean_squared_error', n_jobs=-1)
    score_adr = run_cross_val_score(adr, X_train, y=y_train, scoring='neg_mean_squared_error', n_jobs=-1)

    random_forest_grid = {'max_depth': [3, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}

    gradient_boosting_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
                              'max_depth': [2, 4, 6],
                              'min_samples_leaf': [1, 2, 5, 10],
                              'max_features': [1.0, 0.3, 0.1],
                              'n_estimators': [500],
                              'random_state': [1]}

    rfr_best_params, rfr_best_model = gridsearch_with_output(RandomForestRegressor(), random_forest_grid, X_train, y_train)
    gbr_best_params, gbr_best_model = gridsearch_with_output(GradientBoostingRegressor(), gradient_boosting_grid, X_train, y_train)

    stage_score_plot(GradientBoostingRegressor(), X_train, y_train, X_test, y_test)
    plt.show()
    stage_score_plot(AdaBoostRegressor(), X_train, y_train, X_test, y_test)
    plt.show()
