import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
from skimage.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
import _pickle as pickle
import matplotlib.pyplot as plt
import time

class Regression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_and_save_model(self):
        print("Linear Regression Model: ")
        # Define the hyperparameters to search over
        # Define the Linear Regression model
        # Define the grid search object
        # Fit the grid search object to the training data
        # Create a Linear Regression model with the best hyperparameters found
        Linear_Regression = LinearRegression()
        # Fit the model to the training data
        start_time1 = time.time()
        Linear_Regression.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1-start_time1
        print(f"Train time: {duration1:.2f} seconds")
        # Perform cross-validation to compute the mean squared error
        n_folds = 10
        scores = -cross_val_score(Linear_Regression, self.X_train, self.y_train, cv=n_folds,
                                  scoring='neg_mean_squared_error')
        mse_mean = scores.mean()
        print('Train Mean Squared Error:', mse_mean)
        # Make predictions on the test data
        start_time2 = time.time()
        y_pred_test = Linear_Regression.predict(self.X_test)
        end_time2 = time.time()
        duration2 = end_time2-start_time2
        print(f"Test time: {duration2:.2f} seconds")
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        print('Test Mean Squared Error:', mse_test)
        # Save the trained model to disk
        filename = 'Linear_Model.sav'
        pickle.dump(Linear_Regression, open(filename, 'wb'))
        data = pd.DataFrame({'Expected': self.y_test, 'Predicted': y_pred_test})
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(data['Expected'], data['Predicted'], c=data['Predicted'], cmap='cool')
        sns.regplot(x='Expected', y='Predicted', data=data, scatter=False, color='red', ax=ax)
        ax.set_title('Actual vs Predicted [Linear Regression]')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        plt.show()
        print('_____________________________________________')
class RidgeModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_and_save_model(self):
        print("Ridge Model: ")
        # Define the hyperparameters to search over
        # Define the Ridge model
        ridge = Ridge()
        # Define the grid search object
        # Fit the grid search object to the training data
        # Print the best hyperparameters found
        # Create a Ridge model with the best hyperparameters found
        ridge = Ridge(alpha=1, max_iter=10)
        # Fit the model to the training data
        start_time1 = time.time()
        ridge.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1-start_time1
        print(f"Train time: {duration1:.2f} seconds")
        # Make predictions on the test data
        start_time2 = time.time()
        pred_test = ridge.predict(self.X_test)
        end_time2 = time.time()
        duration2 = end_time2-start_time2
        print(f"Train time: {duration2:.2f} seconds")
        mse_test = mean_squared_error(self.y_test, pred_test)
        print('Test Mean Squared Error:', mse_test)
        # Make predictions on the train data
        pred_train = ridge.predict(self.X_train)
        mse_train = mean_squared_error(self.y_train, pred_train)
        print('Train Mean Squared Error:', mse_train)
        # Save the trained model to disk
        filename = 'Ridge_Model.sav'
        with open(filename, 'wb') as f:
            pickle.dump(ridge, f)
        data = pd.DataFrame({'Expected': self.y_test, 'Predicted': pred_test})

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(data['Expected'], data['Predicted'], c=data['Predicted'], cmap='cool')
        sns.regplot(x='Expected', y='Predicted', data=data, scatter=False, color='red', ax=ax)

        ax.set_title('Actual vs Predicted [Ridge Model]')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        plt.show()
        print('_____________________________________________')

class LassoReg:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_and_save_model(self):
        print("Lasso Reg Model: ")
        lasso_reg = Lasso(alpha=0.0001, random_state=42)
        start_time1 = time.time()
        lasso_reg.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1-start_time1
        print(f"Train time: {duration1:.2f} seconds")
        start_time2 = time.time()
        y_pred_test = lasso_reg.predict(self.X_test)
        end_time2 = time.time()
        duration2 = end_time2 - start_time2
        print(f"Test time: {duration2:.2f} seconds")
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        print('Test Mean Squared Error:', mse_test)
        y_pred_train = lasso_reg.predict(self.X_train)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        print('Train Mean Squared Error:', mse_train)
        filename = 'Lasso_Model.sav'
        with open(filename, 'wb') as f:
            pickle.dump(lasso_reg, f)
        data = pd.DataFrame({'Expected': self.y_test, 'Predicted': y_pred_test})
        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(data['Expected'], data['Predicted'], c=data['Predicted'], cmap='cool')
        sns.regplot(x='Expected', y='Predicted', data=data, scatter=False, color='red', ax=ax)
        ax.set_title('Actual vs Predicted [Lasso]')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.show()
        print('_____________________________________________')



