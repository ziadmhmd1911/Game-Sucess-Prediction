import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
class RandomForest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.time1 = None
        self.time2 = None
        self.acc = None

    def train_and_save_model(self):
        print("Random Forest Without HyperParameter Tuning: ")
        # Create a Random Forest model
        rfm = RandomForestClassifier()

        # Fit the model to the training data
        start_time1 = time.time()
        rfm.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1 - start_time1
        print(f"Train Time Without HyperParameter Tuning: {duration1:.2f} seconds")

        # Make predictions on the training data
        y_pred_train = rfm.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print('Train Accuracy Without HyperParameter Tuning:', train_accuracy)

        # Make predictions on the testing data
        start_time2 = time.time()
        y_pred_test = rfm.predict(self.X_test)
        end_time2 = time.time()
        duration2 = end_time2 - start_time2
        print(f"Test Time Without HyperParameter Tuning: {duration2:.2f} seconds")
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy Without HyperParameter Tuning:', test_accuracy)
        print("End Of RandomForest Without HyperParameter Tuning")
        print("---------------------------------------------------")
        #######################################################################################
        print("Random Forest Model With HyperParameter Tuning: ")
        param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'min_samples_leaf': [1, 2, 3, 4, 5]}


        randomforestmodel = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=randomforestmodel, param_grid=param_grid, scoring='accuracy', cv=5,
                                   n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Print the best hyperparameters and the corresponding accuracy score
        print('Best hyperparameters:', grid_search.best_params_)
        print('Best accuracy:', grid_search.best_score_)

        # Create a Random Forest model with the best hyperparameters found
        best_max_depth = grid_search.best_params_['max_depth']
        best_min_samples_split = grid_search.best_params_['min_samples_split']
        best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']
        best_model = RandomForestClassifier(n_estimators=100, max_depth=best_max_depth,
                                            min_samples_split=best_min_samples_split,
                                            min_samples_leaf=best_min_samples_leaf)

        # Fit the model to the PCA-transformed training data
        start_time1 = time.time()
        best_model.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        self.time1 = end_time1 - start_time1
        print(f"Train time With HyperParameter Tuning: {self.time1:.2f} seconds")

        # Make predictions on the training data and calculate the accuracy score
        y_pred_train = best_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print('Train Accuracy With HyperParameter Tuning:', train_accuracy)

        # Make predictions on the testing data and calculate the accuracy score
        start_time2 = time.time()
        y_pred_test = best_model.predict(self.X_test)
        end_time2 = time.time()
        self.time2 = end_time2 - start_time2
        print(f"Test time With HyperParameter Tuning: {self.time2:.2f} seconds")
        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy With HyperParameter :', accuracy_test)
        self.acc = accuracy_test
        # Save the model to a file
        filename = 'RandomForest_PCA.sav'
        pickle.dump(best_model, open(filename, 'wb'))

        #######Visualize###########
        # Create the confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)

        # Plot the confusion matrix using Seaborn
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Confusion Matrix')
        plt.show()

class SVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.time1 = None
        self.time2 = None
        self.acc = None

    def train_and_save_model(self):
        print("SVM Without HyperParameter Tuning: ")
        elsvm = svm.SVC()
        #Fit The Model to the training data
        start_time1 = time.time()
        elsvm.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1 - start_time1
        print(f"Train Time Without HyperParameter Tuning: {duration1:.2f} seconds")
        #
        #Make Predictions on the training data
        y_pred_train = elsvm.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print("Train Accuracy Without HyperParameter Tuning:", train_accuracy)

        #Make Prediction on the testing data
        start_time2 = time.time()
        y_pred_test = elsvm.predict(self.X_test)
        end_time2 = time.time()
        duration2 = end_time2 - start_time2
        print(f"Test Time Without HyperParameter Tuning: {duration2:.2f} seconds")
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy Without HyperParameter Tuning:', test_accuracy)
        print("End Of SVM Without HyperParameter Tuning")
        print("---------------------------------------------------")
        print("SVM Model With HyperParameter Tuning: ")
        param_grid = {'C': [0.1, 1, 5, 10, 50, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['linear', 'rbf', 'poly']}
        svm_model = svm.SVC(random_state=42)
        grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1)

        # Fit the GridSearchCV object to the training data
        grid_search.fit(self.X_train, self.y_train)
        # Print the best hyperparameters and the corresponding accuracy score
        print('Best hyperparameters:', grid_search.best_params_)
        print('Best accuracy:', grid_search.best_score_)

        best_C = grid_search.best_params_['C']
        best_gamma = grid_search.best_params_['gamma']
        best_kernel = grid_search.best_params_['kernel']
        best_model = svm.SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)

        start_time1 = time.time()
        best_model.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        self.time1 = end_time1 - start_time1
        print(f"Train Time With HyperParameter Tuning: {self.time1:.2f} seconds")
        #
        # Make Predictions on the training data
        y_pred_train = best_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print("Train Accuracy With HyperParameter Tuning:", train_accuracy)

        # Make Prediction on the testing data
        start_time2 = time.time()
        y_pred_test = best_model.predict(self.X_test)
        end_time2 = time.time()
        self.time2 = end_time2 - start_time2
        print(f"Test Time With HyperParameter Tuning: {self.time2:.2f} seconds")
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy With HyperParameter Tuning:', test_accuracy)
        self.acc = test_accuracy

        filename = 'SVM_Model.sav'
        pickle.dump(best_model, open(filename, 'wb'))

        cm = confusion_matrix(self.y_test, y_pred_test)

        # Plot the confusion matrix using Seaborn
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Confusion Matrix')
        plt.show()

class LogisticRegressionTwo:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.time1 = None
        self.time2 = None
        self.acc = None

    def train_and_save_model(self):
        print("Logistic Regression Without HyperParameter Tuning")
        lr_model_without = LogisticRegression()
        #Fit the Model to the training data
        start_time1 = time.time()
        lr_model_without.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1 - start_time1
        print(f"Train Time Without HyperParameter Tuning: {duration1:.2f} seconds")
        #
        #Make Predictions on the training data
        y_pred_train = lr_model_without.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print("Train Accuracy Without HyperParameter Tuning:", train_accuracy)
        #
        # #Make Prediction on the testing data
        start_time2 = time.time()
        y_pred_test = lr_model_without.predict(self.X_test)
        end_time2 = time.time()
        duration2 = end_time2 - start_time2
        print(f"Test Time Without HyperParameter Tuning: {duration2:.2f} seconds")
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy Without HyperParameter Tuning:', test_accuracy)
        print("End Of Logistic Regression Without HyperParameter Tuning")
        print("---------------------------------------------------")

        print("Logistic Regression With HyperParameter Tuning")
        # Define the hyperparameters to tune
        hyperparameters = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100,1000,5000,10000],
            'max_iter': [1, 10, 50, 100, 500, 1000, 5000],
            'class_weight': [None, 'balanced'],
            'fit_intercept': [True, False]
        }

        # Create a logistic regression object
        lr_model_with = LogisticRegression()

        # Create a GridSearchCV object to find the besthyperparameters
        grid_search = GridSearchCV(lr_model_with, hyperparameters, cv=5)

        # Fit the GridSearchCV object to the training data
        start_time1 = time.time()
        grid_search.fit(self.X_train, self.y_train)
        end_time1 = time.time()
        duration1 = end_time1 - start_time1
        print(f"Train Time With HyperParameter Tuning: {duration1:.2f} seconds")

        # Get the best hyperparameters and create a new logistic regression object with those hyperparameters
        best_hyperparameters = grid_search.best_params_
        print('Best hyperparameters:', best_hyperparameters)
        lr_model_with = LogisticRegression(
            C=best_hyperparameters['C'],
            max_iter=best_hyperparameters['max_iter'],
            class_weight=best_hyperparameters['class_weight'],
            fit_intercept=best_hyperparameters['fit_intercept']
        )

        # Fit the new logistic regression object to the training data
        start_time2 = time.time()
        lr_model_with.fit(self.X_train, self.y_train)
        end_time2 = time.time()
        self.time1 = end_time2 - start_time2
        print(f"Train Time With Best HyperParameters: {self.time1:.2f} seconds")

        # Make predictions on the training data
        y_pred_train = lr_model_with.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print("Train Accuracy With HyperParameter Tuning:", train_accuracy)

        # Make predictions on the testing data
        start_time3 = time.time()
        y_pred_test = lr_model_with.predict(self.X_test)
        end_time3 = time.time()
        self.time2 = end_time3 - start_time3
        print(f"Test Time With Best HyperParameters: {self.time2:.2f} seconds")
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy With HyperParameter Tuning:', test_accuracy)
        self.acc = test_accuracy
        filename = 'Logistic_Model.sav'
        pickle.dump(lr_model_with, open(filename, 'wb'))
        print("End Of Logistic Regression With HyperParameter Tuning")
        print("---------------------------------------------------")

        cm = confusion_matrix(self.y_test, y_pred_test)

        # Plot the confusion matrix using Seaborn
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Confusion Matrix')
        plt.show()

class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.time1 = None
        self.time2 = None
        self.acc = None

    def train_and_savd_model(self):
        # Create a decision tree classifier object
        dt_model = DecisionTreeClassifier()
        # Fit the classifier to the training data
        dt_model.fit(self.X_train, self.y_train)
        # Make predictions on the training data
        y_pred_train = dt_model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print("Train Accuracy:", train_accuracy)
        # Make predictions on the testing data
        y_pred_test = dt_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy:', test_accuracy)
        print("End Of Decision Tree Classification")
        print("---------------------------------------------------")
        # Define the hyperparameters to tune
        hyperparameters = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 6, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'max_features': [None, 'sqrt', 'log2']
        }
        print("Decision tree with hyperparams tuning: ")
        # Create a decision tree classifier object
        dt_model_with = DecisionTreeClassifier()
        # Create a GridSearchCV object to find the best hyperparameters
        grid_search = GridSearchCV(dt_model_with, hyperparameters, cv=5)
        # Fit the GridSearchCV object to the training data
        grid_search.fit(self.X_train, self.y_train)
        # Get the best hyperparameters and create a new decision tree classifier# object with those hyperparameters
        best_hyperparameters = grid_search.best_params_
        print('Best hyperparameters:', best_hyperparameters)
        dt_model_with = DecisionTreeClassifier(
            criterion=best_hyperparameters['criterion'],
            max_depth=best_hyperparameters['max_depth'],
            min_samples_split=best_hyperparameters['min_samples_split'],
            min_samples_leaf=best_hyperparameters['min_samples_leaf'],
            max_features=best_hyperparameters['max_features']
        )
        # Fit the new decision tree classifier object to the training data
        startt1 = time.time()
        dt_model_with.fit(self.X_train, self.y_train)
        endt1 = time.time()
        self.time1 = endt1 - startt1
        # Make predictions on the training data
        startt1 = time.time()
        y_pred_train = dt_model_with.predict(self.X_train)
        endt1 = time.time()
        self.time2 = endt1 - startt1
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        print("Train Accuracy With HyperParameter Tuning:", train_accuracy)
        # Make predictions on the testing data
        y_pred_test = dt_model_with.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        print('Test Accuracy With HyperParameter Tuning:', test_accuracy)
        self.acc = test_accuracy
        print("End Of Decision Tree Classification With HyperParameter Tuning")
        filename = 'DecisionTree_Model.sav'
        pickle.dump(dt_model_with, open(filename, 'wb'))
        print("---------------------------------------------------")
        cm = confusion_matrix(self.y_test, y_pred_test)
        # Plot the confusion matrix using Seaborn
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Confusion Matrix')
        plt.show()