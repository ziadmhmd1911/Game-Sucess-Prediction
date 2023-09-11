from sklearn.metrics import r2_score
import Models
from Testing import *
from Models import *
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
#############################Reading Data#########################################
data = pd.read_csv("Data/games-regression-dataset.csv")
#############################End Of reading data##################################
print(data.head(10))
shape = data.shape
print('\nDataFrame Shape :', shape)
print('\nNumber of rows :', shape[0])
print('\nNumber of columns :', shape[1])
print(data.info())
print(data.describe())
print(data.dtypes)
print(data.isnull().sum())
##################Some Columns Infos################################
print(data['Age Rating'].unique())
# calculate age value count
age_count = data['Age Rating'].value_counts()
print(age_count)
print(data['Average User Rating'].unique())
# calculate age value count
Avg_count = data['Average User Rating'].value_counts()
print(Avg_count)
#######################Visualization#############################
print(data.columns)
dataclean = DataCleansing()
data = dataclean.AutoPreProcess(data)
X = data.drop("Average User Rating", axis=1)
y = data["Average User Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = dataclean.PreProcess(X_train)
X_test = dataclean.PreProcess(X_test)
# Call the feature_selection function with the total number of features as k
selected_features = dataclean.feature_selection(X_train, y_train, k=10)
X_train = X_train[selected_features]
X_test = X_test[selected_features]
print(selected_features)
x = float(input("Enter 1 To train Or 2 To Test : "))
if x == 1:
    #############################################
    LRMODEL = Models.Regression(X_train, y_train, X_test, y_test)
    LRMODEL.train_and_save_model()

    RIDGEMODEL = Models.RidgeModel(X_train, y_train, X_test, y_test)
    RIDGEMODEL.train_and_save_model()

    LASSOMODEL = Models.LassoReg(X_train, y_train, X_test, y_test)
    LASSOMODEL.train_and_save_model()
    print("Finished Train")
################"TEST"######################
else:
    new_data = pd.read_csv("Data/ms1-games-tas-test-v2.csv")
    new_data.dropna(axis=0, how='any', subset=["Average User Rating"], inplace=True)
    y_test_new = new_data["Average User Rating"]
    new_data.drop(["Average User Rating"], axis=1, inplace=True)
    TestScript = Testing()
    new_data = TestScript.PreProcess(new_data)
    # for tCol in new_data.columns:
    #     if tCol not in X_train.columns:
    #         new_data.drop(columns=[tCol], inplace=True)
    # new_data = new_data.reindex(columns=X_train.columns)
    new_data = new_data[selected_features]
    linear_Model = pickle.load(open('Linear_Model.sav', 'rb'))
    y_pred_logreg = linear_Model.predict(new_data)
    r2 = r2_score(y_test_new, y_pred_logreg)
    mse = mean_squared_error(y_test_new, y_pred_logreg)
    #print(y_pred_logreg)
    print("R2 = ", r2)
    print("MSE = ", mse)