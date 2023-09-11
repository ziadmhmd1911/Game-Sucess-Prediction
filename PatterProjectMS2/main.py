import matplotlib.pyplot as plt

import Models
from Testing import *
from Models import *
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from DataCleansing import *
#############################Reading Data#########################################
data = pd.read_csv("Data/games-classification-dataset.csv")
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

print(data['Rate'].unique())

# calculate age value count
Rate_count = data['Rate'].value_counts()
print(Rate_count)
dataclean = DataCleansing()
X = data.drop("Rate", axis=1)
y = data["Rate"]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = dataclean.PreProcess(X_train)
X_test = dataclean.PreProcess(X_test)
selected_features = dataclean.feature_selection(X_train, y_train, k=10)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

inq = float(input("Enter 1 To train Or 2 To Test : "))
if inq == 1:
#############################################
    # RANDOMFORESTMODEL = Models.RandomForest(X_train, y_train, X_test, y_test)
    # RANDOMFORESTMODEL.train_and_save_model()

    # SVMMODEL = Models.SVM(X_train, y_train, X_test, y_test)
    # SVMMODEL.train_and_save_model()
    # #
    # LOGISTICREGRSSIONMODEL = Models.LogisticRegressionTwo(X_train, y_train, X_test, y_test)
    # LOGISTICREGRSSIONMODEL.train_and_save_model()
    # #
    # DESICIONTREEMODEL = Models.DecisionTree(X_train, y_train, X_test, y_test)
    # DESICIONTREEMODEL.train_and_savd_model()
    print("Finished Train Successfully")

    plt.figure(figsize=(8, 6))
    models = ['Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree']
    time_durations = [RANDOMFORESTMODEL.time1, SVMMODEL.time1, LOGISTICREGRSSIONMODEL.time1,
                      DESICIONTREEMODEL.time1]  # Time durations in seconds
    bars = plt.bar(models, time_durations, color='skyblue')

    # Add time durations to the top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{time_durations[i]:.2f}".format(time_durations[i]), ha='center',
                 va='bottom')
    plt.xlabel('Machine Learning Models')
    plt.ylabel('Time Duration (seconds)')
    plt.title('Comparison of Time Durations of Fitting for Machine Learning Models')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.show()
    #########################################################################################################
    plt.figure(figsize=(8, 6))
    models = ['Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree']
    time_durations = [RANDOMFORESTMODEL.time2, SVMMODEL.time2, LOGISTICREGRSSIONMODEL.time2,
                      DESICIONTREEMODEL.time2]  # Time durations in seconds
    bars = plt.bar(models, time_durations, color='skyblue')

    # Add time durations to the top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{time_durations[i]:.2f}".format(time_durations[i]),
                 ha='center',
                 va='bottom')
    plt.xlabel('Machine Learning Models')
    plt.ylabel('Time Duration (seconds)')
    plt.title('Comparison of Time Durations of Fitting for Machine Learning Models')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.show()
    #########################################################################################################
    plt.figure(figsize=(8, 6))
    models = ['Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree']
    time_durations = [RANDOMFORESTMODEL.acc, SVMMODEL.acc, LOGISTICREGRSSIONMODEL.acc,
                      DESICIONTREEMODEL.acc]  # Time durations in seconds
    bars = plt.bar(models, time_durations, color='skyblue')

    # Add time durations to the top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{time_durations[i]:.2f}".format(time_durations[i]), ha='center',
                 va='bottom')
    plt.xlabel('Machine Learning Models')
    plt.ylabel('Time Duration (seconds)')
    plt.title('Comparison of Time Durations of Prdication for Machine Learning Models')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.show()
    #####################################################################################################################
else:
    new_data = pd.read_csv("Data/ms2-games-tas-test-v2.csv")
    new_data.dropna(axis=0, how='any', subset=["Rate"], inplace=True)
    y_test_new = new_data["Rate"]
    new_data.drop(["Rate"], axis=1, inplace=True)
    y_test_new = le.fit_transform(y_test_new)
    TestScript = Testing()
    new_data = TestScript.PreProcess(new_data)
    for tCol in new_data.columns:
        if tCol not in X_train.columns:
            new_data.drop(columns=[tCol], inplace=True)
    new_data = new_data.reindex(columns=X_train.columns)
    RandomForestModel = pickle.load(open('SVM_Model.sav', 'rb'))

    y_pred_rfmodel = RandomForestModel.predict(new_data)
    accuracy = accuracy_score(y_test_new, y_pred_rfmodel)
    print("Test set accuracy: {:.2f}".format(accuracy))
    print(len(y_pred_rfmodel))
    print("Finished Test Successfully")