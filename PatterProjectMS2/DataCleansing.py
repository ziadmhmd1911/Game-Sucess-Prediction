import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression

class DataCleansing:
    def __init__(self):
        pass

    def remove_outliers(self,df, column, threshold=3):
        # Calculate z-score for the column
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        # Identify rows with z-score greater than the threshold
        outliers = df[z_scores > threshold]
        # Remove outliers from the DataFrame
        df_clean = df.drop(outliers.index)
        return df_clean

    def removenulls(self,dataframe):
        dataframe.drop(["URL", "ID", "Name", "Subtitle", "Icon URL", "Description"], axis=1, inplace=True)
        # at_least_count = len(dataframe)/3
        # dataframe.dropna(axis=1, thresh=at_least_count, inplace=True)
        dataframe['Languages'] = dataframe['Languages'].fillna('EN')
        dataframe['In-app Purchases'] = dataframe['In-app Purchases'].fillna(0, axis=0)
        return dataframe

    def LogNormaliztion(self,dataframe):
        dataframe['Size'] = np.log(dataframe['Size'])
        dataframe['User Rating Count'] = np.log(dataframe['User Rating Count'])
        return dataframe

    def EncodingForDates(self,dataframe):
        dataframe["Original Release Date"] = pd.to_datetime(dataframe["Original Release Date"]).dt.year
        dataframe["Current Version Release Date"] = pd.to_datetime(dataframe["Current Version Release Date"]).dt.year
        return dataframe

    def WorkOnColumns(self,data):
        data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x:1 if x!= 0 else 0)
        data['Diff Between Years'] = data['Current Version Release Date'] - data['Original Release Date']
        # dataframe.drop(['Original Release Date','Current Version Release Date'],axis=1 , inplace=True)
        data['Size'] = round(data['Size'] / 1000000, 2)
        data['Age Rating'] = data['Age Rating'].str.replace('+', '').astype(int)
        return data

    def labelencoding(self, dataframe):
        label_encoded = dataframe[
            ['User Rating Count', 'Price', 'Size', 'Primary Genre', 'Languages', 'Original Release Date',
             'Current Version Release Date','Genres']]
        onehot_coded = dataframe[['In-app Purchases', 'Age Rating']]
        # Handle missing values in original data
        label_encoded.fillna(value='Unknown', inplace=True)
        # Handle unknown values during one-hot encoding
        enc = OneHotEncoder(handle_unknown='ignore')
        df_oh = pd.DataFrame(enc.fit_transform(onehot_coded).toarray(), columns=enc.get_feature_names_out())
        # Verify and align indices of original data and one-hot encoded data
        label_encoded.reset_index(drop=True, inplace=True)
        df_oh.reset_index(drop=True, inplace=True)
        # Perform label encoding on label_encoded data
        df_le = label_encoded.apply(LabelEncoder().fit_transform)
        # Join the label encoded and one-hot encoded dataframes
        dataframe = pd.concat([df_le, df_oh], axis=1)
        return dataframe

    def ListForNumerical(self,data):
        numericavariableslist = data.select_dtypes(include=np.number).columns.tolist()
        return numericavariableslist

    def ListForCategorical(self,data):
        categoricalvariableslist = data.select_dtypes(include=['object']).columns.tolist()
        return categoricalvariableslist

    def feature_selection(self, X, y, k):
        #Anova
        #Initialize SelectKBest with f_classif as scoring function
        selector = SelectKBest(f_regression, k=10)
        # Fit the selector to the data
        selector.fit(X, y)
        # Get the scores and p-values for each feature
        scores = selector.scores_
        p_values = selector.pvalues_
        # Create a dataframe to store the feature names, scores, and p-values
        feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores, 'P-Value': p_values})
        # Sort the features by score in descending order
        feature_scores.sort_values('Score', ascending=False, inplace=True)
        # Select the top k features
        selected_features = feature_scores.head(k)['Feature'].tolist()
        return selected_features

    def select_features_numeric(self,data, k=10):
        # Select numerical features
        numeric_vars = self.ListForNumerical(data)
        # Select the top k numerical features using SelectKBest with f_classif scoring
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(data[numeric_vars], data['Target_Variable'])
        selected_numeric_vars = list(data[numeric_vars].columns[selector.get_support()])
        return selected_numeric_vars

    def select_features_categorical(self,data, k=10):
        # Select categorical features
        categorical_vars = self.ListForCategorical(data)
        # Convert categorical features to dummy variables
        data_dummies = pd.get_dummies(data[categorical_vars])
        # Select the top k categorical features using SelectKBest with chi2 scoring
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(data_dummies, data['Target_Variable'])
        selected_categorical_vars = list(data_dummies.columns[selector.get_support()])
        return selected_categorical_vars

    def scaledata(self,dataframe):
        scaler = MinMaxScaler()
        dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
        return dataframe

    def PreProcess(self,data):
        data = self.removenulls(data)
        data = self.EncodingForDates(data)
        data = self.LogNormaliztion(data)
        data = self.WorkOnColumns(data)
        data = self.labelencoding(data)
        data = self.scaledata(data)
        return data
