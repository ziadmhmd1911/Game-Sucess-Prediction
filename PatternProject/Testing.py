from DataCleansing import *
from sklearn.impute import SimpleImputer
class Testing:
    def __init__(self):
        pass

    def SetNulls(self,data):
        # Fill missing values in the 'Languages' column
        imputerLang = SimpleImputer(strategy='most_frequent')
        data['Languages'] = imputerLang.fit_transform(data['Languages'].values.reshape(-1, 1))
        ##########################################################################################

        # Fill missing values in 'In-app Purchases' column with 0
        imputerInapp = SimpleImputer(strategy='constant', fill_value=0)
        data['In-app Purchases'] = imputerInapp.fit_transform(data['In-app Purchases'].values.reshape(-1, 1))
        ##########################################################################################

        if data['User Rating Count'].skew() > 1.0 or data['User Rating Count'].skew() < -1.0:
            # Use median imputation for skewed distributions
            imputerUserRating = SimpleImputer(strategy='median')
        else:
            # Use mean imputation for symmetric distributions
            imputerUserRating = SimpleImputer(strategy='mean')
        data['User Rating Count'] = imputerUserRating.fit_transform(data['User Rating Count'].values.reshape(-1, 1))
        ##############################################################################################################

        imputerPrice = SimpleImputer(strategy='constant', fill_value=0)
        data["Price"] = imputerPrice.fit_transform(data["Price"].values.reshape(-1, 1))
        ##############################################################################################################

        if data['Size'].skew() > 1.0 or data['Size'].skew() < -1.0:
            # Use median imputation for skewed distributions
            imputerSize = SimpleImputer(strategy='median')
        else:
            # Use mean imputation for symmetric distributions
            imputerSize = SimpleImputer(strategy='mean')
        data['Size'] = imputerSize.fit_transform(data['Size'].values.reshape(-1, 1))
        ##############################################################################################################

        imputerPrimaryGenre = SimpleImputer(strategy='most_frequent')
        data['Developer'] = imputerPrimaryGenre.fit_transform(data['Developer'].values.reshape(-1, 1))
        ##############################################################################################################

        imputerGenres = SimpleImputer(strategy='most_frequent')
        data['Genres'] = imputerGenres.fit_transform(data['Genres'].values.reshape(-1, 1))
        ##############################################################################################################

        if data['Original Release Date'].skew() > 1.0 or data['Original Release Date'].skew() < -1.0:
            # Use median imputation for skewed distributions
            imputerOrig = SimpleImputer(strategy='median')
        else:
            # Use mean imputation for symmetric distributions
            imputerOrig = SimpleImputer(strategy='mean')
        data['Original Release Date'] = imputerOrig.fit_transform(data['Original Release Date'].values.reshape(-1, 1))
        ##############################################################################################################

        if data['Current Version Release Date'].skew() > 1.0 or data['Current Version Release Date'].skew() < -1.0:
            # Use median imputation for skewed distributions
            imputerCurr = SimpleImputer(strategy='median')
        else:
            # Use mean imputation for symmetric distributions
            imputerCurr = SimpleImputer(strategy='mean')
        data['Current Version Release Date'] = imputerCurr.fit_transform(data['Current Version Release Date'].values.reshape(-1, 1))
        ##############################################################################################################

        imputerAge = SimpleImputer(strategy='most_frequent')
        data['Age Rating'] = imputerAge.fit_transform(data['Age Rating'].values.reshape(-1, 1))
        ##############################################################################################################

        return data


    def Drop(self,data):
        data.drop(["URL", "ID", "Name", "Subtitle", "Icon URL", "Description", "Primary Genre"], axis=1, inplace=True)
        return data

    def LogNormaliztion(self, dataframe):
        dataframe['Size'] = np.log(dataframe['Size'])
        dataframe['User Rating Count'] = np.log(dataframe['User Rating Count'])
        return dataframe

    def EncodingForDates(self,dataframe):
        dataframe["Original Release Date"] = pd.to_datetime(dataframe["Original Release Date"]).dt.year
        dataframe["Current Version Release Date"] = pd.to_datetime(dataframe["Current Version Release Date"]).dt.year
        return dataframe

    def WorkOnColumns(self,data):
        data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x:1 if x!= 0 else 0)
        data['Size'] = round(data['Size'] / 1000000, 2)
        data['Age Rating'] = data['Age Rating'].str.replace('+', '').astype(int)
        return data

    def labelencoding(self, dataframe):
        label_encoded = dataframe[
            ['User Rating Count', 'Price', 'Size', 'Languages', 'Original Release Date',
             'Current Version Release Date', 'Genres', 'Developer']]
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

    def scaledata(self,dataframe):
        scaler = MinMaxScaler()
        dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
        return dataframe

    def PreProcess(self,data):
        data = self.EncodingForDates(data)
        data = self.Drop(data)
        data = self.SetNulls(data)
        data = self.LogNormaliztion(data)
        data = self.WorkOnColumns(data)
        data = self.labelencoding(data)
        data = self.scaledata(data)
        return data



