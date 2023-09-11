import pickle
import streamlit as st
from Testing import *
from sklearn.model_selection import train_test_split


loaded_model = pickle.load(open('Linear_Model.sav', 'rb'))

data = pd.read_csv("Data/games-regression-dataset.csv")
dataclean = DataCleansing()
data = dataclean.AutoPreProcess(data)
X = data.drop("Average User Rating", axis=1)
y = data["Average User Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = dataclean.PreProcess(X_train)
X_test = dataclean.PreProcess(X_test)
# Call the feature_selection function with the total number of features as k
selected_features = dataclean.feature_selection(X_train, y_train, k=10)

with st.form(key="form1"):
    left_column, right_column = st.columns(2)
    with right_column:
        # Create a list of labels for the text boxes
        labels = ["URL","ID", "Name", "Subtitle", "Icon URL", "User Rating Count", "Price", "In-app Purchases", "Description",
                  "Developer", "Age Rating", "Languages", "Size", "Primary Genre", "Genres",
                  "Original Release Date", "Current Version Release Date"]
        # Create a dictionary to store the user input values
        input_values = {}
        # Loop through the list of labels and create a text box for each one
        for label in labels:
            if label in ["Original Release Date", "Current Version Release Date"]:
                date_input = st.date_input(label=label)
                input_values[label] = date_input.strftime("%Y-%m-%d") if date_input else None
            elif label in ["In-app Purchases", "Languages", "Genres", "Age Rating"]:
                input_values[label] = st.text_input(label=label)
                if input_values[label]:
                    input_values[label] = input_values[label].split(",")
                    input_values[label] = [item.strip() for item in input_values[label]]
            else:
                input_values[label] = st.text_input(label=label)
    with left_column:
        submit = st.form_submit_button(label="Apply Model")
        if submit:
            # Create a DataFrame from the input values
            input_df = pd.DataFrame[input_values]
            TestScript = Testing()
            linear_Model = pickle.load(open('Linear_Model.sav', 'rb'))
            y_pred_logreg = linear_Model.predict(input_df)
            st.write("Predicted Value:", y_pred_logreg)