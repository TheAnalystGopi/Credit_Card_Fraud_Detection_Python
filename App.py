import numpy as num
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv(r"D:\data analytics\Projects\Credit Card Fraud Detection Python\archive\creditcard.csv")
credit_card_data
credit_card_data.head()
credit_card_data.info()
credit_card_data.isnull().sum()
credit_card_data.shape
credit_card_data.drop_duplicates(inplace=True)
credit_card_data['Class'].value_counts()
legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
legit.Amount.describe()
fraud.Amount.describe()
legit_sample = legit.sample(n=473)
new = pd.concat([legit_sample, fraud], axis=0)
new['Class'].value_counts()
x = new.drop(columns='Class', axis=1)
y =  new['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
model = LogisticRegression()
model.fit(x_train, y_train)

from sklearn.metrics import precision_score, f1_score, recall_score
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
training_data_precision = precision_score(x_train_prediction, y_train)

print(f"Accuracy Score on Training Data {training_data_accuracy}")
print(f"precision_score on a training data is {training_data_precision}")

## Accuracy on test data
from sklearn.metrics import precision_score, f1_score, recall_score
x_test_prediction = model.predict(x_test)
test_data_accuaracy = accuracy_score(x_test_prediction, y_test)
test_precision_score = precision_score(x_test_prediction, y_test)
print(f"Accuracy score on a test data is {test_data_accuaracy}")
print(f"precision score is {test_precision_score}")

import streamlit as st
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = num.array(input_df_lst, dtype=num.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")


## Termional streamlit run App.py




































