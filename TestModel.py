#!/usr/bin/env python
# coding: utf-8

# # Make Prediction with Test Dataset

# In[ ]:


import numpy as np
import pandas as pd

import gzip
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from category_encoders import OrdinalEncoder

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import plot_tree


# Create a function make_predictions. It should take two arguments: the path of a JSON file that contains test data and the path of a serialized model. The function should load and clean the data using the wrangle function you created, load the model, generate an array of predictions, and convert that array into a Series. (The Series should have the name "AccommodationType" and the same index labels as the test data.) Finally, the function should return its predictions as a Series.
# 
# 

# In[ ]:


def wrangle(data_filepath):
    # Load csv file into pandas dataframe
    df = pd.read_csv(data_filepath)
    #remove target column
    df.drop(columns=["AcomType"],inplace=True)

    #remove columns high and low cardinality 
    df.drop(columns=["id"],inplace=True)

    #Fill nan values of "Age" column
    df["Age"]= df["Age"].fillna(0).astype(int)
    median=df["Age"].median()
    df["Age"]=df["Age"].replace(0, median)

    # Filling missing values " destinationCode" column with the destination having most frequency
    majority_destination=df["destinationCode"].value_counts(ascending=False).index[0]
    df["destinationCode"]= df["destinationCode"].fillna(majority_destination)

    # drop missing values
    df.dropna(inplace=True)

    return df


# In[ ]:


def make_predictions(data_filepath, model_filepath):
    # Wrangle csv file
    X_test = wrangle(data_filepath)
    # Load model
    with open(model_filepath, "rb") as f:
        model=pickle.load(f)
    # Generate predictions
    y_test_pred =model.predict(X_test)
    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_test_pred =pd.Series(y_test_pred, X_test.index, name="AccommodationType")
    return y_test_pred


# In[ ]:


y_test_pred = make_predictions(
    data_filepath= "/Users/Maryam/Desktop/compmanies/Qualogy/TestDataAccomodation.csv",
    model_filepath="model-1-1.pkl",
)

print("predictions shape:", y_test_pred.shape)
#y_test_pred.head()
y_test_pred


# ### Compress our predicted data

# In[ ]:


# Convert the Series to bytes
data_bytes = y_test_pred.to_csv(index=False).encode('utf-8')

# Compress the data
compressed_data = gzip.compress(data_bytes)

# Write compressed data to a file
with gzip.open("compressed_prediction.gz", "wb") as f:
    f.write(compressed_data)

print("Series has been compressed and saved to 'compressed_prediction.gz'")

