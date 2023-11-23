#!/usr/bin/env python
# coding: utf-8

# # Predicting the accommodation type for a trip with Random Forest Classifier

# In[ ]:


import numpy as np
import pandas as pd

import gzip
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import plot_tree


# # Prepare Data

# ### Import

# Loading text file into a Pandas Dataframe

# In[ ]:


#df=wrangle("/Users/Maryam/Desktop/compmanies/Qualogy/train_data.txt")
# read Text file into Pandas DataFrame
filepath="/Users/Maryam/Desktop/compmanies/Qualogy/train_data.txt"
df = pd.read_csv(filepath, names = ['val'])
df.head()
#df.val = df.val.str.replace(r'^\[1\]','', regex=True).str.strip().replace('<NA>',np.NAN)


# ### Preprocessing

# In[ ]:


# cleaning the columns values ( removing" [1]" and redundent spaces)
df.val = df.val.str.replace(r'^\[1\]','', regex=True).str.strip()

# change "<NA>" to "NaN" value
df.val = df.val.replace('<NA>',np.NAN)
df.head()


# ### Reshaping

# In[ ]:


#Reshape DataFrame and change Dataframe with one column to a Dataframe with 8 columns
column_names = ["record","id", "durationOfStay", "gender", "Age", "kids", "destinationCode", "AcomType"]
df = pd.DataFrame(df.val.to_numpy().reshape(int(df.shape[0] /8), 8) , columns=column_names)


# In[ ]:


df.head()


#  A summary of the DataFrame, including information about the index, column names, non-null values, and data types of each column, number of observations and columns

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# Our goal is to discover patterns in the traveler and trip features that predict the
# accommodation type for a trip.

# ### Explore

# Chacking class balancy of target column

# In[ ]:


df["AcomType"].value_counts(normalize=True).plot(kind="bar",
                                                  xlabel="Accommodation Type",
                                                  ylabel="Frequency",
                                                  title="Class Balance"
                                                 );


# Majority class is "Hotel" and Minority class is "Apt". There is no need to resample classes

# Dealing with "Nan" values

# In[ ]:


# all columns including Nan values
df.isnull().sum()


# In[ ]:


#Dealing with "Age" Nan values

df["Age"]= df["Age"].fillna(0).astype(int)
df["Age"].dtype


# In[ ]:


# median of "Age" column
median=round(df["Age"].median())


# In[ ]:


# Fill missing values with mean of "age" column
df["Age"]=df["Age"].replace(0, median)
df["Age"]


# In[ ]:





# In[ ]:


# Dealing with "destinationCode" Nan values
df["destinationCode"].value_counts(ascending=False).plot(kind="bar",
                                                         xlabel="Destination Code",
                                                          ylabel="Frequency",
                                                          title="Frequency of Destination"
                                                         );


# In[ ]:


majority_destination=df["destinationCode"].value_counts(ascending=False).index[0]
majority_destination


# In[ ]:


df["destinationCode"].fillna(majority_destination, inplace=True)


# In[ ]:


# Dealing with "kids" Nan values

df["kids"].value_counts(ascending=False).plot(kind="bar",
                                             xlabel="Kids",
                                             ylabel="Frequency",
                                             title="Frequency of Destination"
                                             );


# In[ ]:


# Drop "Nan" values
df.dropna(inplace=True)
df.shape


# In[ ]:


# Changing the type of columns "duration_stay"  numeric type
df["durationOfStay"]=df["durationOfStay"].astype(int)


# Checking High and Low Cardinality

# In[ ]:


df.select_dtypes("object").nunique()


# Remove columns with High Cardinality and Low cardinality

# In[ ]:


df=df[ ["durationOfStay", "gender", "Age", "kids", "destinationCode", "AcomType"]]
df.head()


# Multicollinearity between numeric coolumns

# In[ ]:


corr=df.select_dtypes("number").corr()
corr


# In[ ]:


sns.heatmap(corr);


# Age and Accommodation Type

# In[ ]:


#Box plot for "Age" column

sns.boxplot(x="AcomType", y="Age", data=df)
plt.xlabel("accommodation Type")
plt.ylabel("Age")
plt.title("Distribution of Age by Class");


# In[ ]:


# Histogram plot for "Age" column
df["Age"].hist(bins=10)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Distribution of Age");


# This implies that the most age renge lies between 37 and 40 years old

# In[ ]:


# Histogram plot for "duration_stay" column
df["durationOfStay"].hist(bins=10)
plt.xlabel("duration stay")
plt.ylabel("Count")
plt.title("Distribution of duration_stay");


# The graph has a right skewed distribution demonesterating that most of observations are focused on the left and a majority of people are likely to book accommodation(both Hotel and Apartment)  for 3 or 4 days

# In[ ]:


#Box plot for "durationOfStay" column
sns.boxplot(x="AcomType", y="durationOfStay", data=df)
plt.xlabel("Accommodation Type")
plt.ylabel("durationOfStay")
plt.title("Distribution of durationOfStay by Class");


# Gender and Accommodation Type

# In[ ]:


df["gender"].value_counts(ascending=False)


# In[ ]:


# Stacked bar chart

stacked_data = df.groupby(['gender', 'AcomType']).size().unstack()

stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# This shows that number of Women that booked Hotel is greater that Men

# In[ ]:


#The relationship between booker and having kids
stacked_data = df.groupby(['gender', 'kids']).size().unstack()

stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# This implies that there is no significant difference between Gender and Kids olumns

# In[ ]:


stacked_data = df.groupby(['kids', 'AcomType']).size().unstack()

stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart')
plt.xlabel('Kids')
plt.ylabel('Count')
plt.show()


# This shows that families having kids are more likely to book apartment rather that Hotel

# In[ ]:


stacked_data = df.groupby(['destinationCode', 'AcomType']).size().unstack()

stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart')
plt.xlabel('destination_code')
plt.ylabel('Count')
plt.show()


# Sweden is the a destination which people are more linkely to book Apartment and Belgium is a destination with the hiest number of Hotel book, In Finland people are more likely to book Hotel. Overall the avarage of booking Hotel is greater that bookin Apartment

# ## Split

# In[ ]:


target = "AcomType"
# Vertical Split
X = df.drop(columns="AcomType")
y = df[target]


# In[ ]:


# Horizental Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


# ## Build Model

# ### Baseline

# Calculate the baseline accuracy score for our model

# In[ ]:


acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))


# Create a pipeline named clf (short for "classifier") that contains a  OrdinalEncoder transformer and a RandomForestClassifier predictor.

# In[ ]:


clf = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(random_state=42)
)
print(clf)


# Create a dictionary with the range of hyperparameters that we want to evaluate for our classifier

# In[ ]:


params ={
    "randomforestclassifier__n_estimators":range(25,100, 25) ,
    "randomforestclassifier__max_depth":range(10,50,10)
    }
params


# Create a GridSearchCV named model that includes our classifier and hyperparameter grid.

# In[ ]:


model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model


# Fit the model

# In[ ]:


# Train model
model.fit(X_train, y_train)


# Extract the cross-validation results from model and load them into a DataFrame named cv_results

# In[ ]:


type(model.cv_results_)


# In[ ]:


cv_results =pd.DataFrame(model.cv_results_)
cv_results.head(2)


# In[ ]:


cv_results.info()


# In[ ]:


# Create mask
mask = cv_results['param_randomforestclassifier__n_estimators']==25
# Plot fit time vs max_depth
plt.plot(cv_results[mask]["param_randomforestclassifier__max_depth"],cv_results[mask]["mean_fit_time"] )
# Label axes
plt.xlabel("Max Depth")
plt.ylabel("Mean Fit Time [seconds]")
plt.title("Training Time vs Max Depth (n_estimators=25)");


# In[ ]:


# Extract best hyperparameters
model.best_params_


# In[ ]:


model.best_score_


# In[ ]:


model.best_estimator_


# ## Evaluate

# Let's see how our model performs

# In[ ]:


acc_train =model.score(X_train, y_train)
acc_test = model.score(X_test,y_test)

print("Training Accuracy:", round(acc_train, 3))
print("Test Accuracy:", round(acc_test, 3))


# In[ ]:


y_test.value_counts()


# In[ ]:


# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred[:10]


# In[ ]:


# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model,X_test, y_test);


# ## Communicate

# Create a horizontal bar chart with the 10 most important features for your model.

# In[ ]:


# Get feature names from training data
features =X_train.columns
# Extract importances from model
importances =model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
# Create a series with feature names and importances
feat_imp = pd.Series(importances, index=features).sort_values()
# Plot 10 most important features
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");


# Save model

# In[ ]:


# Save model
with open("model-1-1.pkl","wb") as f:  
    pickle.dump(model,f)

