# bankrupcty prediction

import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv("data/1_american_dataset.csv")
print(data.head())

# Initialize the model
model = LinearRegression()
X = data.iloc[:, :-1]  # select all columns except the last one
y = data.iloc[:, -1]   # select only the last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)

# k = 10  # select the top 10 features
# selector = SelectKBest(chi2, k)
# X_new = selector.fit_transform(X, y)