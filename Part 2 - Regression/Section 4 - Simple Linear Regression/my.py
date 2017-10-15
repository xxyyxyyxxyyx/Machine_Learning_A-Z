import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import data
dataset = pd.read_csv('Salary_Data.csv');
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# Splitting data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

# Fitting Simple linear Regression to train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predict tests
y_pred = regressor.predict(X_test)

# Visualization training
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualization test
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()