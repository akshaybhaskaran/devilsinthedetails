# Importing the needed libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Reading the dataset
data = pd.read_csv("data.csv")
X = data.iloc[:, 0:1].values
y = data.iloc[:, -1].values

# Visualizing the dataset
plt.scatter(X, y, s=50, c='red', marker="o")
plt.title("Swedish Auto Insurance Scatter Plot")
plt.xlabel("X -- number of Claims")
plt.ylabel("y -- Payment for the claims")
plt.show()


# Splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting the model on the Training set
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicting the model on the Test Set
y_pred = lr_model.predict(X_test)






