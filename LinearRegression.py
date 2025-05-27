import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn # Sci-kit learn : Machine learning python library.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Import in the linear regression class.
from sklearn import metrics

df = pd.read_csv("D:\Blast AI\california_housing_train.csv")

# Matrix of feature vectors "X"
# Target vector "y"
x = df.drop('median_house_value', axis=1)
y = df['median_house_value']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=101)
lm= LinearRegression()
lm.fit(x_train,y_train)
cdf = pd.DataFrame(lm.coef_, x.columns, columns = ['Coeffs'])
predictions = lm.predict(x_test)
residuals = y_test - predictions
metrics.mean_absolute_error(y_test, predictions)# MAE - Mean Absolute Error

print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.show()