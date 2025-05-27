import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Dealing with classification
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
predictions= model.predict(X_test)
residuals = y_test - predictions

print("accuracy on the test set: ", accuracy_score(y_test, predictions))


plt.show()