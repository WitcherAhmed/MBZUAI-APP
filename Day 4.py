import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris= load_iris()
df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
df['target'] = iris['target']
x = df.drop('target', axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc.predict(x_test)
plt.figure(figsize=(6, 10))
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(x_train, y_train)
preds = rfc.predict(x_test)
print(np.mean(preds == y_test))

plt.show()
