import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import( accuracy_score , confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df = pd.read_csv('wine_data.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe())
x = df.drop('class_label' , axis = 1)
y = df['class_label']
print(x.shape)
print(y.shape)
x_train , x_test , y_train , y_test = train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
model = LogisticRegression()
model.fit(x_train,y_train)
print("model training completed")
y_pred = model.predict(x_test)
print(y_pred)
print(y_test.values)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model,'models/wine.pkl')
joblib.dump(scaler,'models/scaler.pkl')
print("model and scaler saved successfully")