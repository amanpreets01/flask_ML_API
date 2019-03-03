import pandas as pd
from sklearn.externals import joblib

data = pd.read_csv('iris_training.csv' , delimiter = ',' , encoding = 'utf-8')

features = data.iloc[: , :-1]
labels = data.iloc[: , -1]

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(features , labels)

filename = 'iris_training.csv'
joblib.dump(clf , filename)