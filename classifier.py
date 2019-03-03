import numpy as np
from sklearn.externals import joblib
import pandas as pd
from flask import jsonify

class IrisClassifier:

    def __init__(self , data):
        self.data = data

    def predict(data):
        
        pdData = pd.DataFrame(data , index = [0])
        clf = joblib.load('iris_classifier.sav')   
        ans = clf.predict(pdData)
        ans = ans[0]
        print(ans)

        return str(ans)