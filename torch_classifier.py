import numpy as np
import torch
import pandas as pd
from flask import jsonify
from torch.autograd import Variable
import torch.nn.functional as F
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4 , 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100 , 3)
        self.dropout = torch.nn.Dropout(0.2)
        self.softmax = torch.nn.Softmax(dim = 1)
        
    def forward(self , x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
model = Net()

class IrisTClassifier:

    def __init__(self , data):
        self.data = data

    def predict(data):
        
        pdData = pd.DataFrame(data , index = [0])
        #Convert into Numpy array so that it can be converted to tensors
        pdData = pdData.values  
        #Convert from numpy to tensors
        pdData = torch.from_numpy(pdData)
        pdData = Variable(torch.Tensor(pdData.float()))
        model.load_state_dict(torch.load('IrisTorchClassifier.pt'))
        model.eval()
        prediction = model(pdData)

        _ , predict_y = torch.max(prediction , 1)

        return str(predict_y.item())