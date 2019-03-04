import pandas as pd

data = pd.read_csv('iris_training.csv' , delimiter = ',' , encoding='utf-8' , names = ['s_l','s_w','p_l','p_w','classes'])

import torch
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

import sklearn
from sklearn.model_selection import train_test_split

features = data.iloc[:,:-1]
labels = data.iloc[:, -1]

features = features.values
labels = labels.values

features = torch.from_numpy(features)
labels = torch.from_numpy(labels)

features = Variable(torch.Tensor(features.float()))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr = 0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    
    loss = criterion(model(features) , labels)
    loss.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch---->{}\tLoss:----->{}'.format(epoch , loss.item()))

torch.save(model.state_dict() , 'IrisTorchClassifier.pt')

test_data = pd.read_csv('iris_test.csv' , delimiter = ',' , encoding='utf-8')

test_features = test_data.iloc[: , :-1]
test_labels = test_data.iloc[: , -1]

test_features = test_features.values

test_features = torch.from_numpy(test_features)
test_features = Variable(torch.Tensor(test_features.float()))

prediction = model(test_features)

_ , predict_y = torch.max(prediction , 1)

from sklearn.metrics import accuracy_score
print('prediction accuracy', accuracy_score(test_labels.data, predict_y.data))

