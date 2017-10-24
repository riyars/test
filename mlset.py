import numpy as np
import matplotlib as plot
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#data = pd.read_csv("train.csv")

#print("Data \n",data)

data = pd.read_csv("train.csv").as_matrix()
print("Data \n",data)

clf=DecisionTreeClassifier()

x_train=data[0:21000,1:]
y_train=data[0:21000,0]

x_test=data[21000,1:]
y_test = data[21000,0]

clf.fit(x_train,y_train)

disp=x_test[0]
disp.shape = [28,28]
plot.inshow(255,disp,cmap='gray')
plot.show()


p=clf.predict([x_test[8]])


