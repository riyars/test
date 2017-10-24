from scipy.spatial import distance

def eucld(a,b):
     return distance.euclidean(a,b)
class myknn():
#Fit method of Classifier

 def fit(self,X_train,Y_train):
    self.x_train = X_train
    self.y_train = Y_train
#Predict method of Classifier

 def predict(self,X_test):
    predictions=[]
    for row in X_test:
      labels = self.closest(row)
      predictions.append(labels)
    return predictions
#Closest Distance

 def closest(self,row):
    best_dist = eucld(row,self.x_train[0])
    best_index = 0
    for i in range(1,len(self.x_train)):
      dist = eucld(row,self.x_train[i])
      if dist < best_dist:
         best_dist = dist
         best_index = i
    return self.y_train[best_index]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris=load_iris()
features=iris.data
labels=iris.target


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(features,labels,test_size=.3)

clf=myknn()
clf.fit(X_train,Y_train)

p=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("AC:",accuracy_score(Y_test,p))


 
