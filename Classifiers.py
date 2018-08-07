from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#[height ,weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38],[154, 54,37],
      [166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,50,60],
      [171,75,42],[181,85,43]]
Y = ['male','female','female','female','male','male','male','female',
      'male','female','male']
#Decision tree classifier
clf = tree.DecisionTreeClassifier()
#logistic regession classifier
clf1 = LogisticRegression()
#Naive-Bayes classifier
clf2 = GaussianNB()
#Random forest Classifier
clf3 = RandomForestClassifier(n_estimators = 1000)
#KNeighbors Classifier
clf4 = KNeighborsClassifier(n_neighbors = 3)

clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
clf4 = clf4.fit(X,Y)

predict = clf.predict([[169,70,44]])
predict1 = clf1.predict([[169,70,44]])
predict2 = clf2.predict([[169,70,44]])
predict3 = clf3.predict([[169,70,44]])
predict4 = clf4.predict([[169,70,44]])

acuracy = clf.score(X,Y)
acuracy1= clf1.score(X,Y)
acuracy2= clf2.score(X,Y)
acuracy3= clf3.score(X,Y)
acuracy4= clf4.score(X,Y)

print (predict)
print (predict1)
print (predict2)
print (predict3)
print (predict4)

print(acuracy)
print(acuracy1)
print(acuracy2)
print(acuracy3)
print(acuracy4)
