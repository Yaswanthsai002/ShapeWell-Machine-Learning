import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_csv("Asana.csv")

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

mms = MinMaxScaler()
x = mms.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(x,Y, test_size=0.3)

rfr = RandomForestClassifier()
rfr.fit(x_train, y_train)

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

scores = cross_val_score(rfr, x, y, cv=26)

print("Accuracy for each fold:", scores)

print("Average accuracy:", scores.mean())

print("Random Forest Classifier : ", round(accuracy_score(y_test,rfr.predict(x_test)), 2))
print("Decision Tree Classifier : ", round(accuracy_score(y_test,dtc.predict(x_test)), 2))
print("K Neighbors   Classifier : ", round(accuracy_score(y_test,knn.predict(x_test)), 2))
print(df['NAME_OF_THE_ASANA'].value_counts())


f=open('model.pkl','wb')
pickle.dump(rfr,f)
f.close()