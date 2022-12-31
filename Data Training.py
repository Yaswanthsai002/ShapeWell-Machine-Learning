import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# df = pd.read_csv("Asana.csv")
df = pd.read_csv("Asana1.csv")

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values


le = LabelEncoder()
y = le.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

rfr = RandomForestClassifier()
rfr.fit(x_train, y_train)


scores = cross_val_score(rfr, X, y, cv=26)

# Print the mean and standard deviation of the scores
print(f'Mean score: {scores.mean():.3f}')
print(f'Standard deviation: {scores.std():.3f}')

print("Random Forest Classifier : ", round(accuracy_score(y_test,rfr.predict(x_test)), 2))

# print(df['NAME_OF_THE_ASANA'].value_counts())


f=open('model.pkl','wb')
pickle.dump((rfr,le),f)
f.close()