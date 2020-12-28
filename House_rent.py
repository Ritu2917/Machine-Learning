import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1]

df = pd.read_csv('HouseRent.csv')
print(df.head())
print(df.shape)
print(df.describe())

# 2]

print(df.isnull().sum())
df = df.drop(['bathroom', 'floor', 'animal', 'hoa (R$)', 'property tax (R$)', 'fire insurance (R$)'], axis=1)
print(df.head())

le = LabelEncoder()

n1 = ['city']
for i in n1:
    df[i] = le.fit_transform(df[i])
n1 = ['furniture']
for i in n1:
    df[i] = le.fit_transform(df[i])
print(df.head())

# 3]

sns.barplot(x='city', y='area', data=df)
plt.show()
sns.boxplot(x='city', y='area', data=df)
plt.show()
sns.boxplot(x='rooms', y='rent amount (R$)', data=df)
plt.show()
sns.countplot(x='city', data=df)
plt.show()

# 4]

X = df.drop(["total (R$)", "furniture", "rent amount (R$)"], axis=1)
Y = df['total (R$)']
print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=112)
print(X_train)

# 5]

# Linear Regression

my_model = LinearRegression()

result = my_model.fit(X_train, Y_train)

predictions = result.predict(X_test)

print(predictions)

print('Accuracy of Linear regression := ', r2_score(Y_test, predictions))

plt.scatter(Y_test, predictions, color='c')
plt.plot(X_test, predictions, color='k')
plt.show()

pred_new = result.predict([[4, 52, 2, 1]])
print(pred_new)

# Logistic Regression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=156)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

my_model1 = LogisticRegression()

result1 = my_model1.fit(X_train, Y_train)

predictions1 = result1.predict(X_test)

print(predictions1)
print("Accuracy of Logistic Regression:", accuracy_score(Y_test, predictions1))

pred_new1 = result1.predict([[4, 52, 2, 1]])
print(pred_new1)

print('By Obseving Accuracy of both model we conclude that Linear Model gives the maximum accuracy.'
      'So Linear model will be the best model for our Preticting House Rent Project')
