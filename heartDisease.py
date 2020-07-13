import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC

dataset=pd.read_csv("11-3-Dataset-Prediction of Heart Disease.csv")

print("\n***PREDICTION OF HEART DISEASE***\n")
print("\nDataset:\n")
print(dataset)
print("\nDataset Size:",dataset.shape)
print("\nDataset Description:\n",dataset.describe())

print("\nNull Values in dataset:\n",dataset.isnull().sum())
dataset["education"]=dataset["education"].fillna(dataset["education"].median())
dataset["cigsPerDay"]=dataset["cigsPerDay"].fillna(dataset["cigsPerDay"].median())
dataset["BPMeds"]=dataset["BPMeds"].fillna(dataset["BPMeds"].median())
dataset["totChol"]=dataset["totChol"].fillna(dataset["totChol"].median())
dataset["BMI"]=dataset["BMI"].fillna(dataset["BMI"].median())
dataset["heartRate"]=dataset["heartRate"].fillna(dataset["heartRate"].median())
dataset["glucose"]=dataset["glucose"].fillna(dataset["glucose"].median())
print("\nNull Values in dataset:\n",dataset.isnull().sum())

x=dataset.drop(["TenYearCHD","education"],axis=1)
y=dataset.TenYearCHD

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=156)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

sns.countplot(y)
plt.show()
print("\nTarget Values:\n",dataset["TenYearCHD"].value_counts())

print("\n***LOGISTIC REGRESSION***\n")
logReg=LogisticRegression()
resultLogReg=logReg.fit(x_train,y_train)
predLogReg=resultLogReg.predict(x_test)
print("\nPredictions on test set using Logistic Regression:\n",predLogReg)
print("\nAccuracy Score acheived using Logistic Regression: ",accuracy_score(y_test,predLogReg),"%")

conMatLogReg=confusion_matrix(y_test,predLogReg)
conMatLogRegDF=pd.DataFrame(conMatLogReg,index=['Actual Negative','Actual Positive'],columns=['Predicted Negative','Predicted Positive'])
print("\nConfusion Matrix for Logistic Regression:\n",conMatLogRegDF)
colConMatLogReg=sns.heatmap(conMatLogRegDF,cmap='coolwarm',annot=True)
print()
plt.show()
print("\nClassification Report for Logistic Regression:\n",metrics.classification_report(y_test,predLogReg))

print("\n***DECISION TREES***\n")
decTree=DecisionTreeClassifier(random_state=0)
resultDecTree=decTree.fit(x_train,y_train)
predDecTree=resultDecTree.predict(x_test)
print("\nPredictions on test set using Decision Tree:\n",predDecTree)
print("\nAccuracy Score acheived using Decision Tree: ",accuracy_score(y_test,predDecTree),"%")

conMatDecTree=confusion_matrix(y_test,predDecTree)
conMatDecTreeDF=pd.DataFrame(conMatDecTree,index=['Actual Negative','Actual Positive'],columns=['Predicted Negative','Predicted Positive'])
print("\nConfusion Matrix for Decision Tree:\n",conMatDecTreeDF)
colConMatDecTree=sns.heatmap(conMatDecTreeDF,cmap='coolwarm',annot=True)
print()
plt.show()
print("\nClassification Report for Decision Tree:\n",metrics.classification_report(y_test,predDecTree))

print("\n***RANDOM FOREST***\n")
ranFor=RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=42)
resultRanFor=ranFor.fit(x_train,y_train)
predRanFor=resultRanFor.predict(x_test)
print("\nPredictions on test set using Random Forest:\n",predRanFor)
print("\nAccuracy Score acheived using Random Forest: ",accuracy_score(y_test,predRanFor),"%")

conMatranFor=confusion_matrix(y_test,predRanFor)
conMatRanForDF=pd.DataFrame(conMatranFor,index=['Actual Negative','Actual Positive'],columns=['Predicted Negative','Predicted Positive'])
print("\nConfusion Matrix for Random Forest:\n",conMatRanForDF)
colConMatRanFor=sns.heatmap(conMatRanForDF,cmap='coolwarm',annot=True)
print()
plt.show()
print("\nClassification Report for Random Forest:\n",metrics.classification_report(y_test,predRanFor))

print("\n***K MEANS CLUSTERING***\n")
ssq=[]
for k in range(1,6):
    KMC=KMeans(n_clusters=k,random_state=123) 
    resultKMC=KMC.fit(dataset)
    ssq.append(KMC.inertia_)

plt.plot(range(1,6),ssq,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Within Cluster SSQ")
plt.title("SSQ Plot")
plt.show()

KMC=KMeans(n_clusters=2,random_state=163)
resultKMC=KMC.fit(dataset)
predKMC=resultKMC.predict(dataset)
print("\nPredictions on test set using K Means Clustering:\n",predKMC)

print("\n***SUPPORT VECTOR MACHINE***\n")
SVM=SVC(kernel='rbf',random_state=0)
resultSVM=SVM.fit(x_train,y_train)
predSVM=resultSVM.predict(x_test)
print("\nPredictions on test set using Support Vector Machines:\n",predSVM)
print("\nAccuracy Score acheived using Support Vector Machines: ",accuracy_score(y_test,predSVM),"%")

conMatSVM=confusion_matrix(y_test,predSVM)
conMatSVMDF=pd.DataFrame(conMatSVM,index=['Actual Negative','Actual Positive'],columns=['Predicted Negative','Predicted Positive'])
print("\nConfusion Matrix for Support Vector Machines:\n",conMatSVMDF)
colConMatSVM=sns.heatmap(conMatSVMDF,cmap='coolwarm',annot=True)
print()
plt.show()
print("\nClassification Report for Support Vector Machines:\n",metrics.classification_report(y_test,predSVM))
 
scores=[accuracy_score(y_test,predLogReg),accuracy_score(y_test,predDecTree),accuracy_score(y_test,predRanFor),accuracy_score(y_test,predSVM)]
algorithms=["Logistic Regression","Decision Trees","Random Forest","Support Vector Machine"]    

for i in range(len(algorithms)):
    print("\nAccuracy Score achieved using "+algorithms[i]+" is: ",scores[i],"%")

sns.set(rc={'figure.figsize':(15,4)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
sns.barplot(algorithms,scores)
plt.show()