import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura_Voice-Exercise/Customer-Churn.csv')
head = datas.head()
print(head)

change = {
    'Nao' : 0,
    'Sim' : 1
}

modifiedData = datas[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(change)
head = modifiedData.head()
print(modifiedData)

dummieDatas = pd.get_dummies(datas.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis = 1))
head = dummieDatas.head()
print(head)

finalDatas = pd.concat([modifiedData, dummieDatas], axis = 1)
head = finalDatas.head()
print(head)

pd.set_option('display.max_columns', 39)
head = finalDatas.head()
print(head)

xMaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

#ax = sns.countplot(x='Churn', data=finalDatas)
#plt.show()

X = finalDatas.drop('Churn', axis = 1)
y = finalDatas['Churn']

smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y) 
finalDatas = pd.concat([X, y], axis=1) 
head = finalDatas.head(2)
print(head)

#ax = sns.countplot(x='Churn', data=finalDatas)
#plt.show()

X = finalDatas.drop('Churn', axis = 1)
y = finalDatas['Churn']

stand = StandardScaler()
standardizedX = stand.fit_transform(X)
print(standardizedX[0])

standardizedXMaria = stand.transform(pd.DataFrame(xMaria, columns = X.columns))
print(standardizedXMaria)

a = standardizedXMaria
b = standardizedX[0]
np.square(a-b)
sum = np.sum(np.square(a-b))
distanceMaria = np.sqrt(sum)
print(distanceMaria)

trainX, testX, trainY, testY = train_test_split(standardizedX, y, test_size = 0.3, random_state = 123)

knn = KNeighborsClassifier(metric = 'euclidean') 
knn.fit(trainX, trainY)
predictKnn = knn.predict(testX) 
print(predictKnn)

median = np.median(trainX)
print(median)

bnb = BernoulliNB(binarize=0.44)
bnb.fit(trainX, trainY)
predictBnb = bnb.predict(testX)
print(predictBnb)

dct = DecisionTreeClassifier(criterion='entropy', random_state=42)
dct.fit(trainX, trainY) 
dct.feature_importances_ 
predictDct = dct.predict(testX)
print(predictDct)

print(confusion_matrix(testY, predictKnn))
print(confusion_matrix(testY, predictBnb))
print(confusion_matrix(testY, predictDct))

#outra métrica para classificar o melhor modelo
print(accuracy_score(testY, predictKnn))
print(accuracy_score(testY, predictBnb))
print(accuracy_score(testY, predictDct))
print('.')

print(precision_score(testY, predictKnn))
print(precision_score(testY, predictBnb))
print(precision_score(testY, predictDct))
print('.')

print(recall_score(testY, predictKnn))
print(recall_score(testY, predictBnb))
print(recall_score(testY, predictDct))

#A métrica escolhida foi a precisão, pois, calcula quantos valores positivos foram preditos de forma correta,
#ou seja, os valores positivos são os com o Churn classificados como sim, e estes são clientes que podem sair da empresa,
#com base nos dados a empresa pode tomar medidas para que este número caia.
#O modelo escolhido com base na métrica de precisão foi a Árvore de decisão, pois sua porcentagem é maior.