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

#pegando os dados pela uri
datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura_Voice-Exercise/Customer-Churn.csv')
head = datas.head()
print(head)

#informando que devem ser trocados os valores sim e não em 0 e 1
change = {
    'Nao' : 0,
    'Sim' : 1
}

#inrformando quais colunas devem ser trocados os sim e não
modifiedData = datas[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(change)
head = modifiedData.head()
print(modifiedData)

#esta função esta retirando todos os dados modificados manualmente, ja os dados que ficaram são dividido em grupos, seus valores são True e False
dummieDatas = pd.get_dummies(datas.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis = 1))
head = dummieDatas.head()
print(head)

#esta função está concatenando os dados 
finalDatas = pd.concat([modifiedData, dummieDatas], axis = 1)
head = finalDatas.head()
print(head)

#função para imprimir a tabela intera (a tabela possui 39 colunas)
pd.set_option('display.max_columns', 39)
head = finalDatas.head()
print(head)

xMaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

#dados não balanceados
#ax = sns.countplot(x='Churn', data=finalDatas)
#plt.show()

#definindo os dados de input e output
X = finalDatas.drop('Churn', axis = 1)
y = finalDatas['Churn']

#instancia um obj SMOTE
smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y) # Realiza a reamostragem do conjunto de dados
finalDatas = pd.concat([X, y], axis=1) # Concatena a variável target (y) com as features (X)
head = finalDatas.head(2)
print(head)

#dados balanceados
#ax = sns.countplot(x='Churn', data=finalDatas)
#plt.show()

X = finalDatas.drop('Churn', axis = 1)
y = finalDatas['Churn']

#O StandardScaler pegará os atributos, subtrairá a média do conjunto e fará a divisão pelo desvio padrão
stand = StandardScaler()
standardizedX = stand.fit_transform(X)
print(standardizedX[0])

#esta função esta normalizando o xMaria, e ao mesmo tempo esta dividindo o vetor nas colunas iguais as de X
standardizedXMaria = stand.transform(pd.DataFrame(xMaria, columns = X.columns))
print(standardizedXMaria)

#Fazendo a distancia euclidiana entre xMaria e com o cliente 0
a = standardizedXMaria
b = standardizedX[0]
np.square(a-b)
sum = np.sum(np.square(a-b))
distanceMaria = np.sqrt(sum)
print(distanceMaria)

#aqui dividimos o standardizedX e o y entre treino e teste, sendo o teste 30% da tabela, os valores são aleatórios
trainX, testX, trainY, testY = train_test_split(standardizedX, y, test_size = 0.3, random_state = 123)

knn = KNeighborsClassifier(metric = 'euclidean') #instanciando o modelo euclidiano
knn.fit(trainX, trainY) #treinando o modelo
predictKnn = knn.predict(testX) #testando o modelo com os valores de teste
print(predictKnn)

#foi escolhido usar a mediana, porque é o valor central dos dados ordenados
median = np.median(trainX)
print(median)

#modelo criado, o 0.44 se da pelo resultado da mediana, ou seja, defini-se que os valores acima de 0.44 se tornarão 1, e abaixo 0
bnb = BernoulliNB(binarize=0.44)
bnb.fit(trainX, trainY)
predictBnb = bnb.predict(testX)
print(predictBnb)

dct = DecisionTreeClassifier(criterion='entropy', random_state=42) #instanciando o modelo
dct.fit(trainX, trainY) #teinando o modelo
dct.feature_importances_ #verificar a importancia de cada atributo, ou seja, aqui é onde o modelo fará uma classificação entre as features para saber qual a melhor para usar
predictDct = dct.predict(testX)
print(predictDct)

#a matriz de decisão será a métrica usada para decidir qual o melhor modelo
print(confusion_matrix(testY, predictKnn))
print(confusion_matrix(testY, predictBnb))
print(confusion_matrix(testY, predictDct))

#outra métrica para classificar o melhor modelo
print(accuracy_score(testY, predictKnn))
print(accuracy_score(testY, predictBnb))
print(accuracy_score(testY, predictDct))
print('.')

#métrica que classifica o melhor modelo, a partir do calculo de quantos VP foram preditos de forma correta
print(precision_score(testY, predictKnn))
print(precision_score(testY, predictBnb))
print(precision_score(testY, predictDct))
print('.')

#métrica que classifica o melhor modelo, com este calculo sabemos o quão bom é o modelo em classificar valores verdadeiramente positivos
print(recall_score(testY, predictKnn))
print(recall_score(testY, predictBnb))
print(recall_score(testY, predictDct))

#A métrica escolhida foi a precisão, pois, calcula quantos valores positivos foram preditos de forma correta,
#ou seja, os valores positivos são os com o Churn classificados como sim, e estes são clientes que podem sair da empresa,
#com base nos dados a empresa pode tomar medidas para que este número caia.
#O modelo escolhido com base na métrica de precisão foi a Árvore de decisão, pois sua porcentagem é maior.