import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

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

#O StandardScaler pegará os atributos, subtrairá a média do conjunto e fará a divisão pelo dresvio padrão
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