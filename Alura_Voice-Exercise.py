import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

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

#esta função esta restirando todos os dados modificados manualmente, ja os dados que ficaram são dividido em grupos, seus valores são True e False
dummieDatas = pd.get_dummies(datas.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis = 1))
head = dummieDatas.head()
print(head)

finalDatas = pd.concat([modifiedData, dummieDatas], axis = 1)
head = finalDatas.head()
print(head)

pd.set_option('display.max_columns', 39)
head = finalDatas.head()
print(head)

xMaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,1]]

#ax = sns.countplot(x='Churn', data=finalDatas)
#plt.show()

X = finalDatas.drop('Churn', axis = 1)
y = finalDatas['Churn']

smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)
finalDatas = pd.concat([X, y], axis=1)
head = finalDatas.head(2)
print(head)

ax = sns.countplot(x='Churn', data=finalDatas)
plt.show()