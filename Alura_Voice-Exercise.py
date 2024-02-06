import pandas as pd

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