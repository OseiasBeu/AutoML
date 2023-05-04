#Prof. Fernando Amaral
import autokeras as ak
import pandas as pd

#Importa dados
imp = pd.read_csv('Churn_treino.csv', sep=";")

#Separa variaveis independentes da classe
x = imp.iloc[:,0:10]
y = imp.iloc[:,10]

# Inicializa com 10 modelos diferentes
modelo = ak.StructuredDataClassifier(max_trials=10) 

#Cria o modelo
modelo.fit( x= x, y =y, epochs=100)

#Previsão
prever = pd.read_csv('Churn_prever.csv', sep=";")
previsao = clf.predict(prever)
