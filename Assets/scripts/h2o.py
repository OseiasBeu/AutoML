#Prof. Fernando Amaral
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

#Importa dados e divide em treino e teste
imp = pd.read_csv('Churn_treino.csv', sep=";")
imp = h2o.H2OFrame(imp)
treino,teste = imp.split_frame(ratios=[.7])

#Transforma a variável dependente em fator
treino["Exited"] = treino["Exited"].asfactor()
teste["Exited"] = teste["Exited"].asfactor()

#Busca o modelo por 60 segundos, podemos em vez disso definir max_models
modelo = H2OAutoML(max_runtime_secs=60)
modelo.train( y="Exited", training_frame=treino)

#Ranking dos melhores
ranking = modelo.leaderboard
ranking = ranking.as_data_frame()

#Prever 
teste = pd.read_csv('Churn_prever.csv', sep=";")
teste = h2o.H2OFrame(teste)
prever = modelo.leader.predict(teste)
prever = prever.as_data_frame(prever)


