#Prof. Fernando Amaral
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

#Caminho dos arquivos de treino e teste
caminho = ["Churn_treino.csv","Churn_teste.csv"]

#prepara os dados, exclui colunas inuteis, define tipos, extrai informaçõeses importantes etc
imp = Reader(sep = ";")
dados = imp.train_test_split(caminho, "Exited")

#Drif
rdrift = Drift_thresholder()
dados = rdrift.fit_transform(dados)

#Objeto otimizador
otimizador = Optimiser()

#Paramametros do otimizaodr, do tipo dicionário
espaco = {
     'fs__strategy':{"search":"choice","space":["variance","rf_feature_importance"]},
     'est__colsample_bytree':{"search":"uniform", "space":[0.3,0.7]}
}

#Criacao do modelo de fato
# max_evals é as interação, default é 40
modelo = otimizador.optimise(espaco,dados,max_evals=15)

#previsão
previsor = Predictor()
previsor.fit_predict(modelo, dados)