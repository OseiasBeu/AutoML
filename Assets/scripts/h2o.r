#Prof. Fernando Amaral
library("h2o")

h2o.init()

#Importa dados 
imp = read.csv("Churn_treino.csv", sep = ";")

#Pega o nome da classe - ultima coluna
y = colnames(imp[length(imp)])

#Transforma para objeto do h20
dados = as.h2o(imp)

#Divide em treino e teste
dados = h2o.splitFrame(data=dados, ratios=0.7)
treino = dados[[1]]
teste = dados[[2]]

#Transforma a variável dependente em fator
treino[,y] = as.factor(treino[,y])
teste[,y] = as.factor(teste[,y])

#Busca o modelo por 60 segundos, podemos em vez disso definir max_models
modelo = h2o.automl(x = colnames(treino[1:(length(imp)-1)]), y = y,training_frame = treino, max_runtime_secs = 60 )

#Ranking dos melhores
lb =as.data.frame(modelo@leaderboard)

#Prever
imp = read.csv("Churn_prever.csv", sep = ";")
imp = as.h2o(imp)
previsao = h2o.predict(modelo@leader, imp)
previsao = as.data.frame(previsao)

  
