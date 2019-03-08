import pandas as pd
import numpy as np

from collections import Counter

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score



def start_process():
    df = pd.read_csv('situacao_cliente.csv')
    X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
    Y_df = df['situacao']

    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df

    X = Xdummies_df.values
    Y = Ydummies_df.values 

    porcentagem_de_treino = 0.8

    tamanho_de_treino = int(porcentagem_de_treino * len(Y))
    tamanho_de_validacao = len(Y) - tamanho_de_treino

    treino_dados = X[:tamanho_de_treino]
    treino_marcacoes = Y[:tamanho_de_treino]

    validacao_dados = X[tamanho_de_treino:]
    validacao_marcacoes = Y[tamanho_de_treino:]

    resultados = {}
    modeloMultinomial = MultinomialNB()
    resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
    resultados[resultadoMultinomial] = modeloMultinomial

    modeloAdaBoost = AdaBoostClassifier()
    resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
    resultados[resultadoAdaBoost] = modeloAdaBoost

    modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0, max_iter=5000))
    resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsRest] = modeloOneVsRest

    modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0, max_iter=5000))
    resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsOne] = modeloOneVsOne

    vencedor = resultados[max(resultados)]
   
    teste_real(vencedor, validacao_dados, validacao_marcacoes)

    total_de_elementos = len(validacao_dados)
    print("Total de teste: %d" % total_de_elementos)


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    taxa_de_acerto = np.mean(cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k))
    print("Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto))
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    modelo.fit(validacao_dados, validacao_marcacoes)
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    print("Taxa de acerto do vencedor no mundo real: {0} %".format(taxa_de_acerto))
    

start_process()