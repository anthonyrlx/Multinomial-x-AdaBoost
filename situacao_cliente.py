import pandas as pd

from collections import Counter

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC


def start_process():
    df = pd.read_csv('situacao_cliente.csv')
    X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
    Y_df = df['situacao']

    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df

    X = Xdummies_df.values
    Y = Ydummies_df.values 

    porcentagem_de_treino = 0.8
    porcentagem_de_teste = 0.1

    tamanho_de_treino = int(porcentagem_de_treino * len(Y))
    tamanho_de_teste = int(porcentagem_de_teste * len(Y))
    tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

    treino_dados = X[:tamanho_de_treino]
    treino_marcacoes = Y[:tamanho_de_treino]

    fim_de_teste = tamanho_de_treino + tamanho_de_teste

    teste_dados = X[tamanho_de_treino:fim_de_teste]
    teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

    validacao_dados = X[fim_de_teste:]
    validacao_marcacoes = Y[fim_de_teste:]

    resultados = {}
    modeloMultinomial = MultinomialNB()
    resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
    resultados[resultadoMultinomial] = modeloMultinomial

    modeloAdaBoost = AdaBoostClassifier()
    resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
    resultados[resultadoAdaBoost] = modeloAdaBoost

    modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0, max_iter=5000))
    resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
    resultados[resultadoOneVsRest] = modeloOneVsRest

    modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0, max_iter=5000))
    resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
    resultados[resultadoOneVsOne] = modeloOneVsOne

    vencedor = resultados[max(resultados)]
   
    teste_real(vencedor, validacao_dados, validacao_marcacoes)

    acerto_base = max(Counter(validacao_marcacoes).values())
    taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
    print("Taxa de acerto base: %f" % taxa_de_acerto_base)

    total_de_elementos = len(validacao_dados)
    print("Total de teste: %d" % total_de_elementos)

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = resultado == teste_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos 

    print("Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto))
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    print("Taxa de acerto do vencedor no mundo real: {0}".format(taxa_de_acerto))

start_process()