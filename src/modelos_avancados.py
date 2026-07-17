import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from src.saidas import salvar_figura


def treinar_modelos_avancados(X_train_bal, y_train_bal, X_test, y_test):
    """
    Exploração adicional - modelos avançados baseados em árvores (dados crus).
    Treina Random Forest, XGBoost e LightGBM.
    Retorna: dicionário de acurácias e dicionário dos modelos treinados.
    """
    modelos = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    }

    resultados = {}
    modelos_treinados = {}
    print("\n--- Modelos Avançados (dados crus) ---")
    for nome, modelo in modelos.items():
        modelo.fit(X_train_bal, y_train_bal)
        acc = accuracy_score(y_test, modelo.predict(X_test))
        resultados[nome] = acc
        modelos_treinados[nome] = modelo
        print(f"{nome}: Acurácia no teste = {acc:.3f}")

    return resultados, modelos_treinados


def grafico_avancados(resultados):
    """Barras comparando a acurácia dos modelos avançados."""
    nomes = list(resultados.keys())
    acuracias = list(resultados.values())

    plt.figure(figsize=(7, 4))
    barras = plt.bar(nomes, acuracias, color=['#8fbf9f', '#f0a868', '#7fa8d4'], width=0.6)
    plt.ylim(0, 1.15)
    plt.title('Acurácia no Teste - Modelos Avançados', pad=8)
    for barra in barras:
        plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.02,
                 f'{barra.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
    salvar_figura('07_modelos_avancados.png')


def grafico_linha_5_modelos(acc_knn, acc_arvore, resultados):
    """Gráfico de linhas cruzando a acurácia dos 5 modelos no teste."""
    nomes = ['KNN', 'Árvore', 'Random Forest', 'XGBoost', 'LightGBM']
    acuracias = [acc_knn, acc_arvore,
                 resultados['Random Forest'], resultados['XGBoost'], resultados['LightGBM']]

    plt.figure(figsize=(9, 5))
    plt.plot(nomes, acuracias, marker='o', linewidth=2, color='#2c7fb8')
    for i, acc in enumerate(acuracias):
        plt.text(i, acc + 0.004, f'{acc:.3f}', ha='center', fontweight='bold')
    plt.title('Comparação de Acurácia no Teste - 5 Modelos')
    plt.ylabel('Acurácia no teste')
    plt.grid(True, alpha=0.3)
    plt.ylim(min(acuracias) - 0.03, max(acuracias) + 0.03)
    salvar_figura('08_comparacao_5_modelos.png')


def veredito_geral(acc_knn, acc_arvore, resultados):
    """
    Veredito geral considerando TODOS os 5 modelos (inclui os avançados).
    Retorna o ranking (dict ordenado) e o nome do melhor modelo.
    """
    todos = {'KNN (K=3)': acc_knn, 'Árvore (depth=5)': acc_arvore}
    todos.update(resultados)

    ranking = dict(sorted(todos.items(), key=lambda x: x[1], reverse=True))
    melhor = next(iter(ranking))

    print("\n=== VEREDITO GERAL (incluindo modelos avançados) ===")
    for nome, acc in ranking.items():
        print(f"{nome}: {acc:.3f}")
    print(f"\nMelhor modelo geral: {melhor} ({ranking[melhor]:.3f})")

    return ranking, melhor


def grafico_importancia_avancados(modelos_treinados, X):
    """Importância das variáveis nos modelos avançados (normalizada em %)."""
    df_imp = pd.DataFrame(index=X.columns)
    for nome, modelo in modelos_treinados.items():
        imp = modelo.feature_importances_
        df_imp[nome] = imp / imp.sum() * 100   # normaliza para porcentagem

    # ordena pela média de importância entre os modelos
    ordem = df_imp.mean(axis=1).sort_values(ascending=False).index
    df_imp = df_imp.loc[ordem]

    df_imp.plot(kind='barh', figsize=(10, 6))
    plt.title('Importância das Variáveis - Modelos Avançados (%)')
    plt.xlabel('Importância (%)')
    plt.ylabel('Variáveis')
    plt.gca().invert_yaxis()   # mais importante no topo
    plt.legend(title='Modelo')
    salvar_figura('09_importancia_avancados.png')

    print("\nImportância das variáveis (modelos avançados, %):")
    print(df_imp.round(2))