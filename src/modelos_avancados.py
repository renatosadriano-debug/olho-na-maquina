import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


def treinar_modelos_avancados(X_train_bal, y_train_bal, X_test, y_test):
    """
    Exploração adicional (bônus) - modelos avançados baseados em árvores.
    Por serem baseados em árvores, usam os dados CRUS (sem escalonamento).
    Treina Random Forest, XGBoost e LightGBM e compara a acurácia no teste.
    """
    modelos = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    }

    resultados = {}
    print("\n--- Modelos Avançados (dados crus) ---")
    for nome, modelo in modelos.items():
        modelo.fit(X_train_bal, y_train_bal)
        acc = accuracy_score(y_test, modelo.predict(X_test))
        resultados[nome] = acc
        print(f"{nome}: Acurácia no teste = {acc:.3f}")

    return resultados


def grafico_avancados(resultados):
    """Gráfico de barras comparando a acurácia dos modelos avançados."""
    nomes = list(resultados.keys())
    acuracias = list(resultados.values())

    plt.figure(figsize=(7, 4))
    barras = plt.bar(nomes, acuracias, color=['#8fbf9f', '#f0a868', '#7fa8d4'], width=0.6)
    plt.ylim(0, 1.15)
    plt.title('Acurácia no Teste - Modelos Avançados', pad=8)
    for barra in barras:
        plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.02,
                 f'{barra.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
    plt.show()